# -*- coding: utf-8 -*-
"""残字候选检索服务：零依赖（http.server）。
启动：python3 server.py   然后浏览器打开 http://127.0.0.1:8765
"""
import os, json, base64, io, re
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
PORT = 8765

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])

class Net(nn.Module):
    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n); s.aux = nn.Linear(256, 182)
    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)

classes = open(os.path.join(HERE, 'cache', 'classes_300.txt'), encoding='utf-8').read().split('\n')
net = Net(len(classes))
ck = torch.load(os.path.join(HERE, 'xh2_aux_parts.pt'), map_location='cpu', weights_only=False)
net.load_state_dict(ck['state'], strict=False)
net.eval()

@torch.no_grad()
def emb_arr(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    t = (t - .5) / .5
    return F.normalize(net.embed(t), dim=1).numpy()

GALLERY = emb_arr([X[i] for i in tr])
print(f'gallery {len(tr)} 已编码', flush=True)

def b64_to_arr(s):
    s = re.sub(r'^data:image/\w+;base64,', '', s)
    im = Image.open(io.BytesIO(base64.b64decode(s))).convert('L').resize((64, 64), Image.BICUBIC)
    return np.array(im)

def arr_to_b64(a, size=96):
    im = Image.fromarray(a.astype(np.uint8)).convert('L').resize((size, size), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

def search(arr, k=20):
    v = emb_arr([arr])[0]
    sim = GALLERY @ v
    order = np.argsort(-sim)[:k*3]
    out, seen = [], set()
    for o in order:
        i = int(tr[o])
        lab = Y[i]
        if lab in seen: continue
        seen.add(lab)
        out.append({'char': lab, 'sim': round(float(sim[o]), 3),
                    'img': arr_to_b64(X[i])})
        if len(out) >= k: break
    return out

INDEX = open(os.path.join(HERE, 'web', 'index.html'), encoding='utf-8').read() \
    if os.path.exists(os.path.join(HERE, 'web', 'index.html')) else '<h1>缺少 index.html</h1>'

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            b = INDEX.encode('utf-8')
            self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        if self.path != '/predict':
            self.send_response(404); self.end_headers(); return
        n = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
            arr = b64_to_arr(body['image'])
            res = search(arr, int(body.get('k', 20)))
            payload = json.dumps({'ok': True, 'cands': res}, ensure_ascii=False).encode()
        except Exception as e:
            payload = json.dumps({'ok': False, 'err': str(e)}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)

if __name__ == '__main__':
    print(f'服务启动：http://127.0.0.1:{PORT}', flush=True)
    HTTPServer(('127.0.0.1', PORT), H).serve_forever()
