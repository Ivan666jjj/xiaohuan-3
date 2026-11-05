# -*- coding: utf-8 -*-
"""实验 1：多样缺失掩码训练。
训练时随机遮挡（矩形/椭圆，10–50%），测试时同样施加 30% 遮挡，
与不加增强的基线对比检索能力。
"""
import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp1_aug')
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])
classes = [c for c in open(os.path.join(HERE,'cache','classes_300.txt'),encoding='utf-8').read().split('\n') if c]
c2i = {c: i for i, c in enumerate(classes)}
tr = np.array([i for i in tr if Y[i] in c2i])
print(f'train {len(tr)} / {len(classes)} 类，test {len(te)}')

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
IMG = 64

def occlude(a, rng, lo=0.10, hi=0.50):
    """随机形状遮挡：覆盖 lo~hi 的宽度。"""
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    w = h = IMG
    r = rng.uniform(lo, hi)
    if rng.random() < 0.5:   # 矩形
        bw, bh = int(w * r), int(h * r * rng.uniform(0.5, 1.5))
        x = rng.randint(0, w - bw); y = rng.randint(0, h - min(bh, h))
        ImageDraw.Draw(im).rectangle([x, y, x + bw, y + min(bh, h)], fill=255)
    else:                     # 椭圆
        bw = int(w * r)
        x = rng.randint(0, w - bw); y = rng.randint(0, max(1, h - bw))
        ImageDraw.Draw(im).ellipse([x, y, x + bw, y + bw], fill=255)
    return np.array(im)

class Net(nn.Module):
    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n)
    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)
    def forward(s, x):
        return s.head(s.embed(x))

def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return ((t - .5) / .5).to(DEV)

rng = random.Random(0)
torch.manual_seed(0); np.random.seed(0)
net = Net(len(classes))
_ck = torch.load(os.path.join(HERE, 'xh2_aux_parts.pt'), map_location='cpu', weights_only=False)
_miss = net.load_state_dict(_ck['state'], strict=False)
print('预训练载入：缺失', len(_miss.missing_keys), '多余', len(_miss.unexpected_keys))
net = net.to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

EPOCHS = 20
for ep in range(EPOCHS):
    net.train()
    idx = list(tr); random.shuffle(idx)
    tot, n = 0.0, 0
    for i in range(0, len(idx), 64):
        b = idx[i:i+64]
        imgs = [occlude(X[j], rng) if rng.random() < 0.85 else X[j] for j in b]
        x = T(imgs)
        y = torch.tensor([c2i[Y[j]] for j in b]).to(DEV)
        opt.zero_grad()
        loss = crit(net(x), y)
        loss.backward(); opt.step()
        tot += loss.item(); n += 1
    print(f'ep{ep+1} loss {tot/n:.4f}', flush=True)

torch.save({'state': net.state_dict(), 'classes': classes}, os.path.join(OUT, 'aug_model.pt'))

# 评估：测试时施加 30% 遮挡（固定 seed，可复现）
@torch.no_grad()
def emb(idx, occl=None):
    net.eval()
    out = []
    for s in range(0, len(idx), 128):
        b = idx[s:s+128]
        imgs = [occlude(X[j], occl) if occl else X[j] for j in b]
        out.append(F.normalize(net.embed(T(imgs)), dim=1))
    return torch.cat(out).cpu().numpy()

def ev(Eg, Eq, label):
    sim = Eq @ Eg.T
    hits = {k: 0 for k in (1, 5, 10, 20)}
    for qi, t in enumerate(te):
        order = np.argsort(-sim[qi])
        labs = [Y[tr[o]] for o in order if slips[tr[o]] != slips[t]]
        for k in hits:
            if Y[t] in labs[:k]: hits[k] += 1
    n = len(te)
    r = {f'R{k}': round(v/n*100, 1) for k, v in hits.items()}
    print(f'{label}: ' + '  '.join(f'{k} {v}%' for k, v in r.items()))
    return r

Eg = emb(tr)
res = {}
res['clean'] = ev(Eg, emb(te), '测试(完整)   ')
o = random.Random(42)
res['occ30'] = ev(Eg, emb(te, o), '测试(遮挡30%)')
json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
