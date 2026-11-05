# -*- coding: utf-8 -*-
"""实验 2：域适应（AdaBN）。
不改权重，只用目标域（遮挡字）重算 BatchNorm 的 running stats，再看检索是否改善。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp2_adabn')
os.makedirs(OUT, exist_ok=True)
IMG = 64

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])

def damage(a, p=0.30):
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    bw = int(IMG*p); x = (IMG-bw)//2
    ImageDraw.Draw(im).rectangle([x, 0, x+bw, IMG], fill=255)
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
        s.head = nn.Linear(256, n); s.aux = nn.Linear(256, 182)
    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)
    def forward(s, x):
        return s.head(s.embed(x))

classes = open(os.path.join(HERE, 'cache', 'classes_300.txt'), encoding='utf-8').read().split('\n')
def load():
    net = Net(len(classes))
    ck = torch.load(os.path.join(HERE, 'xh2_aux_parts.pt'), map_location='cpu', weights_only=False)
    net.load_state_dict(ck['state'], strict=False)
    return net

def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5

@torch.no_grad()
def emb(net, idx, fn=lambda a: a):
    net.eval()
    out = []
    for s in range(0, len(idx), 128):
        b = idx[s:s+128]
        out.append(F.normalize(net.embed(T([fn(X[i]) for i in b])), dim=1))
    return torch.cat(out).numpy()

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

res = {}
net = load()
Eg = emb(net, tr)
res['base'] = ev(Eg, emb(net, te, damage), '基线（遮挡30%）  ')

# AdaBN：用目标域（遮挡的训练+测试图）重算 BN 统计量
net2 = load()
net2.train()
with torch.no_grad():
    for _ in range(3):
        for s in range(0, len(tr), 128):
            b = tr[s:s+128]
            net2(T([damage(X[i]) for i in b]))
        for s in range(0, len(te), 128):
            b = te[s:s+128]
            net2(T([damage(X[i]) for i in b]))
net2.eval()
torch.save({'state': net2.state_dict()}, os.path.join(OUT, 'adabn.pt'))

Eg2 = emb(net2, tr)
res['adabn'] = ev(Eg2, emb(net2, te, damage), 'AdaBN（遮挡30%）')
res['adabn_clean'] = ev(Eg2, emb(net2, te), 'AdaBN（完整图）  ')
json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
