# -*- coding: utf-8 -*-
"""实验 12：原图 + 小波 双分支特征融合。
分支 A：CNN(1 通道原图) → 256 维
分支 B：CNN(4 通道小波子带) → 256 维
拼接 512 维 → 分类头。
对比：单原图、单小波、融合。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp12_fusion')
os.makedirs(OUT, exist_ok=True)
IMG, EPOCHS, BS = 64, 20, 64

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])
classes = sorted(set(Y[tr]))
c2i = {c: i for i, c in enumerate(classes)}
tr = np.array([i for i in tr if Y[i] in c2i])
print(f'train {len(tr)} / {len(classes)} 类，test {len(te)}')


def haar2d(img):
    a = img[0::2, 0::2].astype(np.float32)
    b = img[0::2, 1::2].astype(np.float32)
    c = img[1::2, 0::2].astype(np.float32)
    e = img[1::2, 1::2].astype(np.float32)
    return ((a + b + c + e) / 4, (a + b - c - e) / 4,
            (a - b + c - e) / 4, (a - b - c + e) / 4)


def wav_pack(arrs):
    return np.stack([np.stack(haar2d(a)) for a in arrs])


def blk(a, b):
    return nn.Sequential(
        nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
        nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
        nn.MaxPool2d(2))


class NetFusion(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.fa = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.fb = nn.Sequential(blk(4, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.gate = nn.Linear(512, 2)
        s.head = nn.Linear(512, n)
        s.aux = nn.Linear(512, 182)

    def embed(s, xo, xw):
        za = F.adaptive_avg_pool2d(s.fa(xo), 1).flatten(1)
        zb = F.adaptive_avg_pool2d(s.fb(xw), 1).flatten(1)
        g = torch.softmax(s.gate(torch.cat([za, zb], 1)), 1)
        return torch.cat([za * g[:, :1], zb * g[:, 1:]], 1)


class NetImage(nn.Module):
    def __init__(s, n, ch=1):
        super().__init__()
        s.f = nn.Sequential(blk(ch, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n); s.aux = nn.Linear(256, 182)

    def embed(s, xo, xw=None):
        return F.adaptive_avg_pool2d(s.f(xo), 1).flatten(1)


def prep(img_arrs, wav_arrs):
    xo = torch.from_numpy(np.stack(img_arrs)).float().div_(255.).unsqueeze(1)
    xo = (xo - .5) / .5
    xw = torch.from_numpy(np.stack(wav_arrs)).float() / 127.5 - 1.0
    return xo, xw


def cache(split_idx):
    return (X[split_idx], wav_pack(X[split_idx]))


TR_I, TR_W = cache(tr)
TE_I, TE_W = cache(te)


def train_model(model, tag, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(1, epochs + 1):
        model.train()
        idx = list(range(len(tr))); random.shuffle(idx)
        tot = n = 0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            xo, xw = prep(TR_I[b], TR_W[b])
            y = torch.tensor([c2i[Y[tr[j]]] for j in b])
            loss = F.cross_entropy(model.head(model.embed(xo, xw)), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f'{tag} ep{ep} loss {tot/n:.4f}', flush=True)


@torch.no_grad()
def evaluate(model):
    model.eval()
    Eg = []
    for i in range(0, len(tr), 128):
        xo, xw = prep(TR_I[i:i+128], TR_W[i:i+128])
        Eg.append(F.normalize(model.embed(xo, xw), dim=1))
    Eg = torch.cat(Eg).numpy()
    hits = {k: 0 for k in (1, 5, 10, 20)}
    for q in range(0, len(te), 128):
        xo, xw = prep(TE_I[q:q+128], TE_W[q:q+128])
        Eq = F.normalize(model.embed(xo, xw), dim=1).numpy()
        sim = Eq @ Eg.T
        for r in range(len(Eq)):
            t = te[q+r]
            order = np.argsort(-sim[r])
            labs = [Y[tr[o]] for o in order if slips[tr[o]] != slips[t]]
            for k in hits:
                if Y[t] in labs[:k]: hits[k] += 1
    return {f'R{k}': round(v/len(te)*100, 1) for k, v in hits.items()}


res = {}
m = NetFusion(len(classes))
print('参数', sum(p.numel() for p in m.parameters()))
train_model(m, 'Fusion')
res['fusion'] = evaluate(m)
print('融合     ', res['fusion'], flush=True)

json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
