# -*- coding: utf-8 -*-
"""实验 10：小波频域特征。
输入从「单通道原图」改为「Haar 小波 4 子带」（LL/LH/HL/HH），
让 CNN 第一层直接看到频域信息。其余与基线一致。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp10_wavelet')
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
    """单级 Haar 小波，返回 4 个 32x32 子带。"""
    a = img[0::2, 0::2].astype(np.float32)
    b = img[0::2, 1::2].astype(np.float32)
    c = img[1::2, 0::2].astype(np.float32)
    e = img[1::2, 1::2].astype(np.float32)
    return ((a + b + c + e) / 4, (a + b - c - e) / 4,
            (a - b + c - e) / 4, (a - b - c + e) / 4)


def wav_pack(arrs):
    """返回 N,4,32,32"""
    out = []
    for a in arrs:
        LL, LH, HL, HH = haar2d(a)
        out.append(np.stack([LL, LH, HL, HH]))
    return np.stack(out)


class NetWav(nn.Module):
    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(4, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n); s.aux = nn.Linear(256, 182)

    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)


class NetCNN(nn.Module):
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


def prep(arrs, wav=False):
    if wav:
        t = torch.from_numpy(wav_pack(arrs)).float()
        return (t / 127.5) - 1.0
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5


def train_model(model, tag, wav, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(1, epochs + 1):
        model.train()
        idx = list(tr); random.shuffle(idx)
        tot = n = 0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = prep([X[j] for j in b], wav)
            y = torch.tensor([c2i[Y[j]] for j in b])
            loss = F.cross_entropy(model.head(model.embed(x)), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f'{tag} ep{ep} loss {tot/n:.4f}', flush=True)


@torch.no_grad()
def evaluate(model, wav):
    model.eval()
    Eg = []
    for i in range(0, len(tr), 128):
        Eg.append(F.normalize(model.embed(prep([X[j] for j in tr[i:i+128]], wav)), dim=1))
    Eg = torch.cat(Eg).numpy()
    hits = {k: 0 for k in (1, 5, 10, 20)}
    for q in range(0, len(te), 128):
        b = te[q:q+128]
        Eq = F.normalize(model.embed(prep([X[j] for j in b], wav)), dim=1).numpy()
        sim = Eq @ Eg.T
        for r, t in enumerate(b):
            order = np.argsort(-sim[r])
            labs = [Y[tr[o]] for o in order if slips[tr[o]] != slips[t]]
            for k in hits:
                if Y[t] in labs[:k]: hits[k] += 1
    return {f'R{k}': round(v/len(te)*100, 1) for k, v in hits.items()}


res = {}
m1 = NetCNN(len(classes))
train_model(m1, 'CNN ', False)
res['cnn'] = evaluate(m1, False); print('CNN    ', res['cnn'], flush=True)

m2 = NetWav(len(classes))
train_model(m2, 'Wav ', True)
res['wavelet'] = evaluate(m2, True); print('Wavelet', res['wavelet'], flush=True)

json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
