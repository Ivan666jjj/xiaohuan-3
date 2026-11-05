# -*- coding: utf-8 -*-
"""实验 9：Transformer 检索器。
CNN 提特征 → 展平成 token 序列 → 1 层 Transformer encoder → 池化 → 分类。
其余（数据、轮数、损失）与基线完全一致，只换检索头。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp9_transformer')
os.makedirs(OUT, exist_ok=True)
IMG, EPOCHS, BS = 64, 20, 64
DEV = 'cpu'

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


class NetTr(nn.Module):
    """CNN 主干 + 1 层 Transformer。"""

    def __init__(s, n, d_model=256):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512,
                                         dropout=0.1, batch_first=True)
        s.tr = nn.TransformerEncoder(enc, num_layers=1)
        s.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # 可学习的 [CLS]
        s.head = nn.Linear(d_model, n)
        s.aux = nn.Linear(d_model, 182)

    def embed(s, x):
        f = s.f(x)                      # B,256,4,4
        B = f.shape[0]
        seq = f.flatten(2).transpose(1, 2)   # B,16,256
        cl = s.cls.expand(B, -1, -1)
        seq = torch.cat([cl, seq], 1)        # B,17,256
        out = s.tr(seq)
        return out[:, 0]                     # [CLS] 作为特征


class NetCNN(nn.Module):
    """基线：纯 CNN。"""

    def __init__(s, n):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(True),
                nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 256))
        s.head = nn.Linear(256, n)
        s.aux = nn.Linear(256, 182)

    def embed(s, x):
        return F.adaptive_avg_pool2d(s.f(x), 1).flatten(1)


def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5


def train_model(model, tag, epochs=EPOCHS):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(1, epochs + 1):
        model.train()
        idx = list(tr); random.shuffle(idx)
        tot = n = 0
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            x = T([X[j] for j in b])
            y = torch.tensor([c2i[Y[j]] for j in b])
            z = model.embed(x)
            logits = model.head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f'{tag} ep{ep} loss {tot/n:.4f}', flush=True)
    return model


@torch.no_grad()
def evaluate(model):
    model.eval()
    Eg = []
    for i in range(0, len(tr), 128):
        Eg.append(F.normalize(model.embed(T([X[j] for j in tr[i:i + 128]])), dim=1))
    Eg = torch.cat(Eg).numpy()
    hits = {k: 0 for k in (1, 5, 10, 20)}
    for q in range(0, len(te), 128):
        b = te[q:q + 128]
        Eq = F.normalize(model.embed(T([X[j] for j in b])), dim=1).numpy()
        sim = Eq @ Eg.T
        for r, t in enumerate(b):
            order = np.argsort(-sim[r])
            labs = [Y[tr[o]] for o in order if slips[tr[o]] != slips[t]]
            for k in hits:
                if Y[t] in labs[:k]: hits[k] += 1
    return {f'R{k}': round(v / len(te) * 100, 1) for k, v in hits.items()}


res = {}
m1 = NetCNN(len(classes))
print('参数 CNN', sum(p.numel() for p in m1.parameters()))
train_model(m1, 'CNN  ')
res['cnn'] = evaluate(m1); print('CNN  ', res['cnn'], flush=True)

m2 = NetTr(len(classes))
print('参数 TR ', sum(p.numel() for p in m2.parameters()))
train_model(m2, 'Trans')
res['transformer'] = evaluate(m2); print('Trans', res['transformer'], flush=True)

json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
