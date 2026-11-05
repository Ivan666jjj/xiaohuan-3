# -*- coding: utf-8 -*-
"""实验 11：更贴近真实的残缺生成。
基线用的是「随机矩形/椭圆遮挡」，机械。
这里换成四种简牍常见残损：
  1. 边缘缺损：从图的某一侧咬掉一块
  2. 笔画断裂：细线横穿/竖穿（虫蛀、磨损）
  3. 污渍：不规则斑点，中灰填充
  4. 混合
在同一检索器上对比两种残缺生成的效果。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp11_damage')
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


def dmg_rect(a, p=0.30):
    """基线：随机矩形/椭圆。"""
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    if random.random() < 0.5:
        bw, bh = int(IMG*p), int(IMG*p*random.uniform(0.5, 1.5))
        x = random.randint(0, IMG-bw); y = random.randint(0, IMG-min(bh, IMG))
        ImageDraw.Draw(im).rectangle([x, y, x+bw, y+min(bh, IMG)], fill=255)
    else:
        bw = int(IMG*p); x = random.randint(0, IMG-bw); y = random.randint(0, max(1, IMG-bw))
        ImageDraw.Draw(im).ellipse([x, y, x+bw, y+bw], fill=255)
    return np.array(im)


def dmg_real(a, p=0.30):
    """四种真实残损随机组合。"""
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    dr = ImageDraw.Draw(im)
    kind = random.choice(['edge', 'break', 'stain', 'combo'])
    if kind in ('edge', 'combo'):
        # 从某侧咬掉不规则块
        side = random.choice(['l', 'r', 't', 'b'])
        w = int(IMG * random.uniform(0.15, 0.35))
        pts = []
        for i in range(6):
            t = IMG * i / 5
            j = random.uniform(0, w * 0.6)
            if side == 'l': pts.append((j, t))
            elif side == 'r': pts.append((IMG - j, t))
            elif side == 't': pts.append((t, j))
            else: pts.append((t, IMG - j))
        if side == 'l': pts += [(0, IMG), (0, 0)]
        elif side == 'r': pts += [(IMG, IMG), (IMG, 0)]
        elif side == 't': pts += [(IMG, 0), (0, 0)]
        else: pts += [(IMG, IMG), (0, IMG)]
        dr.polygon(pts, fill=255)
    if kind in ('break', 'combo'):
        # 细线断裂
        n = random.randint(2, 4)
        for _ in range(n):
            if random.random() < 0.5:
                y = random.randint(8, IMG-8)
                dr.line([(0, y), (IMG, y + random.randint(-6, 6))], fill=255,
                        width=random.randint(2, 4))
            else:
                x = random.randint(8, IMG-8)
                dr.line([(x, 0), (x + random.randint(-6, 6), IMG)], fill=255,
                        width=random.randint(2, 4))
    if kind in ('stain', 'combo'):
        # 不规则污渍，中灰
        for _ in range(random.randint(1, 3)):
            cx, cy = random.randint(10, IMG-10), random.randint(10, IMG-10)
            r = random.randint(6, 16)
            dr.ellipse([cx-r, cy-r, cx+r, cy+r], fill=random.randint(150, 200))
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
            loss = F.cross_entropy(model.head(model.embed(x)), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f'{tag} ep{ep} loss {tot/n:.4f}', flush=True)


@torch.no_grad()
def evaluate(model, dmg_fn):
    model.eval()
    Eg = []
    for i in range(0, len(tr), 128):
        Eg.append(F.normalize(model.embed(T([X[j] for j in tr[i:i+128]])), dim=1))
    Eg = torch.cat(Eg).numpy()
    random.seed(123)
    bad = [dmg_fn(X[j]) for j in te]
    hits = {k: 0 for k in (1, 5, 10, 20)}
    for q in range(0, len(bad), 128):
        Eq = F.normalize(model.embed(T(bad[q:q+128])), dim=1).numpy()
        sim = Eq @ Eg.T
        for r in range(len(Eq)):
            t = te[q+r]
            order = np.argsort(-sim[r])
            labs = [Y[tr[o]] for o in order if slips[tr[o]] != slips[t]]
            for k in hits:
                if Y[t] in labs[:k]: hits[k] += 1
    return {f'R{k}': round(v/len(te)*100, 1) for k, v in hits.items()}


res = {}
m = Net(len(classes))
train_model(m, 'Net ')
res['dmg_rect'] = evaluate(m, dmg_rect); print('矩形遮挡   ', res['dmg_rect'], flush=True)
res['dmg_real'] = evaluate(m, dmg_real); print('真实残损   ', res['dmg_real'], flush=True)

json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
