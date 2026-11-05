# -*- coding: utf-8 -*-
"""实验 4：修复器 v2。
改进：训练 50 轮 + 字形特征损失（用 xh2_aux_parts 的 embedding 做感知损失）+ 学习率衰减。
"""
import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = os.path.join(HERE, 'exp4_gan_v2')
os.makedirs(OUT, exist_ok=True)
DEV = 'cpu'
IMG = 64

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])

def damage(a, rng, p=None):
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    r = p if p else rng.uniform(0.10, 0.50)
    if rng.random() < 0.5:
        bw, bh = int(IMG*r), int(IMG*r*rng.uniform(0.5, 1.5))
        x = rng.randint(0, IMG-bw); y = rng.randint(0, IMG-min(bh, IMG))
        ImageDraw.Draw(im).rectangle([x, y, x+bw, y+min(bh, IMG)], fill=255)
    else:
        bw = int(IMG*r)
        x = rng.randint(0, IMG-bw); y = rng.randint(0, max(1, IMG-bw))
        ImageDraw.Draw(im).ellipse([x, y, x+bw, y+bw], fill=255)
    return np.array(im)

class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        def down(a, b): return nn.Sequential(
            nn.Conv2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.LeakyReLU(0.2, True))
        def up(a, b): return nn.Sequential(
            nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(True))
        s.d1 = nn.Sequential(nn.Conv2d(1, c, 3, 1, 1), nn.ReLU(True))
        s.d2, s.d3, s.d4 = down(c, c*2), down(c*2, c*4), down(c*4, c*8)
        s.u1, s.u2, s.u3 = up(c*8, c*4), up(c*8, c*2), up(c*4, c)
        s.out = nn.Conv2d(c*2, 1, 3, 1, 1)
    def forward(s, x):
        d1 = s.d1(x); d2 = s.d2(d1); d3 = s.d3(d2); d4 = s.d4(d3)
        u1 = s.u1(d4); u2 = s.u2(torch.cat([u1, d3], 1))
        u3 = s.u3(torch.cat([u2, d2], 1))
        return torch.tanh(s.out(torch.cat([u3, d1], 1)))

class PatchD(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv2d(2, c, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c, c*2, 4, 2, 1), nn.BatchNorm2d(c*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c*2, c*4, 4, 2, 1), nn.BatchNorm2d(c*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(c*4, 1, 3, 1, 1))
    def forward(s, a, b):
        return s.net(torch.cat([a, b], 1))

# 冻结的字形特征网络（做感知损失）
class FeatNet(nn.Module):
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
feat = FeatNet(len(classes))
ck = torch.load(os.path.join(HERE, 'xh2_aux_parts.pt'), map_location='cpu', weights_only=False)
feat.load_state_dict(ck['state'], strict=False)
feat.eval()
for p in feat.parameters(): p.requires_grad = False

def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5

rng = random.Random(0)
torch.manual_seed(0)
G, D = UNet(), PatchD()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
schedG = torch.optim.lr_scheduler.StepLR(optG, step_size=20, gamma=0.5)
schedD = torch.optim.lr_scheduler.StepLR(optD, step_size=20, gamma=0.5)
bce = nn.BCEWithLogitsLoss()

EPOCHS = 50
print(f'训练修复器 v2：{len(tr)} 样本 × {EPOCHS} 轮（L1 + 对抗 + 字形特征）', flush=True)
for ep in range(EPOCHS):
    idx = list(tr); random.shuffle(idx)
    gl = dl = 0.0; n = 0
    for i in range(0, len(idx), 32):
        b = idx[i:i+32]
        real = T([X[j] for j in b])
        broken = T([damage(X[j], rng) for j in b])
        fake = G(broken)
        optD.zero_grad()
        d_real = D(broken, real); d_fake = D(broken, fake.detach())
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        loss_d.backward(); optD.step()
        optG.zero_grad()
        d_g = D(broken, fake)
        l_adv = bce(d_g, torch.ones_like(d_g))
        l_l1 = F.l1_loss(fake, real)
        l_feat = F.l1_loss(feat.embed(fake), feat.embed(real))
        loss_g = l_adv + 100 * l_l1 + 10 * l_feat
        loss_g.backward(); optG.step()
        gl += loss_g.item(); dl += loss_d.item(); n += 1
    schedG.step(); schedD.step()
    if (ep + 1) % 10 == 0:
        print(f'ep{ep+1} G {gl/n:.3f} D {dl/n:.3f}', flush=True)

torch.save({'G': G.state_dict()}, os.path.join(OUT, 'restorer_v2.pt'))
print('saved', os.path.join(OUT, 'restorer_v2.pt'))

# ---------- 评估 ----------
G.eval()
def restore(a):
    with torch.no_grad():
        y = G(T([a]))[0, 0].numpy()
    return ((y * .5 + .5) * 255).clip(0, 255).astype(np.uint8)

@torch.no_grad()
def emb(arrs):
    out = []
    for s in range(0, len(arrs), 128):
        out.append(F.normalize(feat.embed(T(arrs[s:s+128])), dim=1))
    return torch.cat(out).numpy()

Eg = emb([X[i] for i in tr])
def ev(Eq, label):
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

rr = random.Random(42)
before = [damage(X[j], rr, p=0.30) for j in te]
after = [restore(b) for b in before]

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 10 * np.log10(255**2 / max(mse, 1e-9))

p1 = np.mean([psnr(before[k], X[j]) for k, j in enumerate(te)])
p2 = np.mean([psnr(after[k], X[j]) for k, j in enumerate(te)])
print(f'\nPSNR {p1:.2f} → {p2:.2f} dB')
res = {'psnr_before': round(float(p1), 2), 'psnr_after': round(float(p2), 2)}
res['broken'] = ev(emb(before), '遮挡30%（修复前）')
res['restored'] = ev(emb(after), 'v2 修复后      ')
json.dump(res, open(os.path.join(OUT, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
print('已写', os.path.join(OUT, 'eval.json'))
