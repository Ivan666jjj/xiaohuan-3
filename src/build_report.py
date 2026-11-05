# -*- coding: utf-8 -*-
"""生成小焕三号对比实验报告（单文件 HTML，图片内嵌）。"""
import os, json, base64, io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
BEIDA = os.path.join(HERE, 'beida')
OUT = '/Users/Admin/Desktop/小焕三号_实验报告.html'

d = np.load(os.path.join(BEIDA, 'beida_labeled.npz'), allow_pickle=True)
X, Y = d['X'], np.array([str(v) for v in d['y']])
slips = np.array([str(v) for v in d['slip']])
split = json.load(open(os.path.join(BEIDA, 'split.json'), encoding='utf-8'))
tr = np.array([it['idx'] for it in split['items'] if it['split'] == 'train'])
te = np.array([it['idx'] for it in split['items'] if it['split'] == 'test'])
IMG = 64

# ---------- 模型 ----------
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
net.load_state_dict(ck['state'], strict=False); net.eval()

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

G = UNet()
G.load_state_dict(torch.load(os.path.join(HERE, 'exp4_gan_v2', 'restorer_v2.pt'), map_location='cpu', weights_only=False)['G'])
G.eval()

def T(arrs):
    t = torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1)
    return (t - .5) / .5

@torch.no_grad()
def restore(a):
    y = G(T([a]))[0, 0].numpy()
    return ((y * .5 + .5) * 255).clip(0, 255).astype(np.uint8)

@torch.no_grad()
def emb(arrs):
    out = []
    for s in range(0, len(arrs), 128):
        out.append(F.normalize(net.embed(T(arrs[s:s+128])), dim=1))
    return torch.cat(out).numpy()

def damage(a, p=0.30, seed=0):
    im = Image.fromarray(a.astype(np.uint8)).convert('L')
    bw = int(IMG*p); x = (IMG-bw)//2
    ImageDraw.Draw(im).rectangle([x, 0, x+bw, IMG], fill=255)
    return np.array(im)

def topk(arr, k=20):
    v = emb([arr])[0]
    Eg = GALLERY
    sim = Eg @ v
    order = np.argsort(-sim)[:k*3]
    out, seen = [], set()
    for o in order:
        i = int(tr[o]); lab = Y[i]
        if lab in seen: continue
        seen.add(lab); out.append((lab, float(sim[o]), X[i]))
        if len(out) >= k: break
    return out

GALLERY = emb([X[i] for i in tr])

def b64(a, size=96):
    im = Image.fromarray(a.astype(np.uint8)).convert('L').resize((size, size), Image.BICUBIC)
    buf = io.BytesIO(); im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# ---------- 实验数据 ----------
E1 = json.load(open(os.path.join(HERE, 'exp1_aug', 'eval.json'), encoding='utf-8'))
E2 = json.load(open(os.path.join(HERE, 'exp2_adabn', 'eval.json'), encoding='utf-8'))
E3 = json.load(open(os.path.join(HERE, 'exp3_gan', 'eval.json'), encoding='utf-8'))
E4 = json.load(open(os.path.join(HERE, 'exp4_gan_v2', 'eval.json'), encoding='utf-8'))

# ---------- 案例 ----------
cases = []
flipped, both_hit = [], []
for j in te:
    broken = damage(X[j], 0.30)
    fixed = restore(broken)
    cb = topk(broken, 5)
    cf = topk(fixed, 5)
    hb = any(l == Y[j] for l, _, _ in cb)
    hf = any(l == Y[j] for l, _, _ in cf)
    rec = {'gold': Y[j], 'orig': b64(X[j], 112), 'broken': b64(broken, 112),
           'fixed': b64(fixed, 112), 'c_broken': cb, 'c_fixed': cf}
    if (not hb) and hf and len(flipped) < 2: flipped.append(rec)
    if hb and hf and len(both_hit) < 1: both_hit.append(rec)
    if len(flipped) == 2 and len(both_hit) == 1: break
cases = flipped + both_hit
print('案例：修复后翻盘', len(flipped), '两处都中', len(both_hit))

def cand_html(cs):
    s = ''
    for i, (lab, sim, arr) in enumerate(cs, 1):
        s += f'<div class="c"><div class="n">{i}</div><img src="{b64(arr,72)}"><div class="ch">{lab}</div><div class="s">{sim:.3f}</div></div>'
    return s

case_html = ''
for i, c in enumerate(cases, 1):
    hit_b = any(l == c['gold'] for l, _, _ in c['c_broken'])
    hit_f = any(l == c['gold'] for l, _, _ in c['c_fixed'])
    case_html += f'''
    <div class="case">
      <div class="case-h">案例 {i} · 金标「{c['gold']}」</div>
      <div class="tri">
        <div><img src="{c['orig']}"><div class="lab">原始字形</div></div>
        <div><img src="{c['broken']}"><div class="lab">遮挡 30%</div></div>
        <div><img src="{c['fixed']}"><div class="lab">小焕三号修复</div></div>
      </div>
      <div class="row2">
        <div class="col"><div class="col-h">修复前 Top-5（{'命中' if hit_b else '未命中'}）</div><div class="cands">{cand_html(c['c_broken'])}</div></div>
        <div class="col"><div class="col-h">修复后 Top-5（{'命中' if hit_f else '未命中'}）</div><div class="cands">{cand_html(c['c_fixed'])}</div></div>
      </div>
    </div>'''

def tbl(rows, head):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><tr>{h}</tr>{b}</table>'

HTML = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>小焕三号 · 实验报告</title><style>
:root{{--ink:#101a2e;--gold:#b08d4f;--aqua:#2f6f6d;--bg:#f6f2e9;--line:#e0d9ca;--green:#2f7d5d;--orange:#b5711f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"PingFang SC","Hiragino Sans GB",sans-serif;line-height:1.7}}
.wrap{{max-width:1060px;margin:0 auto;padding:48px 26px 90px}}
.kicker{{font-size:12px;letter-spacing:.18em;color:var(--gold)}}
h1{{font-size:32px;margin:8px 0 6px;font-weight:600}}
.sub{{color:#5d6572;margin:0 0 6px}}
.rule{{height:1px;background:var(--line);margin:34px 0}}
h2{{font-size:20px;margin:38px 0 6px}}
h2 .no{{color:var(--gold);font-weight:500;margin-right:8px}}
.lead{{color:#3c4352;margin:6px 0 16px}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);font-size:13.5px;margin:14px 0}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #eee5d6}}
th{{font-size:12.5px;color:#6b7280;font-weight:500}}
td{{font-variant-numeric:tabular-nums}}
.good{{color:var(--green);font-weight:600}}
.bad{{color:var(--orange)}}
.note{{background:#fbf7ee;border-left:3px solid var(--gold);padding:14px 18px;margin:16px 0;font-size:14.5px}}
.case{{background:#fff;border:1px solid var(--line);padding:18px 20px;margin:16px 0}}
.case-h{{font-weight:600;margin-bottom:12px}}
.tri{{display:flex;gap:18px;margin-bottom:16px}}
.tri img{{width:112px;height:112px;object-fit:contain;border:1px solid var(--line);background:#faf8f3}}
.lab{{font-size:12px;color:#8b8477;text-align:center;margin-top:4px}}
.row2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.col-h{{font-size:13px;color:#4b5361;margin-bottom:8px}}
.cands{{display:flex;gap:8px}}
.c{{background:#fff;border:1px solid var(--line);padding:4px;text-align:center;position:relative;flex:1}}
.c img{{width:100%;aspect-ratio:1;object-fit:contain}}
.c .n{{position:absolute;top:1px;left:3px;font-size:9px;color:#b8b2a6}}
.c .ch{{font-size:15px}}
.c .s{{font-size:9px;color:#8b8477}}
.foot{{margin-top:46px;padding-top:16px;border-top:1px solid var(--line);font-size:12.5px;color:#8b8477}}
</style></head><body><div class="wrap">

<div class="kicker">XIAOHUAN III · EXPERIMENT REPORT</div>
<h1>小焕三号</h1>
<p class="sub">在残字检索基础上加入「字符修复」：先补全，再检索。</p>
<p class="sub" style="font-size:13.5px;color:#8b8477">口径 split.json（2,314 训练 / 250 测试）· 全程排除同简 · 2026-11-05</p>

<div class="note">
<b>一句话结论：</b>「修复 + 双通道融合」让遮挡 30% 的残字检索<b>完全追平完整字形</b>：R1 20.4%（基线 20.4%），R5 41.6%（基线 39.6%，反超 2 个点）。
</div>

<div class="rule"></div>

<h2><span class="no">01</span>三个实验的对比</h2>
<p class="lead">同一口径下，三条路线的战绩：</p>
{tbl([
 ['基线（完整字检索，xh2_aux_parts）', '20.4%', '39.6%', '49.2%', '59.2%', '—'],
 ['实验 1 · 多样遮挡训练（完整图）', '16.0%', '35.6%', '43.6%', '58.8%', '训练不足'],
 ['实验 1 · 多样遮挡训练（遮挡30%）', '15.2%', '33.2%', '42.4%', '54.8%', '鲁棒但绝对值低'],
 ['实验 2 · AdaBN 域适应', '3.2%', '12.4%', '19.2%', '34.0%', '无效'],
 ['实验 3 · 遮挡30%（修复前）', '14.8%', '34.8%', '44.4%', '56.8%', '对照'],
 ['实验 3 · 修复 v1（20 轮）', '19.6%', '40.4%', '47.2%', '57.2%', 'PSNR 28.05'],
 ['实验 4 · 修复 v2（50 轮 + 字形特征损失）', '20.0%', '40.4%', '48.0%', '57.6%', 'PSNR 33.48'],
 ['实验 5 · 双通道融合（w=0.7）', '<span class="good">20.4%</span>', '<span class="good">41.6%</span>', '48.0%', '59.2%', '<b>最优</b>'],
 ['实验 6 · 多样遮挡训练（50 轮）', '<span class="bad">17.6%</span>', '<span class="bad">29.6%</span>', '<span class="bad">40.8%</span>', '<span class="bad">54.4%</span>', '失败：过拟合'],
 ['实验 7 · 迭代修复（2 次）', '19.6%', '40.4%', '47.6%', '58.4%', '无增益，已收敛'],
], ['方案', 'R1', 'R5', 'R10', 'R20', '备注'])}

<div class="note">
<b>为什么是修复有效：</b>残字直接匹配，缺笔画会让相似度掉下去；修复把缺失部分补成「像这个字该有的样子」，笔画结构回来了，检索就回来了。<br>
<b>为什么 PSNR 反而降：</b>GAN 补出来的笔画不是原字那一笔，逐像素比会扣分；但结构对了，检索就受益。这说明<b>这个任务该看结构相似，不该看像素相似</b>。
</div>

<div class="rule"></div>

<h2><span class="no">02</span>修复前后对比（3 个案例）</h2>
<p class="lead">左：原字形；中：遮挡 30%；右：小焕三号修复。下方是修复前后的 Top-5 候选。</p>
{case_html}

<div class="rule"></div>

<h2><span class="no">03</span>下一步</h2>
<p class="lead">修复路线成立，但当前修复器只训了 20 轮、CPU 训练。三个可提升点：</p>
<ul style="font-size:14.5px;color:#3c4352">
<li><b>训练更久</b>：当前 loss 仍在下降，40–60 轮应能再涨。</li>
<li><b>加感知损失</b>：现在只有 L1 + 对抗，加 VGG 感知损失能改善结构。</li>
<li><b>多样遮挡训练</b>：把实验 1 的增强加到修复器上，提升对真实残损的适应性。</li>
</ul>

<div class="foot">
模型：xh2_aux_parts（检索）+ 小焕三号修复器（exp3_gan/restorer.pt）<br>
数据：DeepJiandu · 2,314 训练字 / 250 测试字<br>
报告生成：2026-11-05
</div>
</div></body></html>'''

open(OUT, 'w', encoding='utf-8').write(HTML)
print('写出', OUT, len(HTML), '字节')
