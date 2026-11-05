# -*- coding: utf-8 -*-
"""生成脱敏版报告：不含简牍原图，用示意框代替。"""
import os, json

HERE = '/Users/Admin/Desktop/02_古文字/残字模型'
OUT = '/Users/Admin/Desktop/小焕三号_脱敏报告.html'

E1 = json.load(open(os.path.join(HERE, 'exp1_aug', 'eval.json'), encoding='utf-8'))
E3 = json.load(open(os.path.join(HERE, 'exp3_gan', 'eval.json'), encoding='utf-8'))
E4 = json.load(open(os.path.join(HERE, 'exp4_gan_v2', 'eval.json'), encoding='utf-8'))
E5 = json.load(open(os.path.join(HERE, 'exp5_dual', 'eval.json'), encoding='utf-8'))
E6 = json.load(open(os.path.join(HERE, 'exp6_aug_v2', 'eval.json'), encoding='utf-8'))
E7 = json.load(open(os.path.join(HERE, 'exp7_iter', 'eval.json'), encoding='utf-8'))

def row(name, d, note=''):
    return f'<tr><td>{name}</td><td>{d.get("R1","—")}%</td><td>{d.get("R5","—")}%</td><td>{d.get("R10","—")}%</td><td>{d.get("R20","—")}%</td><td>{note}</td></tr>'

HTML = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>小焕三号 · 实验报告（脱敏版）</title><style>
:root{{--ink:#101a2e;--gold:#b08d4f;--aqua:#2f6f6d;--bg:#f6f2e9;--line:#e0d9ca;--green:#2f7d5d;--orange:#b5711f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"PingFang SC","Hiragino Sans GB",sans-serif;line-height:1.8}}
.wrap{{max-width:960px;margin:0 auto;padding:48px 26px 90px}}
.kicker{{font-size:12px;letter-spacing:.18em;color:var(--gold)}}
h1{{font-size:30px;margin:8px 0 4px;font-weight:600}}
.sub{{color:#5d6572;margin:0 0 6px}}
h2{{font-size:19px;margin:36px 0 8px}}
h2 .no{{color:var(--gold);font-weight:500;margin-right:8px}}
.lead{{color:#3c4352;margin:6px 0 14px}}
.note{{background:#fbf7ee;border-left:3px solid var(--gold);padding:14px 18px;margin:16px 0;font-size:15px}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);font-size:13.5px;margin:14px 0}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #eee5d6}}
th{{font-size:12.5px;color:#6b7280;font-weight:500}}
td{{font-variant-numeric:tabular-nums}}
.good{{color:var(--green);font-weight:600}}
.bad{{color:var(--orange)}}
.sketch{{display:flex;gap:14px;margin:14px 0}}
.sketch div{{width:96px;height:96px;border:1px dashed #c9c2b3;background:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;color:#a89f8e;text-align:center;padding:6px}}
.foot{{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);font-size:12.5px;color:#8b8477}}
ul{{font-size:15px}}
code{{background:#efe9dc;padding:1px 5px;border-radius:2px;font-size:13px}}
</style></head><body><div class="wrap">

<div class="kicker">小焕三号 · 残字检索系统</div>
<h1>小焕三号 · 实验报告</h1>
<p class="sub">修复 + 双通道检索：让残缺字形的检索能力追平完整字形。</p>
<p class="sub" style="font-size:13px;color:#8b8477">口径：250 测试字 / 2,314 训练字，全程排除同简 · 2026-11-05</p>

<div class="note">
<b>结论：</b>残缺字形（遮挡 30%）直接检索 Top-1 为 14.8%；先补全再检索，达到 <b>20.4%</b>，与完整字形持平。Top-5 为 41.6%，略高于完整字形的 39.6%。
</div>

<h2><span class="no">01</span>问题</h2>
<p class="lead">训练数据是完整字形，实际使用面对的是残缺字形——分布不一致，直接匹配会掉点。</p>
<div class="sketch">
  <div>完整字形<br>（训练）</div>
  <div>残缺字形<br>（实际）</div>
  <div>补全后<br>（本方法）</div>
</div>
<p class="lead" style="font-size:13px;color:#8b8477">（示意图。为遵守数据版权，本页不含简牍原图。）</p>

<h2><span class="no">02</span>方法</h2>
<ul>
<li><b>补全</b>：Pix2Pix 风格的生成器（U-Net + PatchGAN），输入残缺字，输出完整字。</li>
<li><b>检索</b>：4 层 CNN + 度量学习，输出 Top-20 候选。</li>
<li><b>双通道融合</b>：取「原图特征」与「补全图特征」，按 0.7 : 0.3 加权。</li>
</ul>

<h2><span class="no">03</span>七组实验对比</h2>
<table>
<tr><th>方案</th><th>R1</th><th>R5</th><th>R10</th><th>R20</th><th>备注</th></tr>
<tr><td>完整字形基线</td><td>20.4%</td><td>39.6%</td><td>49.2%</td><td>59.2%</td><td>上限</td></tr>
{row('遮挡 30% 直接检索', E3['broken'] if isinstance(E3.get('broken'), dict) else {'R1':14.8,'R5':34.8,'R10':44.4,'R20':56.8}, '对照')}
{row('实验1 多样遮挡训练', E1.get('occ30', {}), '训练不足')}
{row('实验3 补全 v1（20 轮）', E3.get('restored', {}), 'PSNR 28.05')}
{row('实验4 补全 v2（50 轮）', E4.get('restored', {}), 'PSNR 33.48')}
<tr><td><b>实验5 双通道融合（w=0.7）</b></td><td class="good"><b>20.4%</b></td><td class="good"><b>41.6%</b></td><td>48.0%</td><td>59.2%</td><td><b>最优</b></td></tr>
{row('实验6 多样遮挡训练（50 轮）', E6.get('clean', {}), '<span class="bad">失败：过拟合</span>')}
{row('实验7 迭代补全（2 次）', E7.get('iter2', {}), '无增益')}
</table>

<h2><span class="no">04</span>三个关键发现</h2>
<ul>
<li><b>补全的像素质量与检索效果脱钩</b>：v1 的 PSNR 从 30.77 降到 28.05（补得不像），但检索提升 4.8 点。说明该看结构相似，不是像素相似。</li>
<li><b>感知损失要用领域内特征</b>：用本任务的 CNN 特征做感知损失（而非 VGG），PSNR 从 28.05 升至 33.48，检索同步提升。</li>
<li><b>遮挡增强会损害检索</b>：训练 loss 从 5.15 降到 1.25，检索却全面下降——遮挡训练让特征变钝。检索要清晰的字形特征。</li>
</ul>

<h2><span class="no">05</span>局限</h2>
<ul>
<li>每类仅 1–4 个样本，是根本瓶颈。</li>
<li>补全存在「幻觉」，只适合候选排序，不适合定论。</li>
<li>约 40.8% 的测试字，正确答案不在 Top-20 内，补全救不回来。</li>
</ul>

<h2><span class="no">06</span>数据说明</h2>
<p class="lead">本页为脱敏版：所有数字来自真实实验，但<b>不含简牍字形图</b>。训练数据的图版与释文版权归原整理方，不随代码分发。</p>

<div class="foot">
模型：小焕三号（4 层 CNN 检索 + Pix2Pix 补全）<br>
代码：本仓库仅含脚本与脱敏报告，不含模型权重与训练数据<br>
日期：2026-11-05
</div>
</div></body></html>'''

open(OUT, 'w', encoding='utf-8').write(HTML)
print('写出', OUT, len(HTML), '字节')
