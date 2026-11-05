# -*- coding: utf-8 -*-
"""实验 7：迭代修复。修复一次 vs 修复两次。"""
import os, json, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
HERE='/Users/Admin/Desktop/02_古文字/残字模型'; BEIDA=os.path.join(HERE,'beida')
OUT=os.path.join(HERE,'exp7_iter'); os.makedirs(OUT,exist_ok=True); IMG=64
d=np.load(os.path.join(BEIDA,'beida_labeled.npz'),allow_pickle=True)
X,Y=d['X'],np.array([str(v) for v in d['y']]); slips=np.array([str(v) for v in d['slip']])
split=json.load(open(os.path.join(BEIDA,'split.json'),encoding='utf-8'))
tr=np.array([it['idx'] for it in split['items'] if it['split']=='train'])
te=np.array([it['idx'] for it in split['items'] if it['split']=='test'])
class Net(nn.Module):
    def __init__(s,n):
        super().__init__()
        def blk(a,b): return nn.Sequential(nn.Conv2d(a,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU(True),
            nn.Conv2d(b,b,3,1,1),nn.BatchNorm2d(b),nn.ReLU(True),nn.MaxPool2d(2))
        s.f=nn.Sequential(blk(1,32),blk(32,64),blk(64,128),blk(128,256))
        s.head=nn.Linear(256,n); s.aux=nn.Linear(256,182)
    def embed(s,x): return F.adaptive_avg_pool2d(s.f(x),1).flatten(1)
class UNet(nn.Module):
    def __init__(s,c=32):
        super().__init__()
        def down(a,b): return nn.Sequential(nn.Conv2d(a,b,4,2,1),nn.BatchNorm2d(b),nn.LeakyReLU(0.2,True))
        def up(a,b): return nn.Sequential(nn.ConvTranspose2d(a,b,4,2,1),nn.BatchNorm2d(b),nn.ReLU(True))
        s.d1=nn.Sequential(nn.Conv2d(1,c,3,1,1),nn.ReLU(True))
        s.d2,s.d3,s.d4=down(c,c*2),down(c*2,c*4),down(c*4,c*8)
        s.u1,s.u2,s.u3=up(c*8,c*4),up(c*8,c*2),up(c*4,c)
        s.out=nn.Conv2d(c*2,1,3,1,1)
    def forward(s,x):
        d1=s.d1(x); d2=s.d2(d1); d3=s.d3(d2); d4=s.d4(d3)
        u1=s.u1(d4); u2=s.u2(torch.cat([u1,d3],1)); u3=s.u3(torch.cat([u2,d2],1))
        return torch.tanh(s.out(torch.cat([u3,d1],1)))
classes=open(os.path.join(HERE,'cache','classes_300.txt'),encoding='utf-8').read().split('\n')
net=Net(len(classes)); ck=torch.load(os.path.join(HERE,'xh2_aux_parts.pt'),map_location='cpu',weights_only=False)
net.load_state_dict(ck['state'],strict=False); net.eval()
G=UNet(); G.load_state_dict(torch.load(os.path.join(HERE,'exp4_gan_v2','restorer_v2.pt'),map_location='cpu',weights_only=False)['G']); G.eval()
def T(arrs):
    t=torch.from_numpy(np.stack(arrs)).float().div_(255.).unsqueeze(1); return (t-.5)/.5
@torch.no_grad()
def restore(arrs):
    out=[]
    for s in range(0,len(arrs),64):
        y=G(T(arrs[s:s+64])).numpy(); out.append(((y*.5+.5)*255).clip(0,255).astype(np.uint8)[:,0])
    return np.concatenate(out)
@torch.no_grad()
def emb(arrs):
    out=[]
    for s in range(0,len(arrs),128): out.append(F.normalize(net.embed(T(arrs[s:s+128])),dim=1))
    return torch.cat(out).numpy()
rng=random.Random(42)
def damage(a,p=0.30):
    im=Image.fromarray(a.astype(np.uint8)).convert('L')
    if rng.random()<0.5:
        bw,bh=int(IMG*p),int(IMG*p*rng.uniform(0.5,1.5)); x=rng.randint(0,IMG-bw); y=rng.randint(0,IMG-min(bh,IMG))
        ImageDraw.Draw(im).rectangle([x,y,x+bw,y+min(bh,IMG)],fill=255)
    else:
        bw=int(IMG*p); x=rng.randint(0,IMG-bw); y=rng.randint(0,max(1,IMG-bw))
        ImageDraw.Draw(im).ellipse([x,y,x+bw,y+bw],fill=255)
    return np.array(im)
Eg=emb([X[i] for i in tr]); broken=[damage(X[j]) for j in te]
r1=restore(broken); r2=restore(r1)
def ev(E,label):
    sim=E@Eg.T; hits={k:0 for k in (1,5,10,20)}
    for qi,t in enumerate(te):
        order=np.argsort(-sim[qi]); labs=[Y[tr[o]] for o in order if slips[tr[o]]!=slips[t]]
        for k in hits:
            if Y[t] in labs[:k]: hits[k]+=1
    n=len(te); r={f'R{k}':round(v/n*100,1) for k,v in hits.items()}
    print(f'{label}: '+'  '.join(f'{k} {v}%' for k,v in r.items())); return r
res={}
res['broken']=ev(emb(broken),'遮挡30%      ')
res['iter1']=ev(emb(r1),'修复 1 次    ')
res['iter2']=ev(emb(r2),'修复 2 次    ')
diff=np.abs(r1.astype(float)-r2.astype(float)).mean()
print(f'两次修复的平均像素差 {diff:.2f}')
json.dump(res,open(os.path.join(OUT,'eval.json'),'w'),ensure_ascii=False,indent=1)
