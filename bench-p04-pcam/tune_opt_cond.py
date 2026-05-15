import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

def cond_grad(H,y):
 y=y-y.mean(); y=np.clip(y,np.log(0.1),np.log(10)); pi=np.exp(y); pi=pi/pi.mean(); d=np.sqrt(pi)
 S=(d[:,None]*H)*d[None,:]; S=(S+S.T)/2; vals,V=np.linalg.eigh(S); vals=np.maximum(vals,1e-12)
 g=V[:,-1]**2 - V[:,0]**2
 return vals[-1]/vals[0],g,pi

def run(H,steps,lr,method='adam',restarts=8,seed=0):
 rng=np.random.default_rng(seed); best=1e9; bp=None
 starts=[np.zeros(H.shape[0])]+[rng.normal(0,s,H.shape[0]) for s in [0.2,0.5,1.0,1.5,2.0,0.8,1.2]][:restarts-1]
 for y in starts:
  m=np.zeros_like(y); v=np.zeros_like(y); cur_lr=lr
  for t in range(1,steps+1):
   c,g,pi=cond_grad(H,y)
   if c<best: best=c; bp=pi.copy()
   if method=='adam':
    m=.9*m+.1*g; v=.999*v+.001*g*g; y-=cur_lr*m/(np.sqrt(v)+1e-8)
   else:
    y-=cur_lr*g/(np.linalg.norm(g)+1e-9)
   y-=y.mean(); y=np.clip(y,np.log(.1),np.log(10)); y-=y.mean(); cur_lr*=0.999
 return best,bp

X=make_patterns(16,64,42); m=PCAMModel(X,build_default_R(64,seed=42)); rng=np.random.default_rng(42); idxs=rng.choice(16,5,False)
for steps in [100,300,800,1500]:
 for lr in [0.01,0.03,0.1,0.3,0.8]:
  ratios=[]; ranges=[]
  for idx in idxs:
   H=m.hessian(X[idx]); H=(H+H.T)/2; base=cond_grad(H,np.zeros(64))[0]; best,pi=run(H,steps,lr,'adam',8,idx)
   ratios.append(base/best); ranges.append((pi.min(),pi.max()))
  print('steps',steps,'lr',lr,'mean ratio',np.mean(ratios),'min',np.min(ratios),'range avg',np.mean([r[0] for r in ranges]),np.mean([r[1] for r in ranges]),flush=True)
