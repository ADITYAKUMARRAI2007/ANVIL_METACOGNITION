import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

def cond(H,pi):
 pi=np.asarray(pi,float); pi=np.clip(pi,0.1,10); pi=pi/pi.mean(); d=np.sqrt(pi); S=(d[:,None]*H)*d[None,:]; S=(S+S.T)/2; e=np.linalg.eigvalsh(S); return e[-1]/e[0]

def analyse(seed=42,idx=0):
 X=make_patterns(16,64,seed); m=PCAMModel(X,build_default_R(64,seed=seed)); H=m.hessian(X[idx]); H=(H+H.T)/2
 R=m.R; print('base cond H',cond(H,np.ones(64)),'R',cond(R,np.ones(64)))
 w,V=np.linalg.eigh(H); print('eig min/max',w[0],w[-1],'top align ones',abs(V[:,-1]@np.ones(64)/8),'low align ones',abs(V[:,0]@np.ones(64)/8))
 print('diag range',np.diag(H).min(),np.diag(H).max(),'off abs mean',np.mean(np.abs(H-np.diag(np.diag(H)))))
 # try analytic rank-one model alpha I + delta 11T condition lower bound? Diagonal scaling of alpha I + delta ssT.
 # sample extreme sparse pi patterns by number high dims.
 best=(cond(H,np.ones(64)),None)
 for k in range(1,64):
  for high in [True,False]:
   pi=np.ones(64)*0.1
   if high: pi[:k]=10
   else: pi[:k]=0.1; pi[k:]=10
   c=cond(H,pi)
   if c<best[0]: best=(c,(k,high,pi.min(),pi.max()))
 print('best block extreme',best,'ratio',cond(H,np.ones(64))/best[0])
 # random log-uniform raw clipped
 rng=np.random.default_rng(0); b=cond(H,np.ones(64)); bp=None
 for t in range(20000):
  y=rng.uniform(np.log(.1),np.log(10),64); pi=np.exp(y); c=cond(H,pi)
  if c<b: b=c; bp=pi
 print('best random',b,'ratio',cond(H,np.ones(64))/b,'range',None if bp is None else (bp.min(),bp.max()))
for idx in [0,1,5,10]: analyse(42,idx)
