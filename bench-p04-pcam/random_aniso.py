import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

def cond(H,pi):
 pi=np.clip(pi,0.1,10); pi=pi/pi.mean(); ps=np.sqrt(pi); S=(ps[:,None]*H)*ps[None,:]; S=(S+S.T)/2; e=np.linalg.eigvalsh(S); e=e[e>1e-9]; return e[-1]/e[0]
for seed in [42]:
 X=make_patterns(16,64,seed); m=PCAMModel(X,build_default_R(64,seed=seed)); H=(m.hessian(X[0])+m.hessian(X[0]).T)/2; base=cond(H,np.ones(64)); print('base',base)
 rng=np.random.default_rng(0); best=base; bp=None
 for scale in [0.5,1,2,3,5,10]:
  for t in range(20000):
   y=rng.normal(0,scale,64); pi=np.exp(y); c=cond(H,pi)
   if c<best: best=c; bp=pi/pi.mean(); print('scale',scale,'t',t,'best',best,'ratio',base/best,'range',bp.min(),bp.max(),flush=True)
 print('final',best,base/best)
