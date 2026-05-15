import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

def cond_for(H, y):
    pi=np.exp(y); pi=np.clip(pi,0.1,10); pi=pi/pi.mean(); ps=np.sqrt(pi)
    S=(ps[:,None]*H)*ps[None,:]; S=(S+S.T)/2
    e=np.linalg.eigvalsh(S); e=e[e>1e-9]
    return e[-1]/e[0]

def grad_fd(H,y):
    g=np.zeros_like(y); f=cond_for(H,y); eps=1e-3
    for i in range(len(y)):
        yy=y.copy(); yy[i]+=eps
        g[i]=(cond_for(H,yy)-f)/eps
    return g,f

def opt(H, steps=80, lr=0.05):
    y=np.zeros(H.shape[0])
    best=y.copy(); bf=cond_for(H,y)
    for t in range(steps):
        g,f=grad_fd(H,y)
        y-=lr*g/(np.linalg.norm(g)+1e-9)
        y=np.clip(y,-2.3,2.3); y-=y.mean()
        nf=cond_for(H,y)
        if nf<bf: bf=nf; best=y.copy()
        lr*=0.98
    return np.exp(best)/np.exp(best).mean(), bf

for seed in [42,101]:
 X=make_patterns(16,64,seed); model=PCAMModel(X,build_default_R(64,seed=seed))
 rng=np.random.default_rng(seed); idxs=rng.choice(16,5,False)
 vals=[]
 for idx in idxs:
  H=model.hessian(X[idx]); H=(H+H.T)/2
  base=cond_for(H,np.zeros(64))
  pi,b=opt(H,steps=30)
  vals.append(base/b)
  print(seed,idx,'base',base,'opt',b,'ratio',base/b,'pi range',pi.min(),pi.max())
 print('mean',np.mean(vals))
