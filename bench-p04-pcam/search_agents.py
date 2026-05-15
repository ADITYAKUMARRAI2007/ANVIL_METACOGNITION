from __future__ import annotations
import numpy as np
from typing import Any
from adapter import Adapter
from harness import run_multi

class ParamAgent(Adapter):
    exp=0.0; rel_exp=0.0; blend=1.0; use_nearest=True; soften=0.0
    def __init__(self,X,params):
        self.X=np.asarray(X,float); self.K,self.N=self.X.shape
        self.R=np.asarray(params['R'],float); self.eta=float(params['eta']); self.beta=float(params['beta'])
        self.diagR=np.diag(self.R)
        self.templates=np.vstack([self.templ(i) for i in range(self.K)])
        self.global_t=self.norm(self.templates.mean(0))
        self.abs_med=np.median(np.abs(self.X),0); self.scale=np.std(self.X,0)+1e-6
    def sm(self,a):
        z=self.beta*(self.X@a); z-=z.max(); e=np.exp(z); return e/e.sum()
    def templ(self,i):
        x=self.X[i]; s=self.sm(x); m=s@self.X; C=self.X-m; var=np.sum(s[:,None]*C*C,0)
        d=np.diag(self.R)-self.eta*self.beta*var
        pos=d[d>1e-8]; floor=np.percentile(pos,5) if pos.size else 1e-3; d=np.maximum(d,max(floor,1e-4))
        # include row offdiag maybe Gershgorin curvature
        if self.soften:
            H=self.R - self.eta*self.beta*(self.X.T@((np.diag(s)-np.outer(s,s))@self.X))
            row=np.sum(np.abs(H),axis=1)
            d=(1-self.soften)*d+self.soften*row
        return self.norm(d**self.exp)
    def norm(self,pi):
        pi=np.clip(np.asarray(pi,float),0.1,10); return pi/pi.mean()
    def predict_precision(self,q):
        q=np.asarray(q,float); q=q/(np.linalg.norm(q)+1e-12); sims=self.X@q
        if self.use_nearest:
            pi=self.templates[int(np.argmax(sims))]
        else:
            # soft top weights
            z=8*sims; z-=z.max(); w=np.exp(z); w/=w.sum(); pi=self.norm(w@self.templates)
        if self.blend<1: pi=self.blend*pi+(1-self.blend)*self.global_t
        if self.rel_exp:
            rel=np.abs(q)/(self.abs_med+0.5*self.scale+1e-6); rel=np.clip(rel,0.25,2.5)**self.rel_exp; pi=pi*rel
        return self.norm(pi)

def make_cls(exp,rel,blend,nearest,soften):
    return type(f'A_e{exp}_r{rel}_b{blend}_n{nearest}_s{soften}',(ParamAgent,),dict(exp=exp,rel_exp=rel,blend=blend,use_nearest=nearest,soften=soften))

cands=[]
for exp in [-2,-1,-0.5,0.5,1,2]:
  for rel in [0,0.1,0.2,-0.1]:
    for blend in [1,0.7]:
      cands.append((exp,rel,blend,True,0.0))
for exp in [-1,1]:
  for soften in [0.5,1.0]:
    cands.append((exp,0,1,True,soften))
for exp in [-1,1]:
    cands.append((exp,0,1,False,0))

for p in cands:
    Cls=make_cls(*p)
    rep=run_multi(lambda X,params: Cls(X,params), seeds=[42,101], K=16,N=64, noise_levels=[0.7,0.8], n_per_level=20, n_aniso=5)
    agg=rep['aggregated']; sc=rep['score']
    print(p, 'd',round(agg['mean_delta'],3),'min',round(agg['min_delta'],3),'spread',round(agg['mean_spread'],2),'score',sc['total_automated'], flush=True)
