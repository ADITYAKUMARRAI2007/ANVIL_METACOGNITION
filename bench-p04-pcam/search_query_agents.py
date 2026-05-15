import numpy as np
from adapter import Adapter
from harness import run_multi

class QA(Adapter):
    mode='absq'; lam=1.0
    def __init__(self,X,params): self.X=np.asarray(X,float); self.K,self.N=self.X.shape
    def norm(self,pi):
        pi=np.clip(pi,0.1,10); return pi/pi.mean()
    def predict_precision(self,q):
        q=np.asarray(q,float); q=q/(np.linalg.norm(q)+1e-12); sims=self.X@q; order=np.argsort(sims)
        b=order[-1]; s=order[-2]; xb=self.X[b]; xs=self.X[s]
        if self.mode=='absq': score=np.abs(q)
        elif self.mode=='invabsq': score=-np.abs(q)
        elif self.mode=='agree': score=q*xb
        elif self.mode=='disagree': score=-q*xb
        elif self.mode=='margin': score=q*(xb-xs)
        elif self.mode=='absmargin': score=np.abs(xb-xs)
        elif self.mode=='rival': score=-(q*xs)
        elif self.mode=='best_minus_abs': score=np.abs(q)*np.sign(q*xb)
        elif self.mode=='topgap': score=(q*xb)-(q*xs)
        else: score=np.zeros_like(q)
        score=(score-score.mean())/(score.std()+1e-9)
        return self.norm(np.exp(self.lam*score))

def cls(mode,lam): return type('C',(QA,),dict(mode=mode,lam=lam))
for mode in ['absq','invabsq','agree','disagree','margin','absmargin','rival','best_minus_abs','topgap']:
  for lam in [0.05,0.1,0.2,0.4,0.8,1.2]:
    C=cls(mode,lam)
    rep=run_multi(lambda X,p:C(X,p),seeds=[42,101],K=16,N=64,noise_levels=[0.7,0.8],n_per_level=30,n_aniso=3)
    agg=rep['aggregated']; print(mode,lam,'d',round(agg['mean_delta'],3),'min',round(agg['min_delta'],3),'spr',round(agg['mean_spread'],2),'score',rep['score']['total_automated'],flush=True)
