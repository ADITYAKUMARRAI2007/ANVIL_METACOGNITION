import numpy as np
from adapter import Adapter
from harness import run_multi

class F(Adapter):
    formula='identity'
    def __init__(self,X,params):
        self.X=np.asarray(X,float); self.K,self.N=self.X.shape
        self.R=np.asarray(params['R'],float); self.eta=float(params['eta']); self.beta=float(params['beta'])
        self.templates=np.vstack([self.template(i) for i in range(self.K)])
    def sm(self,a):
        z=self.beta*(self.X@a); z-=z.max(); e=np.exp(z); return e/e.sum()
    def norm(self,pi):
        pi=np.clip(pi,0.1,10); return pi/pi.mean()
    def H(self,i):
        s=self.sm(self.X[i]); D=np.diag(s)-np.outer(s,s); H=self.R-self.eta*self.beta*(self.X.T@(D@self.X)); return (H+H.T)/2
    def template(self,i):
        H=self.H(i); d=np.diag(H); row=np.sum(np.abs(H),1); off=row-np.abs(d)
        if self.formula=='invdiag': pi=1/np.maximum(d,1e-6)
        elif self.formula=='diag': pi=np.maximum(d,1e-6)
        elif self.formula=='invrow': pi=1/np.maximum(row,1e-6)
        elif self.formula=='row': pi=row
        elif self.formula=='invoff': pi=1/np.maximum(off,1e-6)
        elif self.formula=='off': pi=off
        elif self.formula=='gersh': pi=1/np.maximum(d+off,1e-6)
        elif self.formula=='var':
            # softmax variance
            s=self.sm(self.X[i]); m=s@self.X; C=self.X-m; var=np.sum(s[:,None]*C*C,0); pi=1/(var+1e-6)
        elif self.formula=='xabs': pi=1/(np.abs(self.X[i])+0.05)
        elif self.formula=='xabs_pos': pi=np.abs(self.X[i])+0.05
        else: pi=np.ones(self.N)
        return self.norm(pi)
    def predict_precision(self,q):
        q=np.asarray(q,float); q=q/(np.linalg.norm(q)+1e-12); mx=np.max(self.X@q)
        if mx>0.89:
            return self.templates[int(np.argmax(self.X@q))]
        score=-np.abs(q); score=(score-score.mean())/(score.std()+1e-9); return self.norm(np.exp(0.4*score))

def cls(name): return type('C',(F,),dict(formula=name))
for name in ['identity','invdiag','diag','invrow','row','invoff','off','gersh','var','xabs','xabs_pos']:
 C=cls(name); rep=run_multi(lambda X,p:C(X,p),seeds=[42,101],K=16,N=64,noise_levels=[0.7,0.8],n_per_level=50,n_aniso=16)
 agg=rep['aggregated']; print(name,'delta',round(agg['mean_delta'],3),'minD',round(agg['min_delta'],3),'spread',round(agg['mean_spread'],4),'minS',round(agg['min_spread'],4),'score',rep['score']['total_automated'],flush=True)
