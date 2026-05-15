import numpy as np
from adapter import Adapter
from harness import run_multi

class A(Adapter):
    formula='diag_invH'
    def __init__(self,X,params):
        self.X=np.asarray(X,float); self.K,self.N=self.X.shape
        self.R=np.asarray(params['R'],float); self.eta=float(params['eta']); self.beta=float(params['beta'])
        self.templates=np.vstack([self.templ(i) for i in range(self.K)])
    def norm(self,pi):
        pi=np.asarray(pi,float); pi=np.nan_to_num(pi,nan=1,posinf=10,neginf=0.1); pi=np.clip(pi,0.1,10); return pi/pi.mean()
    def sm(self,a):
        z=self.beta*(self.X@a); z-=z.max(); e=np.exp(z); return e/e.sum()
    def H(self,i):
        s=self.sm(self.X[i]); D=np.diag(s)-np.outer(s,s); H=self.R-self.eta*self.beta*(self.X.T@(D@self.X)); return (H+H.T)/2
    def templ(self,i):
        H=self.H(i); w,V=np.linalg.eigh(H); w=np.maximum(w,1e-8)
        if self.formula=='diag_invH': pi=np.diag(V@np.diag(1/w)@V.T)
        elif self.formula=='diag_sqrt_invH': pi=np.sqrt(np.diag(V@np.diag(1/w)@V.T))
        elif self.formula=='leverage_low': pi=np.sum((V[:,:8]**2)*(1/w[:8]),axis=1)
        elif self.formula=='leverage_high': pi=1/(np.sum((V[:,-8:]**2)*w[-8:],axis=1)+1e-9)
        elif self.formula=='eig_balance': pi=np.sum((V**2)*(1/np.sqrt(w))[None,:],axis=1)
        else: pi=np.ones(self.N)
        return self.norm(pi)
    def predict_precision(self,q):
        q=np.asarray(q,float); q=q/(np.linalg.norm(q)+1e-12); sims=self.X@q
        if sims.max()>0.89: return self.templates[int(np.argmax(sims))]
        score=-np.abs(q); score=(score-score.mean())/(score.std()+1e-9); return self.norm(np.exp(0.4*score))

def cls(f): return type('C',(A,),dict(formula=f))
for f in ['diag_invH','diag_sqrt_invH','leverage_low','leverage_high','eig_balance']:
 C=cls(f); rep=run_multi(lambda X,p:C(X,p),seeds=[42,101],K=16,N=64,noise_levels=[0.7,0.8],n_per_level=50,n_aniso=16)
 agg=rep['aggregated']; print(f,'delta',round(agg['mean_delta'],3),'spread',round(agg['mean_spread'],4),'minS',round(agg['min_spread'],4),'score',rep['score']['total_automated'],flush=True)
