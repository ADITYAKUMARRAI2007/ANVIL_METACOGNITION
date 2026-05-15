from __future__ import annotations
import numpy as np
from adapter import Adapter
from harness import run_multi


def norm_pi(pi):
    pi=np.asarray(pi,float); pi=np.nan_to_num(pi,nan=1,posinf=10,neginf=0.1); pi=np.clip(pi,0.1,10); return pi/pi.mean()

def cond_and_grad(H, y):
    # y parameterizes pi = exp(y)/mean; mean scaling does not affect condition.
    y = y - y.mean()
    pi = np.exp(np.clip(y, np.log(0.1), np.log(10.0)))
    pi = pi / pi.mean()
    d = np.sqrt(pi)
    S = (d[:,None] * H) * d[None,:]
    S = 0.5*(S+S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, 1e-12)
    imin, imax = 0, len(vals)-1
    # derivative log(kappa) wrt log(pi_i) is 0.5? For S=P^1/2 H P^1/2,
    # d log lambda / d log pi_i = v_i^2 for normalized eigenvector.
    g = vecs[:,imax]**2 - vecs[:,imin]**2
    return float(vals[imax]/vals[imin]), g, pi

def optimize_pi(H, steps=400, lr=0.8, restarts=4, seed=0):
    rng=np.random.default_rng(seed)
    best_pi=np.ones(H.shape[0]); best_c=cond_and_grad(H, np.zeros(H.shape[0]))[0]
    starts=[np.zeros(H.shape[0])]
    for _ in range(restarts-1): starts.append(rng.normal(0,0.5,H.shape[0]))
    for y in starts:
        y=y.copy(); m=np.zeros_like(y); v=np.zeros_like(y)
        for t in range(1,steps+1):
            c,g,pi=cond_and_grad(H,y)
            if c<best_c: best_c=c; best_pi=pi.copy()
            # Adam on log condition number
            m=0.9*m+0.1*g; v=0.999*v+0.001*(g*g)
            step=lr*m/(np.sqrt(v)+1e-8)
            y -= step
            y -= y.mean()
            # Clip relative pi range; center after clipping.
            y=np.clip(y,np.log(0.1),np.log(10.0)); y-=y.mean()
            lr_t=lr*(0.995**t)
        lr=lr*0.7
    return norm_pi(best_pi), best_c

class OptCondAgent(Adapter):
    def __init__(self,X,params):
        self.X=np.asarray(X,float); self.K,self.N=self.X.shape
        self.R=np.asarray(params['R'],float); self.eta=float(params['eta']); self.beta=float(params['beta'])
        self.templates=[]; self.conds=[]
        for i in range(self.K):
            H=self.H(i)
            pi,c=optimize_pi(H,steps=250,lr=0.15,restarts=2,seed=i)
            self.templates.append(pi); self.conds.append(c)
        self.templates=np.asarray(self.templates)
    def sm(self,a):
        z=self.beta*(self.X@a); z-=z.max(); e=np.exp(z); return e/e.sum()
    def H(self,i):
        s=self.sm(self.X[i]); D=np.diag(s)-np.outer(s,s); H=self.R-self.eta*self.beta*(self.X.T@(D@self.X)); return 0.5*(H+H.T)
    def predict_precision(self,q):
        q=np.asarray(q,float); q=q/(np.linalg.norm(q)+1e-12); sims=self.X@q; best=int(np.argmax(sims))
        if sims[best]>0.89: return self.templates[best]
        score=-np.abs(q); score=(score-score.mean())/(score.std()+1e-9); return norm_pi(np.exp(0.4*score))

if __name__=='__main__':
    rep=run_multi(lambda X,p:OptCondAgent(X,p),seeds=[42,101],K=16,N=64,noise_levels=[0.7,0.8],n_per_level=50,n_aniso=16)
    print(rep)
