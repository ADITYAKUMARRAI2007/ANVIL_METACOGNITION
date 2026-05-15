import numpy as np
from data import make_patterns
from pcam_model import PCAMModel, build_default_R

def pi_from_y(y):
    y=np.asarray(y,float); y=y-y.mean(); y=np.clip(y,np.log(.1),np.log(10)); pi=np.exp(y); return pi/pi.mean()

def spread(H, pi):
    pi=np.asarray(pi,float); pi=np.clip(pi,.1,10); pi=pi/pi.mean(); d=np.sqrt(pi)
    S=(d[:,None]*H)*d[None,:]; S=(S+S.T)/2; e=np.linalg.eigvalsh(S); e=e[e>1e-9]
    return float(e[-1]/e[0])

def grad_kappa(H,y,mode='kappa'):
    pi=pi_from_y(y); d=np.sqrt(pi); S=(d[:,None]*H)*d[None,:]; S=(S+S.T)/2
    vals,V=np.linalg.eigh(S); vals=np.maximum(vals,1e-12)
    if mode=='kappa': g=V[:,-1]**2 - V[:,0]**2
    elif mode=='lmin': g= - V[:,0]**2
    elif mode=='deflated':
        # ignore top eigvec: reduce lambda_2/lambda_min in residual spread
        g=V[:,-2]**2 - V[:,0]**2
    return vals[-1]/vals[0], g, pi

def adam(H, mode='kappa', steps=1200, lr=.01, starts=6, seed=0):
    rng=np.random.default_rng(seed); best=(spread(H,np.ones(H.shape[0])), np.ones(H.shape[0]))
    inits=[np.zeros(H.shape[0]), np.log(1/np.maximum(np.diag(H),1e-6)), np.log(np.maximum(np.diag(H),1e-6))]
    inits += [rng.normal(0,s,H.shape[0]) for s in [.2,.5,1.0]][:max(0,starts-len(inits))]
    for y0 in inits:
        y=y0.copy(); m=np.zeros_like(y); v=np.zeros_like(y)
        for t in range(1,steps+1):
            _,g,pi=grad_kappa(H,y,mode)
            c=spread(H,pi)
            if c<best[0]: best=(c,pi.copy())
            m=.9*m+.1*g; v=.999*v+.001*g*g
            y-=lr*m/(np.sqrt(v)+1e-8); y-=y.mean(); y=np.clip(y,np.log(.1),np.log(10)); y-=y.mean()
    return best

def ruiz(H, iters=100):
    D=np.ones(H.shape[0])
    for _ in range(iters):
        S=(D[:,None]*H)*D[None,:]
        rn=np.sqrt(np.sum(np.abs(S),axis=1)+1e-12)
        D=D/np.sqrt(rn)
        pi=D*D; pi=pi/pi.mean(); D=np.sqrt(np.clip(pi,.1,10))
    return (spread(H,D*D), (D*D)/(D*D).mean())

def evolution(H, gens=250, pop=80, sigma=.6, seed=0):
    rng=np.random.default_rng(seed); n=H.shape[0]
    mean=np.zeros(n); best=(spread(H,np.ones(n)), np.ones(n))
    for g in range(gens):
        Y=mean + rng.normal(0,sigma,(pop,n))
        scores=[]
        for y in Y:
            pi=pi_from_y(y); c=spread(H,pi); scores.append(c)
            if c<best[0]: best=(c,pi.copy())
        idx=np.argsort(scores)[:max(4,pop//5)]
        mean=Y[idx].mean(axis=0); mean-=mean.mean(); sigma*=0.992
    return best

for seed in [42,101,202]:
    X=make_patterns(16,64,seed); model=PCAMModel(X,build_default_R(64,seed=seed)); rng=np.random.default_rng(seed); idxs=rng.choice(16,3,False)
    print('\nSEED',seed)
    for idx in idxs:
        H=model.hessian(X[idx]); H=(H+H.T)/2; base=spread(H,np.ones(64))
        methods={
          'ruiz': ruiz(H),
          'adam_kappa': adam(H,'kappa',seed=idx),
          'adam_lmin': adam(H,'lmin',seed=idx),
          'adam_deflated': adam(H,'deflated',seed=idx),
          'evolution': evolution(H,seed=idx),
        }
        print('idx',idx,'base',round(base,4))
        for name,(c,pi) in methods.items():
            print(' ',name,'spread',round(c,4),'ratio',round(base/c,4),'pi_range',round(pi.min(),3),round(pi.max(),3))
