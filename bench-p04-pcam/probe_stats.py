import numpy as np
from data import make_patterns, make_test_queries
from pcam_model import build_default_R, PCAMModel
for seed in [42,101,202]:
 X=make_patterns(16,64,seed); rng=np.random.default_rng(seed)
 queries,truths,levels=make_test_queries(X,[0.5,0.7,0.8],250,seed)
 sims=queries@X.T; mx=sims.max(1)
 print('seed',seed,'corrupt maxsim quant',np.quantile(mx,[0,0.1,0.25,0.5,0.75,0.9,1]))
 # anisotropy probe generation exactly
 vals=[]
 idxs=rng.choice(16,size=16,replace=False)
 for idx in idxs:
  probe=X[idx]+rng.standard_normal(64)*0.05; probe=probe/np.linalg.norm(probe); vals.append((X@probe).max())
 print('probe maxsim quant',np.quantile(vals,[0,0.1,0.25,0.5,0.75,0.9,1]))
