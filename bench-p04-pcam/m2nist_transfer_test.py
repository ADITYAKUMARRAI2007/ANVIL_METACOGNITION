from __future__ import annotations

from pathlib import Path
import numpy as np

from adapters.dummy import DummyAgent
from adapters.myteam import Engine
from harness import pack_params
from pcam_model import PCAMModel, build_default_R

DATA = Path('m2nist_cache/combined.npy')


def load_m2nist():
    X = np.load(DATA).astype(np.float64) / 255.0
    return X.reshape(X.shape[0], -1)


def fit_pca(X: np.ndarray, n_components=64):
    mu = X.mean(axis=0)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:n_components]


def transform_pca(X, mu, comps):
    Z = (X - mu) @ comps.T
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def corrupt_feature(q, p, rng, sigma=0.4):
    N = q.shape[0]
    out = q.copy()
    out[rng.random(N) < p] = 0.0
    out = out + rng.standard_normal(N) * (sigma / np.sqrt(N))
    return out / (np.linalg.norm(out) + 1e-12)


def eval_seed(seed: int, K=16, n_per_level=40, levels=(0.5,0.7,0.8)):
    Xpix = load_m2nist()
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(Xpix), size=min(3000, len(Xpix)), replace=False)
    mu, comps = fit_pca(Xpix[fit_idx], 64)
    Z = transform_pca(Xpix, mu, comps)

    # Store random M2NIST scenes as attractors. This tests distribution transfer,
    # not class labels/segmentation.
    chosen = rng.choice(len(Z), size=K, replace=False)
    Xstore = Z[chosen]

    model = PCAMModel(Xstore, build_default_R(64, seed=seed))
    params = pack_params(model)
    agent = Engine(Xstore, params)
    dummy = DummyAgent(Xstore, params)

    print(f'===== M2NIST SEED {seed} =====')
    seed_a = seed_b = total = 0
    for p in levels:
        a = b = 0
        for _ in range(n_per_level):
            idx = int(rng.integers(K))
            q = corrupt_feature(Xstore[idx], p, rng)
            pred_b = model.classify(model.run(q, dummy.predict_precision(q), u_const=q))
            pred_a = model.classify(model.run(q, agent.predict_precision(q), u_const=q))
            b += int(pred_b == idx)
            a += int(pred_a == idx)
        seed_a += a; seed_b += b; total += n_per_level
        print(f'noise {p:.1f}: baseline {b}/{n_per_level}={b/n_per_level:.3f} | agent {a}/{n_per_level}={a/n_per_level:.3f} | delta {(a-b)/n_per_level:+.3f}')
    print(f'TOTAL seed {seed}: baseline {seed_b}/{total}={seed_b/total:.3f} | agent {seed_a}/{total}={seed_a/total:.3f} | delta {(seed_a-seed_b)/total:+.3f}')
    return seed_b, seed_a, total


def main():
    print('M2NIST transfer test; does not modify adapters/myteam.py')
    totals = np.array([0,0,0], dtype=int)
    for seed in [42,101,202]:
        totals += np.array(eval_seed(seed))
    b,a,n = totals
    print('===== OVERALL M2NIST TRANSFER =====')
    print(f'baseline {b}/{n}={b/n:.4f}')
    print(f'agent    {a}/{n}={a/n:.4f}')
    print(f'delta    {(a-b)/n:+.4f}')

if __name__ == '__main__':
    main()
