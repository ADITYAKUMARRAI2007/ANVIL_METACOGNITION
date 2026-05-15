from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

from adapters.dummy import DummyAgent
from adapters.myteam import Engine
from harness import pack_params
from pcam_model import PCAMModel, build_default_R

DATA = Path('mnist_cache')
URLS = {
    'images': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
    'labels': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
}


def download(url: str, path: Path):
    path.parent.mkdir(exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print('downloading', url, '->', path, flush=True)
    urllib.request.urlretrieve(url, path)


def load_mnist():
    img_gz = DATA / 'train-images-idx3-ubyte.gz'
    lab_gz = DATA / 'train-labels-idx1-ubyte.gz'
    download(URLS['images'], img_gz)
    download(URLS['labels'], lab_gz)
    with gzip.open(img_gz, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols).astype(np.float64) / 255.0
    with gzip.open(lab_gz, 'rb') as f:
        magic, nlab = struct.unpack('>II', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(int)
    return X, y


def fit_pca(X: np.ndarray, n_components=64):
    mu = X.mean(axis=0)
    Xc = X - mu
    # economical PCA via SVD; components are rows of Vt
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:n_components]


def transform_pca(X: np.ndarray, mu: np.ndarray, comps: np.ndarray):
    Z = (X - mu) @ comps.T
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return Z


def make_pca_mnist_patterns(seed: int, K=16, N=64, pca_fit_n=6000):
    Xpix, y = load_mnist()
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(Xpix), size=min(pca_fit_n, len(Xpix)), replace=False)
    mu, comps = fit_pca(Xpix[fit_idx], N)
    Z = transform_pca(Xpix, mu, comps)

    # Pick 16 stored samples: one per digit first, then extra ambiguous-looking random digits.
    chosen = []
    used_digits = set()
    for d in range(10):
        ids = np.where(y == d)[0]
        chosen.append(int(rng.choice(ids)))
        used_digits.add(d)
    while len(chosen) < K:
        chosen.append(int(rng.integers(len(Xpix))))
    Xstore = Z[chosen]
    labels = y[chosen]
    return Xstore, labels, Z, y


def corrupt_feature(q: np.ndarray, p: float, rng: np.random.Generator, sigma=0.4):
    # Same corruption style as public bench but applied to PCA features.
    N = q.shape[0]
    out = q.copy()
    mask = rng.random(N) < p
    out[mask] = 0.0
    out = out + rng.standard_normal(N) * (sigma / np.sqrt(N))
    out = out / (np.linalg.norm(out) + 1e-12)
    return out


def eval_seed(seed: int, n_per_level=50, levels=(0.5, 0.7, 0.8)):
    Xstore, labels, _, _ = make_pca_mnist_patterns(seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(Xstore, R)
    params = pack_params(model)
    agent = Engine(Xstore, params)
    dummy = DummyAgent(Xstore, params)
    rng = np.random.default_rng(seed)

    print(f'===== PCA-MNIST SEED {seed} =====')
    seed_a = seed_b = total = 0
    for p in levels:
        a = b = 0
        for _ in range(n_per_level):
            idx = int(rng.integers(len(Xstore)))
            q = corrupt_feature(Xstore[idx], p, rng)
            pb = dummy.predict_precision(q)
            pa = agent.predict_precision(q)
            pred_b = model.classify(model.run(q, pb, u_const=q))
            pred_a = model.classify(model.run(q, pa, u_const=q))
            b += int(pred_b == idx)
            a += int(pred_a == idx)
        n = n_per_level
        seed_a += a; seed_b += b; total += n
        print(f'noise {p:.1f}: baseline {b}/{n}={b/n:.3f} | agent {a}/{n}={a/n:.3f} | delta {(a-b)/n:+.3f}')
    print(f'TOTAL seed {seed}: baseline {seed_b}/{total}={seed_b/total:.3f} | agent {seed_a}/{total}={seed_a/total:.3f} | delta {(seed_a-seed_b)/total:+.3f}')
    return seed_b, seed_a, total


def main():
    print('PCA-MNIST transfer test; does not modify benchmark files or adapters/myteam.py')
    totals = np.array([0, 0, 0], dtype=int)
    for seed in [42, 101, 202]:
        b, a, n = eval_seed(seed, n_per_level=40)
        totals += np.array([b, a, n])
    b, a, n = totals
    print('===== OVERALL PCA-MNIST TRANSFER =====')
    print(f'baseline {b}/{n}={b/n:.4f}')
    print(f'agent    {a}/{n}={a/n:.4f}')
    print(f'delta    {(a-b)/n:+.4f}')


if __name__ == '__main__':
    main()
