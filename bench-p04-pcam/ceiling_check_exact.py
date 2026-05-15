import argparse
import numpy as np

from data import make_patterns
from pcam_model import PCAMModel, build_default_R
from checks import per_pattern_spread


def normalise_pi(raw, pi_min=0.1, pi_max=10.0):
    pi = np.asarray(raw, dtype=np.float64)
    pi = np.clip(pi, pi_min, pi_max)
    m = pi.mean()
    if m > 0:
        pi = pi / m
    return pi


def spread_for_logpi(logpi, model, H_pattern):
    pi = normalise_pi(np.exp(logpi), model.pi_min, model.pi_max)
    s = per_pattern_spread(model, pi, H_pattern)
    if s is None or not np.isfinite(s):
        return 1e18
    return s


def random_boundary_search(model, pattern, base_spread, trials=3000, seed=0):
    rng = np.random.default_rng(seed)
    N = model.N

    best_pi = np.ones(N)
    best_spread = per_pattern_spread(model, best_pi, pattern)

    # Try identity, random log-normal, and boundary-heavy candidates.
    for t in range(trials):
        mode = t % 4

        if mode == 0:
            logpi = rng.normal(0.0, 0.35, size=N)
            raw = np.exp(logpi)

        elif mode == 1:
            logpi = rng.normal(0.0, 0.9, size=N)
            raw = np.exp(logpi)

        elif mode == 2:
            raw = np.where(rng.random(N) < 0.5, model.pi_min, model.pi_max)

        else:
            raw = np.ones(N)
            idx = rng.choice(N, size=max(1, N // 4), replace=False)
            raw[idx] = model.pi_min
            idx2 = rng.choice(N, size=max(1, N // 4), replace=False)
            raw[idx2] = model.pi_max

        pi = normalise_pi(raw, model.pi_min, model.pi_max)
        s = per_pattern_spread(model, pi, pattern)

        if s is not None and s < best_spread:
            best_spread = s
            best_pi = pi.copy()

    return best_spread, best_pi


def finite_difference_descent(model, pattern, start_pi, steps=250, lr=0.05, eps=1e-4):
    """
    Simple dependency-free optimizer.
    Uses finite differences on log(pi), then applies the model's clip+mean normalization.
    Slower than scipy, but works with NumPy only.
    """
    N = model.N
    x = np.log(np.maximum(start_pi, 1e-9))
    best_x = x.copy()
    best = spread_for_logpi(x, model, pattern)

    for step in range(steps):
        grad = np.zeros(N)
        base = spread_for_logpi(x, model, pattern)

        for i in range(N):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            fp = spread_for_logpi(xp, model, pattern)
            fm = spread_for_logpi(xm, model, pattern)
            grad[i] = (fp - fm) / (2 * eps)

        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad = grad / gnorm

        # Backtracking line search
        improved = False
        local_lr = lr
        for _ in range(12):
            candidate = x - local_lr * grad
            val = spread_for_logpi(candidate, model, pattern)
            if val < base:
                x = candidate
                improved = True
                if val < best:
                    best = val
                    best_x = candidate.copy()
                break
            local_lr *= 0.5

        if not improved:
            break

    best_pi = normalise_pi(np.exp(best_x), model.pi_min, model.pi_max)
    return best, best_pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202])
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--patterns", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3000)
    parser.add_argument("--fd_steps", type=int, default=120)
    args = parser.parse_args()

    print("Exact anisotropy ceiling check")
    print("Goal: minimize the same per_pattern_spread used by checks.py")
    print()

    all_ratios = []

    for seed in args.seeds:
        X = make_patterns(K=args.K, N=args.N, seed=seed)
        R = build_default_R(N=args.N, seed=seed)
        model = PCAMModel(X, R=R)

        print(f"SEED {seed}")
        print("-" * 60)

        for idx in range(min(args.patterns, args.K)):
            pattern = model.X[idx]
            base = per_pattern_spread(model, np.ones(args.N), pattern)

            rand_spread, rand_pi = random_boundary_search(
                model, pattern, base, trials=args.trials, seed=seed * 1000 + idx
            )

            fd_spread, fd_pi = finite_difference_descent(
                model, pattern, rand_pi, steps=args.fd_steps, lr=0.08
            )

            best = min(rand_spread, fd_spread)
            ratio = base / best if best > 0 else 0.0
            all_ratios.append(ratio)

            print(
                f"pattern {idx:02d} | "
                f"base {base:.4f} | "
                f"random {rand_spread:.4f} | "
                f"fd {fd_spread:.4f} | "
                f"best ratio {ratio:.4f}x"
            )

        print()

    print("=" * 60)
    print(f"Mean best ratio: {np.mean(all_ratios):.4f}x")
    print(f"Max best ratio: {np.max(all_ratios):.4f}x")
    print(f"Min best ratio: {np.min(all_ratios):.4f}x")

    if np.max(all_ratios) < 1.2:
        print("\nREAD: ceiling looks very low. >10x is not realistic with honest diagonal precision here.")
    elif np.max(all_ratios) < 2.0:
        print("\nREAD: some improvement exists, but still far from 10x.")
    else:
        print("\nREAD: hidden anisotropy potential exists. Build final adapter around the best optimizer.")


if __name__ == "__main__":
    main()
