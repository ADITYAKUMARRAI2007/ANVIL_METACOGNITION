from __future__ import annotations

import importlib
import argparse
import numpy as np

from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R
from harness import pack_params
from adapters.dummy import DummyAgent


def load_agent(spec):
    mod, cls = spec.split(':')
    return getattr(importlib.import_module(mod), cls)


def pi_stats(pi):
    pi = np.asarray(pi, dtype=float)
    return float(pi.min()), float(pi.max()), float(pi.mean()), float(pi.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default='adapters.myteam:Engine')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 101, 202, 303, 404])
    ap.add_argument('--K', type=int, default=16)
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--noise-levels', nargs='+', type=float, default=[0.5, 0.7, 0.8])
    ap.add_argument('--n-per-level', type=int, default=80)
    ap.add_argument('--sample-lines', type=int, default=5)
    args = ap.parse_args()

    Agent = load_agent(args.adapter)
    print('RETRIEVAL AUDIT')
    print('Adapter:', args.adapter)
    print('This directly calls model.run(q, pi, u_const=q) for every query.')
    print('Seeds:', args.seeds)
    print('Noise levels:', args.noise_levels)
    print('n_per_level:', args.n_per_level)
    print()

    total_agent = total_base = total_n = 0

    for seed in args.seeds:
        X = make_patterns(K=args.K, N=args.N, seed=seed)
        R = build_default_R(N=args.N, seed=seed)
        model = PCAMModel(X, R)
        params = pack_params(model)
        agent = Agent(X, params)
        dummy = DummyAgent(X, params)
        queries, truths, levels = make_test_queries(X, args.noise_levels, args.n_per_level, seed=seed)

        seed_agent = seed_base = 0
        print(f'===== SEED {seed} =====')
        for level in args.noise_levels:
            idxs = np.where(np.isclose(levels, level))[0]
            a_correct = b_correct = 0
            examples_printed = 0
            for j in idxs:
                q = queries[j]
                truth = int(truths[j])
                pi_a = agent.predict_precision(q)
                pi_b = dummy.predict_precision(q)
                out_a = model.run(q, pi_a, u_const=q)
                out_b = model.run(q, pi_b, u_const=q)
                pred_a = model.classify(out_a)
                pred_b = model.classify(out_b)
                ok_a = pred_a == truth
                ok_b = pred_b == truth
                a_correct += int(ok_a)
                b_correct += int(ok_b)
                if examples_printed < args.sample_lines:
                    mn, mx, mean, std = pi_stats(model.clip_and_normalise(pi_a))
                    print(
                        f'query {j:04d} noise {level:.1f} truth {truth:02d} '
                        f'base_pred {pred_b:02d} agent_pred {pred_a:02d} '
                        f'base_ok {int(ok_b)} agent_ok {int(ok_a)} '
                        f'pi[min,max,mean,std]=[{mn:.3f},{mx:.3f},{mean:.3f},{std:.3f}]'
                    )
                    examples_printed += 1
            n = len(idxs)
            seed_agent += a_correct
            seed_base += b_correct
            print(
                f'noise {level:.1f}: baseline {b_correct}/{n}={b_correct/n:.4f} | '
                f'agent {a_correct}/{n}={a_correct/n:.4f} | delta {(a_correct-b_correct)/n:+.4f}'
            )
        n_seed = len(queries)
        total_agent += seed_agent
        total_base += seed_base
        total_n += n_seed
        print(
            f'SEED {seed} TOTAL: baseline {seed_base}/{n_seed}={seed_base/n_seed:.4f} | '
            f'agent {seed_agent}/{n_seed}={seed_agent/n_seed:.4f} | delta {(seed_agent-seed_base)/n_seed:+.4f}'
        )
        print()

    print('===== OVERALL =====')
    print(f'baseline {total_base}/{total_n}={total_base/total_n:.4f}')
    print(f'agent    {total_agent}/{total_n}={total_agent/total_n:.4f}')
    print(f'delta    {(total_agent-total_base)/total_n:+.4f}')


if __name__ == '__main__':
    main()
