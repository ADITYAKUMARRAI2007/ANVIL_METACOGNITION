# P-04 · PCAM Precision Agent Submission

Team adapter: `adapters/myteam.py`
Class: `Engine`
Dependencies: NumPy only, as required by `requirements.txt`.

## Approach

The agent controls the diagonal precision vector used by the frozen PCAM dynamics. It has two branches:

1. **Retrieval precision for corrupted queries**
   - The query is normalised and compared against stored patterns.
   - Low-confidence queries receive stronger inverse-`|query|` precision shaping.
   - This helps under mask/noise corruption by reducing reliance on unreliable query dimensions while preserving the mean-1 precision constraint expected by the harness.

2. **Geometry precision for clean attractor-like queries**
   - For high-confidence, well-separated queries, the agent uses a per-attractor Hessian template.
   - Hessians are evaluated at the true PCAM equilibrium via the provided frozen `PCAMModel`, then a deterministic diagonal preconditioner is optimised to reduce local spread.
   - Ambiguous queries stay retrieval-first to avoid hurting retrieval accuracy.

The PCAM model itself is not retrained or modified. The agent only returns one positive precision vector per query.

## Setup

```bash
git clone https://github.com/Sauhard74/Anvil-P-E
cd Anvil-P-E/bench-p04-pcam
pip install -r requirements.txt
```

## Run baseline

```bash
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

## Run this submission

Quick check:

```bash
python self_check.py --adapter adapters.myteam:Engine --quick
```

Full evaluation with the public multi-seed command:

```bash
python run.py --adapter adapters.myteam:Engine \
  --seeds 7 13 31 97 211 503 1009 --out report.json
```

## Verified local results

Quick self-check after cleanup:

```text
retrieval     70.00 / 70
anisotropy     3.05 / 20
TOTAL         73.05 / 90
```

Full default self-check:

```text
retrieval     70.00 / 70
anisotropy     3.24 / 20
TOTAL         73.24 / 90
mean Δ accuracy        +0.263
min Δ accuracy         +0.095
mean spread reduction   1.30×
```

Additional transfer and stress-test summaries are in `reports.json`.

## Notes

- No external dependencies beyond NumPy.
- No GPU required.
- No offline neural model or trained weights are used.
- The main limitation is anisotropy: retrieval is strong, but spread reduction is modest compared with the hardest target.
