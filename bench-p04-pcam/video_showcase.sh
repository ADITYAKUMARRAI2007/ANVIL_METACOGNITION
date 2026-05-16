#!/usr/bin/env bash
set -euo pipefail

# Video showcase script for P-04 submission.
# Purpose: terminal-friendly, cached-log based presentation under ~5 minutes.
# It does NOT rerun long benchmarks. It validates files, then displays
# official report.json + reports.json in readable sections.

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/video_showcase_${STAMP}.log"

PY="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x "/Users/adityakumarrai/.openclaw/workspace/p04_submission_archive_20260516_082924/final_structure_cleanup/.venv/bin/python" ]]; then
  PY="/Users/adityakumarrai/.openclaw/workspace/p04_submission_archive_20260516_082924/final_structure_cleanup/.venv/bin/python"
fi

PAUSE="${PAUSE:-1.2}"
pause() { sleep "$PAUSE"; }

section() {
  echo
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
  pause
}

{
section "ANVIL P-04 · PCAM PRECISION AGENT · VIDEO SHOWCASE"
echo "Folder: $ROOT"
echo "Adapter: adapters.myteam:Engine"
echo "Agent file: adapters/myteam.py"
echo "Official report: report.json"
echo "Evidence report: reports.json"
echo "This script shows cached/validated logs so the video stays under 5 minutes."
echo "Log being written: $LOG"
pause

section "1 / 7 · CLEAN SUBMISSION FILES"
echo "Required benchmark files + our submission/report files:"
find . -maxdepth 2 -type f \
  ! -path './logs/*' \
  ! -path './.git/*' \
  | sed 's#^./##' | sort
pause

section "2 / 7 · CODE QUALITY / REPRODUCIBILITY CHECK"
echo "+ $PY -m py_compile adapter.py adapters/myteam.py data.py harness.py metrics.py pcam_model.py run.py self_check.py"
$PY -m py_compile adapter.py adapters/myteam.py data.py harness.py metrics.py pcam_model.py run.py self_check.py
echo "+ $PY -m json.tool report.json reports.json"
$PY -m json.tool report.json >/dev/null
$PY -m json.tool reports.json >/dev/null
echo "PASS: code compiles and both JSON reports are valid."
pause

section "3 / 7 · AGENT APPROACH SUMMARY"
cat <<'TEXT'
The agent returns a diagonal precision vector for every corrupted query.

Branch A · Retrieval precision
- Normalises the query and compares it with stored patterns.
- Uses adaptive inverse-|query| precision for corrupted/ambiguous queries.
- Low-confidence queries receive stronger precision shaping.

Branch B · Geometry precision
- For clean, high-margin attractor-like queries, uses Hessian templates.
- Hessians are evaluated at the true PCAM equilibrium via the frozen PCAMModel.
- Ambiguous queries stay retrieval-first to avoid damaging retrieval.

No PCAM model retraining. No external dependencies beyond NumPy.
TEXT
pause

section "4 / 7 · OFFICIAL 7-SEED RESULT FROM report.json"
$PY - <<'PY'
import json
r=json.load(open('report.json'))
print('Command used:')
print('python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json')
print()
print('Seeds:', r['config']['seeds'])
agg=r['aggregated']; score=r['score']
print(f"Mean Δ accuracy:       {agg['mean_delta']:+.4f}")
print(f"Worst-seed Δ accuracy: {agg['min_delta']:+.4f}")
print(f"Mean spread reduction: {agg['mean_reduction']:.2f}x")
print(f"Min spread reduction:  {agg['min_reduction']:.2f}x")
print()
print('Score:')
print(f"  Retrieval:  {score['retrieval_pts']:>6.2f} / 70")
print(f"  Anisotropy: {score['anisotropy_pts']:>6.2f} / 20")
print(f"  Total:      {score['total_automated']:>6.2f} / {score['max_automated']:.0f}")
print()
print('Per-seed table:')
print(f"{'seed':>6} {'direct':>8} {'base':>8} {'agent':>8} {'delta':>9} {'reduction':>10}")
for row in r['per_seed']:
    print(f"{row['seed']:>6} {row['direct_classify_acc']:>8.3f} {row['baseline_acc']:>8.3f} {row['agent_acc']:>8.3f} {row['delta']:>+9.3f} {row['spread_reduction']:>9.2f}x")
PY
pause

section "5 / 7 · QUICK SELF-CHECK RESULT"
$PY - <<'PY'
import json
r=json.load(open('reports.json'))
q=r['official_p04']['quick_self_check']
print('Command: python self_check.py --adapter adapters.myteam:Engine --quick')
print(f"Mean Δ accuracy:       {q['mean_delta']:+.3f}")
print(f"Worst-seed Δ accuracy: {q['min_delta']:+.3f}")
print(f"Mean spread reduction: {q['mean_spread_reduction']:.2f}x")
print(f"Retrieval:             {q['retrieval_points']:.2f} / 70")
print(f"Anisotropy:            {q['anisotropy_points']:.2f} / 20")
print(f"Total automated:       {q['total_automated']:.2f} / {q['max_automated']:.0f}")
PY
pause

section "6 / 7 · TRANSFER DATASETS FROM reports.json"
$PY - <<'PY'
import json
r=json.load(open('reports.json'))
print(f"{'dataset':<18} {'baseline':>14} {'agent':>14} {'delta':>10}")
print('-'*60)
for name, item in r['transfer_tests'].items():
    print(f"{name:<18} {item['baseline']+'='+format(item['baseline_acc'],'.4f'):>14} {item['agent']+'='+format(item['agent_acc'],'.4f'):>14} {item['delta']:>+10.4f}")
PY
pause

section "7 / 7 · STRESS / ANTI-BENCHMARK EVIDENCE"
$PY - <<'PY'
import json
r=json.load(open('reports.json'))
cur=r['stress_tests']['current_agent']
print('Hidden-style stress matrix, current agent:')
print(f"{'scenario':<22} {'meanΔ':>9} {'minΔ':>9} {'reduction':>11} {'score':>10}")
print('-'*70)
for k,v in cur.items():
    if k=='summary':
        continue
    print(f"{k:<22} {v['mean_delta']:>+9.3f} {v['min_delta']:>+9.3f} {str(v['spread_reduction'])+'x':>11} {str(v['score'])+'/90':>10}")
print()
s=cur['summary']
print(f"Stress summary: worst_min_delta={s['worst_min_delta']:+.4f}, mean_total_score={s['mean_total_score']:.2f}/90, mean_reduction={s['mean_reduction']:.2f}x")
print()
anti=r['stress_tests']['anti_benchmark_corruptions_current']
print('Anti-benchmark corruptions overall:')
print(f"baseline={anti['overall_baseline_acc']:.3f}, agent={anti['overall_agent_acc']:.3f}, delta={anti['overall_delta']:+.3f}, min_seedkind_delta={anti['min_seedkind_delta']:+.3f}")
PY
pause

section "DONE · WHAT TO SUBMIT"
cat <<'TEXT'
Required submission files:
- adapters/myteam.py      # Engine implementation
- README.md               # approach + setup + dependencies
- report.json             # official 7-seed run output
- reports.json            # extra evidence / transfer / stress summaries

Run command for judges:
python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
TEXT

echo "Showcase complete. Saved terminal log: $LOG"
} 2>&1 | tee "$LOG"

rm -rf __pycache__ adapters/__pycache__
