#!/usr/bin/env bash
set -euo pipefail

# Autonomous terminal-video showcase for ANVIL P-04.
#
# Run from Terminal:
#   cd /Users/adityakumarrai/.openclaw/workspace/Anvil-P-E/bench-p04-pcam
#   ./record_showcase.sh
#
# Optional practice mode without screen recording:
#   ./record_showcase.sh --no-record
#
# What it does:
#   1. Shows a countdown.
#   2. Starts macOS screen recording if possible.
#   3. Slowly prints commands and outputs for a readable demo.
#   4. Uses cached report.json/reports.json so video stays under ~5 minutes.
#   5. Saves terminal log and video file.

MODE="record"
if [[ "${1:-}" == "--no-record" ]]; then
  MODE="no-record"
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./record_showcase.sh              # starts screen recording + terminal demo
  ./record_showcase.sh --no-record  # terminal demo only, no recording

If macOS asks for Screen Recording permission, allow Terminal/iTerm/VS Code,
then rerun the script.
EOF
  exit 0
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs videos
STAMP="$(date +%Y%m%d_%H%M%S)"
TERM_LOG="logs/record_showcase_${STAMP}.log"
VIDEO_FILE="videos/p04_showcase_${STAMP}.mov"
REC_PID=""

PY="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif [[ -x "/Users/adityakumarrai/.openclaw/workspace/p04_submission_archive_20260516_082924/final_structure_cleanup/.venv/bin/python" ]]; then
  PY="/Users/adityakumarrai/.openclaw/workspace/p04_submission_archive_20260516_082924/final_structure_cleanup/.venv/bin/python"
fi

TYPE_DELAY="${TYPE_DELAY:-0.018}"
PAUSE="${PAUSE:-1.2}"

cleanup() {
  if [[ -n "${REC_PID:-}" ]] && kill -0 "$REC_PID" 2>/dev/null; then
    kill -INT "$REC_PID" 2>/dev/null || true
    wait "$REC_PID" 2>/dev/null || true
  fi
  rm -rf __pycache__ adapters/__pycache__
}
trap cleanup EXIT INT TERM

slow_print() {
  local text="$1"
  local i ch
  for (( i=0; i<${#text}; i++ )); do
    ch="${text:i:1}"
    printf '%s' "$ch"
    sleep "$TYPE_DELAY"
  done
  printf '\n'
}

cmd() {
  local command="$1"
  slow_print "$ $command"
  sleep 0.35
  eval "$command"
}

section() {
  echo
  echo "================================================================================"
  slow_print "$1"
  echo "================================================================================"
  sleep "$PAUSE"
}

start_recording() {
  if [[ "$MODE" == "no-record" ]]; then
    echo "Recording disabled (--no-record)."
    return 0
  fi

  if ! command -v screencapture >/dev/null 2>&1; then
    echo "screencapture not found; continuing without recording."
    MODE="no-record"
    return 0
  fi

  echo "Starting macOS screen recording..."
  echo "Output video: $VIDEO_FILE"
  echo "If macOS asks permission, allow screen recording and rerun."

  # macOS screencapture video mode. This records until interrupted.
  # Some macOS versions require selecting a region/window interactively.
  # We run it in the background and stop it at script end.
  (screencapture -v "$VIDEO_FILE") &
  REC_PID=$!
  sleep 2

  if ! kill -0 "$REC_PID" 2>/dev/null; then
    echo "Screen recording did not stay running. Continuing terminal demo only."
    REC_PID=""
    MODE="no-record"
  fi
}

run_demo() {
  clear
  section "ANVIL P-04 · PCAM PRECISION AGENT · RECORDED TERMINAL SHOWCASE"
  echo "Folder: $ROOT"
  echo "Adapter: adapters.theVisioneers:Engine"
  echo "Official output: report.json"
  echo "Evidence output: reports.json"
  echo "Terminal log: $TERM_LOG"
  echo "Video file: $VIDEO_FILE"
  sleep 1

  section "COUNTDOWN"
  for n in 5 4 3 2 1; do
    echo "Starting demo in $n..."
    sleep 0.7
  done

  start_recording
  sleep 1

  section "1 / 8 · SHOW CLEAN SUBMISSION FILES"
  cmd "find . -maxdepth 2 -type f ! -path './logs/*' ! -path './videos/*' | sed 's#^./##' | sort"
  sleep "$PAUSE"

  section "2 / 8 · SHOW SUBMISSION AGENT FILE"
  cmd "sed -n '1,220p' adapters/theVisioneers.py"
  sleep "$PAUSE"

  section "3 / 8 · SHOW README SETUP AND RUN COMMANDS"
  cmd "sed -n '1,220p' README.md"
  sleep "$PAUSE"

  section "4 / 8 · CODE QUALITY CHECK"
  cmd "$PY -m py_compile adapter.py adapters/theVisioneers.py data.py harness.py metrics.py pcam_model.py run.py self_check.py"
  cmd "$PY -m json.tool report.json >/dev/null && $PY -m json.tool reports.json >/dev/null && echo 'JSON reports valid'"
  sleep "$PAUSE"

  section "5 / 8 · OFFICIAL 7-SEED RESULT FROM report.json"
  slow_print "$ python run.py --adapter adapters.theVisioneers:Engine --seeds 7 13 31 97 211 503 1009 --out report.json"
  echo "(Already generated; showing cached official result to keep video short.)"
  "$PY" - <<'PY'
import json
r=json.load(open('report.json'))
agg=r['aggregated']; score=r['score']
print('Seeds:', r['config']['seeds'])
print(f"Mean delta: {agg['mean_delta']:+.4f}")
print(f"Worst-seed delta: {agg['min_delta']:+.4f}")
print(f"Mean spread reduction: {agg['mean_reduction']:.2f}x")
print(f"Retrieval: {score['retrieval_pts']:.2f} / 70")
print(f"Anisotropy: {score['anisotropy_pts']:.2f} / 20")
print(f"Total automated: {score['total_automated']:.2f} / {score['max_automated']:.0f}")
print('\nPer-seed:')
print(f"{'seed':>6} {'direct':>8} {'base':>8} {'agent':>8} {'delta':>9} {'reduction':>10}")
for row in r['per_seed']:
    print(f"{row['seed']:>6} {row['direct_classify_acc']:>8.3f} {row['baseline_acc']:>8.3f} {row['agent_acc']:>8.3f} {row['delta']:>+9.3f} {row['spread_reduction']:>9.2f}x")
PY
  sleep "$PAUSE"

  section "6 / 8 · QUICK LIVE SELF-CHECK"
  cmd "$PY self_check.py --adapter adapters.theVisioneers:Engine --quick"
  sleep "$PAUSE"

  section "7 / 8 · TRANSFER DATASETS FROM reports.json"
  "$PY" - <<'PY'
import json
r=json.load(open('reports.json'))
print(f"{'dataset':<18} {'baseline':>14} {'agent':>14} {'delta':>10}")
print('-'*62)
for name,item in r['transfer_tests'].items():
    b=f"{item['baseline']}={item['baseline_acc']:.4f}"
    a=f"{item['agent']}={item['agent_acc']:.4f}"
    print(f"{name:<18} {b:>14} {a:>14} {item['delta']:>+10.4f}")
PY
  sleep "$PAUSE"

  section "8 / 8 · STRESS TEST SUMMARY + SUBMISSION FILES"
  "$PY" - <<'PY'
import json
r=json.load(open('reports.json'))
s=r['stress_tests']['current_agent']['summary']
anti=r['stress_tests']['anti_benchmark_corruptions_current']
print('Hidden-style stress matrix summary:')
print(f"worst_min_delta={s['worst_min_delta']:+.4f}")
print(f"mean_total_score={s['mean_total_score']:.2f}/90")
print(f"mean_reduction={s['mean_reduction']:.2f}x")
print('\nAnti-benchmark corruptions:')
print(f"baseline={anti['overall_baseline_acc']:.3f}")
print(f"agent={anti['overall_agent_acc']:.3f}")
print(f"delta={anti['overall_delta']:+.3f}")
print(f"min_seedkind_delta={anti['min_seedkind_delta']:+.3f}")
PY
  echo
  echo "Submit these key files:"
  echo "  adapters/theVisioneers.py"
  echo "  README.md"
  echo "  report.json"
  echo "  reports.json"
  echo
  echo "Judges can run:"
  echo "  python run.py --adapter adapters.theVisioneers:Engine --seeds 7 13 31 97 211 503 1009 --out report.json"
  sleep 1

  section "DONE"
  echo "Terminal log saved to: $TERM_LOG"
  if [[ "$MODE" == "record" ]]; then
    echo "Video saved to: $VIDEO_FILE"
  fi
}

run_demo 2>&1 | tee "$TERM_LOG"
