#!/usr/bin/env bash
set -euo pipefail

# One-command Ubuntu performance prep + OpenMC worker auto-benchmark for this repo.
# Usage examples:
#   bash scripts/ubuntu_perf_prep_and_autobench.sh
#   bash scripts/ubuntu_perf_prep_and_autobench.sh --workers "4 6 8" --points 8
#   bash scripts/ubuntu_perf_prep_and_autobench.sh --prep-only

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
WORKER_CANDIDATES="4 6 8"
BENCH_POINTS=8
BENCH_PARTICLES=6000
BENCH_BATCHES=40
BENCH_INACTIVE=10
RUNS_DIR="runs"
PREP_ONLY=0
SKIP_PREP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --workers)
      WORKER_CANDIDATES="$2"
      shift 2
      ;;
    --points)
      BENCH_POINTS="$2"
      shift 2
      ;;
    --particles)
      BENCH_PARTICLES="$2"
      shift 2
      ;;
    --batches)
      BENCH_BATCHES="$2"
      shift 2
      ;;
    --inactive)
      BENCH_INACTIVE="$2"
      shift 2
      ;;
    --runs-dir)
      RUNS_DIR="$2"
      shift 2
      ;;
    --prep-only)
      PREP_ONLY=1
      shift
      ;;
    --skip-prep)
      SKIP_PREP=1
      shift
      ;;
    -h|--help)
      sed -n '1,35p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

has_openmc() {
  local py_bin="$1"
  "$py_bin" - <<'PY' >/dev/null 2>&1
import openmc
print(openmc.__version__)
PY
}

if [[ -z "$PYTHON_BIN" ]]; then
  candidates=()
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if [[ -x "/home/dina/miniconda3/envs/openmc-env/bin/python" ]]; then
    candidates+=("/home/dina/miniconda3/envs/openmc-env/bin/python")
  fi

  for cand in "${candidates[@]}"; do
    if has_openmc "$cand"; then
      PYTHON_BIN="$cand"
      break
    fi
  done

  if [[ -z "$PYTHON_BIN" ]]; then
    echo "No Python interpreter with OpenMC found." >&2
    echo "Use --python /path/to/python from your OpenMC environment." >&2
    exit 1
  fi
else
  require_cmd "$PYTHON_BIN"
  if ! has_openmc "$PYTHON_BIN"; then
    echo "The provided Python does not have OpenMC installed: $PYTHON_BIN" >&2
    echo "Use --python /path/to/python from your OpenMC environment." >&2
    exit 1
  fi
fi

echo "[bench] Using Python interpreter: $PYTHON_BIN"

PY_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
export PATH="$PY_BIN_DIR:$PATH"

if ! command -v openmc >/dev/null 2>&1; then
  echo "OpenMC executable not found on PATH." >&2
  echo "Expected it in: $PY_BIN_DIR" >&2
  echo "Install/activate your OpenMC runtime or pass --python from an environment that includes the openmc executable." >&2
  exit 1
fi

echo "[bench] Using OpenMC executable: $(command -v openmc)"

if [[ ! -f "openmc_apr1400_phase1_runner_final.py" ]]; then
  echo "Run this from the repository root containing openmc_apr1400_phase1_runner_final.py" >&2
  exit 1
fi

run_sudo() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    "$@"
  fi
}

apply_perf_prep() {
  echo "[prep] Applying performance-focused settings..."

  if command -v powerprofilesctl >/dev/null 2>&1; then
    run_sudo powerprofilesctl set performance || true
  else
    echo "[prep] powerprofilesctl not available; skipping profile switch."
  fi

  # Try forcing governor to performance if cpufreq controls are present.
  if compgen -G "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor" >/dev/null; then
    while IFS= read -r gov_file; do
      run_sudo sh -c "echo performance > '$gov_file'" || true
    done < <(ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null)
  else
    echo "[prep] cpufreq governor files not found; skipping governor pinning."
  fi

  # Lower swappiness to reduce chance of paging during long CPU-heavy workloads.
  run_sudo sysctl vm.swappiness=10 >/dev/null || true

  echo "[prep] Current profile/governor/memory snapshot:"
  (powerprofilesctl get 2>/dev/null || echo "powerprofilesctl: unavailable")
  (cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "governor: unavailable")
  free -h | sed -n '1,3p'
  (cat /proc/sys/vm/swappiness 2>/dev/null | awk '{print "swappiness=" $1}') || true
}

if [[ "$SKIP_PREP" -eq 0 ]]; then
  apply_perf_prep
else
  echo "[prep] Skipped by --skip-prep"
fi

if [[ "$PREP_ONLY" -eq 1 ]]; then
  echo "[done] Performance prep completed (benchmark skipped)."
  exit 0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
bench_root="$RUNS_DIR/benchmark_workers_${timestamp}"
mkdir -p "$bench_root"

results_csv="$bench_root/worker_benchmark_summary.csv"
echo "worker,elapsed_sec,n_success,cases_per_hour,run_dir" > "$results_csv"

echo "[bench] Worker candidates: $WORKER_CANDIDATES"
echo "[bench] Benchmark settings: points=$BENCH_POINTS particles=$BENCH_PARTICLES batches=$BENCH_BATCHES inactive=$BENCH_INACTIVE"

design_csv="$bench_root/benchmark_design_safe_temps.csv"
"$PYTHON_BIN" - <<PY
import csv

points = max(4, int("""$BENCH_POINTS"""))
alpha_lo, alpha_hi = 0.25, 1.25
boron_lo, boron_hi = 0.0, 2200.0
fuel_choices = [600.0, 900.0]
moderator_choices = [294.0, 600.0]

rows = []
for i in range(points):
  t = 0.0 if points == 1 else i / (points - 1)
  alpha = alpha_lo + t * (alpha_hi - alpha_lo)
  boron = boron_lo + t * (boron_hi - boron_lo)
  fuel_temp = fuel_choices[i % len(fuel_choices)]
  moderator_temp = moderator_choices[(i // len(fuel_choices)) % len(moderator_choices)]
  if moderator_temp > fuel_temp:
    moderator_temp = 294.0
  rows.append({
    "alpha_gd": alpha,
    "fuel_temp": fuel_temp,
    "moderator_temp": moderator_temp,
    "boron_ppm": boron,
    "replicate": False,
    "region_tag": "worker_benchmark_safe_temps",
  })

out_path = r'''$design_csv'''
with open(out_path, "w", newline="", encoding="utf-8") as f:
  writer = csv.DictWriter(
    f,
    fieldnames=["alpha_gd", "fuel_temp", "moderator_temp", "boron_ppm", "replicate", "region_tag"],
  )
  writer.writeheader()
  writer.writerows(rows)

print(out_path)
PY
echo "[bench] Generated temperature-safe design CSV: $design_csv"

best_worker=""
best_cph="0"

for w in $WORKER_CANDIDATES; do
  run_dir="$bench_root/w${w}"
  log_file="$bench_root/w${w}.log"
  mkdir -p "$run_dir"

  echo "[bench] Running worker=$w ..."
  start_s="$(date +%s)"
  set +e
  "$PYTHON_BIN" openmc_apr1400_phase1_runner_final.py \
    --design-csv "$design_csv" \
    --workers "$w" \
    --threads 1 \
    --particles "$BENCH_PARTICLES" \
    --batches "$BENCH_BATCHES" \
    --inactive "$BENCH_INACTIVE" \
    --run-output-dir "$run_dir" > "$log_file" 2>&1
  rc=$?
  set -e
  end_s="$(date +%s)"
  elapsed="$((end_s - start_s))"

  dataset_csv="$run_dir/processed_data/openmc_dataset.csv"
  n_success=0
  if [[ $rc -eq 0 && -f "$dataset_csv" ]]; then
    n_success="$($PYTHON_BIN - <<PY
import csv
path = r'''$dataset_csv'''
count = 0
with open(path, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        v = str(row.get('success', '')).strip().lower()
        if v in {'1', 'true', 'yes', 'y'}:
            count += 1
print(count)
PY
)"
  fi

  cph="$($PYTHON_BIN - <<PY
elapsed = max(1, int($elapsed))
ns = int($n_success)
print(f"{(ns * 3600.0) / elapsed:.2f}")
PY
)"

  echo "$w,$elapsed,$n_success,$cph,$run_dir" >> "$results_csv"

  if [[ $rc -ne 0 ]]; then
    echo "[bench] worker=$w failed (rc=$rc). See $log_file"
    continue
  fi

  if [[ "$n_success" -eq 0 ]]; then
    echo "[bench] worker=$w completed but had 0 successful OpenMC cases. See $run_dir/reports/result_summary.json and $run_dir/logs/*_stderr.log"
    continue
  fi

  better="$($PYTHON_BIN - <<PY
print(1 if float('$cph') > float('$best_cph') else 0)
PY
)"
  if [[ "$better" == "1" ]]; then
    best_cph="$cph"
    best_worker="$w"
  fi

  echo "[bench] worker=$w elapsed=${elapsed}s success=$n_success cases_per_hour=$cph"
done

echo
echo "[bench] Summary CSV: $results_csv"
column -s, -t "$results_csv" || cat "$results_csv"

if [[ -z "$best_worker" ]]; then
  echo "[bench] No successful benchmark run found."
  exit 1
fi

echo
echo "[bench] Recommended worker count for this machine: $best_worker (benchmark CPH=$best_cph)"
echo "[bench] Suggested full-run command:"
echo "$PYTHON_BIN run_apr1400_pipeline.py --full --workers $best_worker --threads 1 --n-unique-full 48 --particles 15000 --batches 100 --inactive 25 --cv-folds 10"
echo
echo "[note] Benchmark settings are intentionally lighter than production to finish quickly. If two workers are close, prefer the lower worker count for better thermals/stability."
