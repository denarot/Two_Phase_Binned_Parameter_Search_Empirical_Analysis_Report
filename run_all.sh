#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# Full pipeline for:
#   "Two-Phase Search for Optimal Random Forest Tree Count:
#    Seven Theorems Proving O(log N) Optimality"
#
# Usage
# -----
#   bash run_all.sh              # full run (10 seeds, real datasets, ~hours)
#   bash run_all.sh --fast       # 2 seeds, synthetic data (~minutes, for CI /
#                                #   pipeline validation before real experiments)
#   bash run_all.sh --step 2     # resume from step 2 (skip experiments)
#   bash run_all.sh --only 3     # run only step 3 (figures)
#
# Steps
# -----
#   1  Experiments     run_experiments.py all
#   2  Analysis        analysis.py
#   3  Figures         generate_figures.py
#   4  Paper guide     update_paper_values.py
#
# Output
# ------
#   results/   — JSON result files (one per experiment)
#   figures/   — PDF + PNG figures (4 × 2 = 8 files)
#   logs/      — per-step stdout/stderr logs with timestamps
#
# Exit codes
# ----------
#   0  All steps completed successfully, no headline changes
#   1  All steps completed, headline numbers changed (see Step 4 output)
#   2  A step failed — check logs/ for details
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FAST=0
START_STEP=1
ONLY_STEP=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            FAST=1
            shift
            ;;
        --step)
            START_STEP="$2"; shift 2 ;;
        --only)
            ONLY_STEP="$2"; shift 2 ;;
        --help|-h)
            head -40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1  (run with --help for usage)"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

sep()  { printf '%0.s─' {1..74}; echo; }
hsep() { printf '%0.s═' {1..74}; echo; }

log() {
    local ts; ts="$(date '+%H:%M:%S')"
    echo -e "${CYAN}[${ts}]${RESET} $*"
}
ok()   { echo -e "${GREEN}  ✓${RESET}  $*"; }
warn() { echo -e "${YELLOW}  ⚠ ${RESET}  $*"; }
err()  { echo -e "${RED}  ✗${RESET}  $*" >&2; }
bold() { echo -e "${BOLD}$*${RESET}"; }

elapsed() {
    local secs=$1
    printf '%dm %02ds' $((secs/60)) $((secs%60))
}

mkdir -p logs results figures

PIPELINE_START=$SECONDS

# ---------------------------------------------------------------------------
# Step runner — runs a command, tees to log, times it, exits on failure
# ---------------------------------------------------------------------------
run_step() {
    local step_num="$1"; local step_name="$2"; shift 2
    local cmd=("$@")
    local log_file="logs/step${step_num}_$(date '+%Y%m%d_%H%M%S').log"

    # Skip if --step N was set and we haven't reached it
    if (( step_num < START_STEP )); then
        warn "Step ${step_num} skipped (--step ${START_STEP} set)"
        return 0
    fi

    # Skip if --only N was set and this isn't the target step
    if (( ONLY_STEP > 0 && step_num != ONLY_STEP )); then
        return 0
    fi

    hsep
    bold "  Step ${step_num}: ${step_name}"
    log "Command: ${cmd[*]}"
    log "Log:     ${log_file}"
    sep

    local step_start=$SECONDS
    local exit_code=0

    if ! "${cmd[@]}" 2>&1 | tee "$log_file"; then
        exit_code=${PIPESTATUS[0]}
    fi

    local step_elapsed=$(( SECONDS - step_start ))

    if [[ $exit_code -eq 0 ]]; then
        ok "Step ${step_num} completed in $(elapsed $step_elapsed)"
    elif [[ $exit_code -eq 1 && $step_num -eq 4 ]]; then
        # Step 4 exits 1 to signal headline changes — not a pipeline failure
        warn "Step ${step_num} completed with headline change alerts (exit code 1)"
        warn "Review the output above and update abstract/conclusion as directed."
        HEADLINE_ALERTS=1
    else
        err "Step ${step_num} FAILED (exit code ${exit_code}) — check ${log_file}"
        err "Pipeline aborted at step ${step_num}."
        exit 2
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight: check Python and all dependencies are importable
# ---------------------------------------------------------------------------
preflight() {
    hsep
    bold "  Pre-flight checks"
    sep

    local pyver
    pyver="$(python --version 2>&1)"
    log "Python: ${pyver}"

    local minor
    minor="$(python -c 'import sys; print(sys.version_info.minor)')"
    if (( minor < 10 )); then
        warn "Python 3.10+ recommended (found 3.${minor}). Proceeding anyway."
    fi

    local all_ok=1
    for pkg in sklearn numpy scipy matplotlib joblib; do
        if python -c "import ${pkg}" 2>/dev/null; then
            local ver
            ver="$(python -c "import importlib.metadata; print(importlib.metadata.version('${pkg//_/-}'))" 2>/dev/null \
                  || python -c "import ${pkg}; print(getattr(${pkg}, '__version__', '?'))" 2>/dev/null)"
            ok "${pkg} ${ver}"
        else
            err "Missing dependency: ${pkg}  (run: pip install -r requirements.txt)"
            all_ok=0
        fi
    done

    for f in two_phase_search.py baselines.py datasets.py run_experiments.py \
              analysis.py generate_figures.py update_paper_values.py; do
        if [[ -f "$f" ]]; then
            ok "${f}"
        else
            err "Missing project file: ${f}"
            all_ok=0
        fi
    done

    if [[ $all_ok -eq 0 ]]; then
        err "Pre-flight failed. Fix the issues above before running the pipeline."
        exit 2
    fi

    ok "All pre-flight checks passed."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
cd "$SCRIPT_DIR"

echo
hsep
if [[ $FAST -eq 1 ]]; then
    bold "  Two-Phase Search Pipeline  [FAST MODE: 2 seeds, synthetic data]"
    warn "FAST MODE: results are for pipeline validation only, NOT for the paper."
    warn "Remove --fast for real experiments."
else
    bold "  Two-Phase Search Pipeline  [FULL MODE: 10 seeds, real datasets]"
    warn "Full mode downloads datasets on first run and takes several hours."
    warn "Use --fast for a quick pipeline check (~5 minutes)."
fi
log "Started: $(date)"
log "Working directory: $SCRIPT_DIR"
hsep
echo

preflight

HEADLINE_ALERTS=0

# ---------------------------------------------------------------------------
# Step 1: Experiments
# ---------------------------------------------------------------------------
if [[ $FAST -eq 1 ]]; then
    run_step 1 "Run all experiments (fast: 2 seeds, synthetic)" \
        env DATASETS_SYNTHETIC=1 python run_experiments.py all --seeds 2
else
    run_step 1 "Run all experiments (10 seeds, real datasets)" \
        python run_experiments.py all
fi

# ---------------------------------------------------------------------------
# Step 2: Statistical analysis
# ---------------------------------------------------------------------------
run_step 2 "Statistical analysis (Tables 1, 2, Appendix A)" \
    python analysis.py

# ---------------------------------------------------------------------------
# Step 3: Figure generation
# ---------------------------------------------------------------------------
run_step 3 "Generate figures (Figures 0–3)" \
    python generate_figures.py

# ---------------------------------------------------------------------------
# Step 4: Paper update guide
# update_paper_values.py exits 1 for headline changes (not a failure).
# run_step handles exit code 1 specially for step 4; || true prevents
# set -e from aborting before run_step can apply that special-case logic.
# ---------------------------------------------------------------------------
run_step 4 "Generate paper update guide" \
    python update_paper_values.py || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_ELAPSED=$(( SECONDS - PIPELINE_START ))

echo
hsep
bold "  Pipeline complete"
sep
log "Finished: $(date)"
log "Total elapsed: $(elapsed $TOTAL_ELAPSED)"
echo
ok "Results saved to:  results/"
echo "    $(ls results/*.json 2>/dev/null | wc -l | tr -d ' ') JSON files:"
ls results/*.json 2>/dev/null | sed 's/^/    /' || true
echo
ok "Figures saved to:  figures/"
echo "    $(ls figures/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs + $(ls figures/*.png 2>/dev/null | wc -l | tr -d ' ') PNGs"
echo
ok "Step logs in:      logs/"
echo

if [[ $HEADLINE_ALERTS -eq 1 ]]; then
    warn "One or more HEADLINE numbers changed materially."
    warn "Review Step 4 output (or logs/step4_*.log) and update:"
    warn "  · Abstract: speedup range, success rate"
    warn "  · Section 2.3: Covertype speedup and evaluation count"
    warn "  · Section 2.5: audience benefit claims"
    warn "  · Section 9: conclusion summary numbers"
    echo
    exit 1
else
    ok "No headline changes detected. Paper numbers appear consistent."
    echo
    exit 0
fi
