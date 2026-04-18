"""
verify_results.py
-----------------
Runs seven sanity checks on results/ before updating the paper.
Designed to catch numerical anomalies, theorem violations, and
data-integrity issues that would otherwise silently corrupt the paper.

All checks are independent — a failure in one does not abort the others.
The final line of output is the overall verdict: ALL CHECKS PASSED or
N CHECK(S) FAILED, along with an exit code (0 = all pass, 1 = any fail).

Usage
-----
  python verify_results.py              # all checks
  python verify_results.py --check 3   # check 3 only (phase count)
  python verify_results.py --verbose   # show per-run detail for every check

Checks
------
  1  Monotonicity          (Section 7.1 / Figure 0 validation criterion)
  2  Evaluation count      (Theorem 4 total bound)
  3  Phase count           (Theorems 1 + 3 per-phase bounds)
  4  Accuracy preservation (abstract ε-optimality claim, |Δacc| < 0.01)
  5  Bidirectional conv.   (Section 7.6, relative diff < 0.10)
  6  Budget compliance     (Table 6 memory / 4 MB flag consistency)
  7  Success rate          (aggregate computation correctness)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants — must match run_experiments.py
# ---------------------------------------------------------------------------
K_MIN_DEFAULT    = 10
K_MAX_DEFAULT    = 1000
SUCCESS_THRESH   = 10      # |k_two_phase − k_grid| ≤ 10 counts as success
MONO_TOLERANCE   = 0.005   # plateau variation tolerance (Section 7.1)
ACC_TOLERANCE    = 0.01    # accuracy preservation tolerance (abstract)
BIDIR_THRESHOLD  = 0.10    # bidirectional relative diff threshold (Section 7.6)
EDGE_BUDGET_MB   = 4.0     # flash budget for edge deployment (Section 7.7)
RESULTS_DIR      = "results"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
GREEN  = "\033[0;32m"; RED   = "\033[0;31m"
YELLOW = "\033[1;33m"; CYAN  = "\033[0;36m"
BOLD   = "\033[1m";    RESET = "\033[0m"

def _pass(msg=""): return f"{GREEN}PASS{RESET}" + (f"  {msg}" if msg else "")
def _fail(msg=""): return f"{RED}FAIL{RESET}" + (f"  {msg}" if msg else "")
def _warn(msg=""): return f"{YELLOW}WARN{RESET}" + (f"  {msg}" if msg else "")
def _info(msg):    return f"{CYAN}{msg}{RESET}"

def _sep(c="─", w=72): print(c * w)
def _hsep(): _sep("═")

# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    check_id:   int
    name:       str
    passed:     bool
    warnings:   List[str] = field(default_factory=list)
    failures:   List[str] = field(default_factory=list)
    details:    List[str] = field(default_factory=list)
    skipped:    bool = False
    skip_reason: str = ""

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load(name: str) -> Optional[dict]:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def _infer_bracket(run: dict) -> Tuple[int, int]:
    """
    Recover the Phase 1 bracket [L, U] passed to Phase 2.

    Phase 1 exits when plateau detected at k_next; at that point
    L = k_current (second-to-last Phase 1 evaluation) and
    U = k_next    (last Phase 1 evaluation).
    Falls back to a 1-wide bracket if only one Phase 1 point exists.
    """
    traj = run["trajectory"]
    p1_ks = [t["k"] for t in traj if t["phase"] == 1]
    if len(p1_ks) >= 2:
        return int(p1_ks[-2]), int(p1_ks[-1])
    if p1_ks:
        return int(p1_ks[0]), int(p1_ks[0]) + 1
    return K_MIN_DEFAULT, K_MAX_DEFAULT

# ---------------------------------------------------------------------------
# Check 1 — Monotonicity
# ---------------------------------------------------------------------------
def check_monotonicity(verbose: bool) -> CheckResult:
    result = CheckResult(1, "Monotonicity (Section 7.1 / Figure 0)", passed=True)
    data = _load("monotonicity")
    if data is None:
        result.skipped = True
        result.skip_reason = "monotonicity.json not found"
        return result

    for ds_name, ds in data.items():
        mean_curve = np.array(ds["mean_curve"], dtype=float)
        std_curve  = np.array(ds["std_curve"],  dtype=float)
        k_values   = ds["k_values"]

        # (a) Mean curve non-decreasing within tolerance
        violations = []
        for i in range(len(mean_curve) - 1):
            drop = mean_curve[i] - mean_curve[i + 1]
            if drop > MONO_TOLERANCE:
                violations.append(
                    f"k={k_values[i]}→{k_values[i+1]}: "
                    f"acc dropped {drop:.5f} > tolerance {MONO_TOLERANCE}"
                )
        if violations:
            result.passed = False
            for v in violations:
                result.failures.append(f"[{ds_name}] Monotonicity violation: {v}")
        elif verbose:
            max_drop = max(
                (mean_curve[i] - mean_curve[i + 1] for i in range(len(mean_curve) - 1)),
                default=0.0,
            )
            result.details.append(
                f"[{ds_name}] mean curve non-decreasing ✓  "
                f"(max drop = {max_drop:.5f})"
            )

        # (b) Plateau variation < MONO_TOLERANCE in last 4 k values
        plateau_var = float(np.std(mean_curve[-4:]))
        if plateau_var >= MONO_TOLERANCE:
            result.warnings.append(
                f"[{ds_name}] Plateau variation {plateau_var:.5f} ≥ {MONO_TOLERANCE} "
                f"(Assumption 1 may be borderline; check Section 7.1)"
            )
        elif verbose:
            result.details.append(
                f"[{ds_name}] plateau variation {plateau_var:.5f} < {MONO_TOLERANCE} ✓"
            )

        # (c) Paper's stored flag agrees
        if not ds.get("is_monotone") and not violations:
            result.warnings.append(
                f"[{ds_name}] is_monotone=False in JSON but no violations detected "
                f"within tolerance — check raw curves"
            )

    return result


# ---------------------------------------------------------------------------
# Check 2 — Total Evaluation Count vs Theorem 4 Bound
# ---------------------------------------------------------------------------
def check_evaluation_count(verbose: bool) -> CheckResult:
    result = CheckResult(2, "Evaluation count ≤ Theorem 4 bound", passed=True)
    data = _load("primary_comparison")
    if data is None:
        result.skipped = True
        result.skip_reason = "primary_comparison.json not found"
        return result

    # Theorem 4: T_total ≤ 2·⌈log₂(N)⌉ + 1 where N = K_MAX − K_MIN.
    # Add 1 margin for window-filling evaluations Theorem 4 doesn't count.
    n = K_MAX_DEFAULT - K_MIN_DEFAULT
    bound = 2 * math.ceil(math.log2(max(n, 2))) + 2

    for ds_name, ds in data.items():
        for run in ds["two_phase"]["runs"]:
            actual = run["total_evaluations"]
            if actual > bound:
                result.passed = False
                result.failures.append(
                    f"[{ds_name} seed={run['seed']}] "
                    f"total_evaluations={actual} > bound={bound} "
                    f"(K_MIN={K_MIN_DEFAULT}, K_MAX={K_MAX_DEFAULT}, N={n})"
                )
            elif verbose:
                result.details.append(
                    f"[{ds_name} seed={run['seed']}] "
                    f"evals={actual} ≤ bound={bound} ✓"
                )

    return result


# ---------------------------------------------------------------------------
# Check 3 — Per-phase Evaluation Counts vs Theorems 1 + 3
# ---------------------------------------------------------------------------
def check_phase_counts(verbose: bool) -> CheckResult:
    result = CheckResult(3, "Phase counts ≤ Theorems 1 + 3 bounds", passed=True)
    data = _load("primary_comparison")
    if data is None:
        result.skipped = True
        result.skip_reason = "primary_comparison.json not found"
        return result

    # Theorem 1: Phase 1 ≤ ⌈log₂(K_MAX / K_MIN)⌉ + 1
    p1_bound = math.ceil(math.log2(K_MAX_DEFAULT / K_MIN_DEFAULT)) + 1

    for ds_name, ds in data.items():
        for run in ds["two_phase"]["runs"]:
            L, U = _infer_bracket(run)
            p1 = run["phase1_evaluations"]
            p2 = run["phase2_evaluations"]

            # Theorem 3: Phase 2 ≤ ⌈log₂(U − L)⌉ (0 if bracket width ≤ 1)
            bracket_width = max(U - L, 1)
            p2_bound = math.ceil(math.log2(bracket_width)) if bracket_width > 1 else 0

            seed = run["seed"]
            ok_p1 = p1 <= p1_bound
            ok_p2 = p2 <= p2_bound

            if not ok_p1:
                result.passed = False
                result.failures.append(
                    f"[{ds_name} seed={seed}] Phase 1: "
                    f"actual={p1} > bound={p1_bound} "
                    f"(K_MIN={K_MIN_DEFAULT}, K_MAX={K_MAX_DEFAULT})"
                )
            if not ok_p2:
                result.passed = False
                result.failures.append(
                    f"[{ds_name} seed={seed}] Phase 2: "
                    f"actual={p2} > bound={p2_bound} "
                    f"(bracket=[{L},{U}], width={bracket_width})"
                )
            if verbose and ok_p1 and ok_p2:
                result.details.append(
                    f"[{ds_name} seed={seed}] "
                    f"P1={p1}≤{p1_bound} ✓  P2={p2}≤{p2_bound} ✓"
                )

    return result


# ---------------------------------------------------------------------------
# Check 4 — Accuracy Preservation (|Δacc| < 0.01)
# ---------------------------------------------------------------------------
def check_accuracy_preservation(verbose: bool) -> CheckResult:
    result = CheckResult(
        4, f"Accuracy preservation |Δacc| < {ACC_TOLERANCE} (abstract claim)",
        passed=True
    )
    data = _load("primary_comparison")
    if data is None:
        result.skipped = True
        result.skip_reason = "primary_comparison.json not found"
        return result

    for ds_name, ds in data.items():
        grid_by_seed = {r["seed"]: r for r in ds["grid"]["runs"]}
        tp_by_seed   = {r["seed"]: r for r in ds["two_phase"]["runs"]}

        common_seeds = set(grid_by_seed) & set(tp_by_seed)
        if not common_seeds:
            result.warnings.append(
                f"[{ds_name}] No overlapping seeds between grid and two_phase runs"
            )
            continue

        gaps = []
        for seed in sorted(common_seeds):
            grid_acc = grid_by_seed[seed]["cv_accuracy"]
            tp_acc   = tp_by_seed[seed]["cv_accuracy"]
            gap = abs(tp_acc - grid_acc)
            gaps.append(gap)

            if gap >= ACC_TOLERANCE:
                result.passed = False
                result.failures.append(
                    f"[{ds_name} seed={seed}] "
                    f"|Δacc| = {gap:.5f} ≥ {ACC_TOLERANCE}  "
                    f"(grid={grid_acc:.5f}, two_phase={tp_acc:.5f})"
                )
            elif verbose:
                result.details.append(
                    f"[{ds_name} seed={seed}] "
                    f"|Δacc| = {gap:.5f} < {ACC_TOLERANCE} ✓"
                )

        if gaps:
            mean_gap = float(np.mean(gaps))
            max_gap  = float(np.max(gaps))
            if verbose or max_gap >= ACC_TOLERANCE * 0.8:
                result.details.append(
                    f"[{ds_name}] mean |Δacc|={mean_gap:.5f}  "
                    f"max |Δacc|={max_gap:.5f}  "
                    f"({'within' if max_gap < ACC_TOLERANCE else 'EXCEEDS'} tolerance)"
                )

    return result


# ---------------------------------------------------------------------------
# Check 5 — Bidirectional Convergence
# ---------------------------------------------------------------------------
def check_bidirectional(verbose: bool) -> CheckResult:
    result = CheckResult(
        5, f"Bidirectional convergence |Δk|/max < {BIDIR_THRESHOLD} (Section 7.6)",
        passed=True
    )
    data = _load("bidirectional_validation")
    if data is None:
        result.skipped = True
        result.skip_reason = "bidirectional_validation.json not found"
        return result

    for ds_name, ds in data.items():
        runs = ds["runs"]
        rel_diffs = [r["k_rel_diff"] for r in runs]
        failures  = [r for r in runs if r["k_rel_diff"] >= BIDIR_THRESHOLD]

        if failures:
            result.passed = False
            for r in failures:
                result.failures.append(
                    f"[{ds_name} seed={r['seed']}] "
                    f"|k_fwd={r['k_forward']} − k_rev={r['k_reverse']}| "
                    f"/ max = {r['k_rel_diff']:.4f} ≥ {BIDIR_THRESHOLD}"
                )
        else:
            mean_rel = float(np.mean(rel_diffs))
            max_rel  = float(np.max(rel_diffs))
            if verbose:
                result.details.append(
                    f"[{ds_name}] mean rel_diff={mean_rel:.4f}  "
                    f"max={max_rel:.4f} < {BIDIR_THRESHOLD} ✓"
                )

        # Sanity: stored rel_diff_mean matches computed value
        stored_mean = ds.get("rel_diff_mean")
        if stored_mean is not None:
            computed_mean = float(np.mean(rel_diffs))
            if abs(stored_mean - computed_mean) > 1e-6:
                result.warnings.append(
                    f"[{ds_name}] stored rel_diff_mean={stored_mean:.6f} "
                    f"≠ recomputed={computed_mean:.6f} — "
                    f"JSON aggregates may be stale"
                )

    return result


# ---------------------------------------------------------------------------
# Check 6 — Budget Compliance Consistency
# ---------------------------------------------------------------------------
def check_budget_compliance(verbose: bool) -> CheckResult:
    result = CheckResult(6, "Budget compliance flags consistent (Table 6)", passed=True)
    data = _load("edge_deployment")
    if data is None:
        result.skipped = True
        result.skip_reason = "edge_deployment.json not found"
        return result

    for ds_name, ds in data.items():
        for run in ds["runs"]:
            seed = run["seed"]
            for config, metrics in run["metrics"].items():
                size_mb  = metrics["size_mb"]
                flag     = metrics["within_4mb_budget"]
                expected = size_mb <= EDGE_BUDGET_MB

                if flag != expected:
                    result.passed = False
                    result.failures.append(
                        f"[{ds_name} seed={seed} config={config}] "
                        f"size={size_mb:.3f}MB but within_4mb_budget={flag} "
                        f"(expected {expected})"
                    )
                elif verbose:
                    result.details.append(
                        f"[{ds_name} seed={seed} config={config}] "
                        f"size={size_mb:.3f}MB → "
                        f"{'fits' if flag else 'exceeds'} 4MB ✓"
                    )

        # Aggregate compliance_rate consistency
        for config in ["grid", "two_phase", "default"]:
            agg_key = f"agg_{config}"
            if agg_key not in ds:
                continue
            agg = ds[agg_key]
            stored_rate = agg.get("budget_compliance_rate")
            if stored_rate is None:
                continue

            flags = [
                run["metrics"][config]["within_4mb_budget"]
                for run in ds["runs"]
                if config in run["metrics"]
            ]
            if not flags:
                continue
            computed_rate = float(np.mean(flags))

            if abs(stored_rate - computed_rate) > 1e-6:
                result.warnings.append(
                    f"[{ds_name} {config}] stored budget_compliance_rate="
                    f"{stored_rate:.4f} ≠ recomputed={computed_rate:.4f}"
                )
            elif verbose:
                result.details.append(
                    f"[{ds_name} {config}] compliance_rate={computed_rate:.2f} ✓"
                )

    return result


# ---------------------------------------------------------------------------
# Check 7 — Success Rate Computation
# ---------------------------------------------------------------------------
def check_success_rate(verbose: bool) -> CheckResult:
    result = CheckResult(7, "Success rate computation correct", passed=True)
    data = _load("primary_comparison")
    if data is None:
        result.skipped = True
        result.skip_reason = "primary_comparison.json not found"
        return result

    for ds_name, ds in data.items():
        grid_ks = [r["k_hat"] for r in ds["grid"]["runs"]]
        tp_ks   = [r["k_hat"] for r in ds["two_phase"]["runs"]]

        if len(grid_ks) != len(tp_ks):
            result.warnings.append(
                f"[{ds_name}] grid has {len(grid_ks)} runs, "
                f"two_phase has {len(tp_ks)} — seed mismatch"
            )
            continue

        successes = [
            abs(tp - gd) <= SUCCESS_THRESH
            for tp, gd in zip(tp_ks, grid_ks)
        ]
        computed_rate = float(np.mean(successes))
        stored_rate   = ds["two_phase"].get("success_rate")

        if stored_rate is not None and abs(computed_rate - stored_rate) > 1e-6:
            result.passed = False
            result.failures.append(
                f"[{ds_name}] stored success_rate={stored_rate:.6f} "
                f"≠ recomputed={computed_rate:.6f} "
                f"(threshold |k_tp − k_grid| ≤ {SUCCESS_THRESH})"
            )
        elif verbose:
            result.details.append(
                f"[{ds_name}] success_rate={computed_rate:.4f} "
                f"(stored={stored_rate:.4f}) ✓  "
                f"successes={sum(successes)}/{len(successes)}"
            )

        # Bonus: check aggregate speedup arithmetic
        grid_evals = float(np.mean([r["total_evaluations"] for r in ds["grid"]["runs"]]))
        tp_evals   = float(np.mean([r["total_evaluations"] for r in ds["two_phase"]["runs"]]))
        if tp_evals > 0:
            computed_speedup = grid_evals / tp_evals
            stored_speedup   = ds["two_phase"].get("speedup")
            if stored_speedup is not None and abs(computed_speedup - stored_speedup) > 0.05:
                result.warnings.append(
                    f"[{ds_name}] stored speedup={stored_speedup:.2f}× "
                    f"≠ recomputed={computed_speedup:.2f}×"
                )

    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def _print_result(r: CheckResult, verbose: bool) -> None:
    _sep()
    if r.skipped:
        label = _warn("SKIP")
    elif r.passed:
        label = _pass()
    else:
        label = _fail()

    print(f"  Check {r.check_id}: {r.name}")
    print(f"  Result: {label}")

    if r.skipped:
        print(f"  Reason: {r.skip_reason}")

    for line in r.failures:
        print(f"    {RED}✗{RESET} {line}")

    for line in r.warnings:
        print(f"    {YELLOW}⚠{RESET} {line}")

    if verbose:
        for line in r.details:
            print(f"    {CYAN}·{RESET} {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_CHECKS = [
    check_monotonicity,
    check_evaluation_count,
    check_phase_counts,
    check_accuracy_preservation,
    check_bidirectional,
    check_budget_compliance,
    check_success_rate,
]


def run_checks(check_ids: Optional[List[int]], verbose: bool) -> int:
    """Run the requested checks and return exit code (0=all pass, 1=any fail)."""
    print()
    _hsep()
    print(f"{BOLD}  Two-Phase Search — Results Verification{RESET}")
    print(f"  Results directory: {os.path.abspath(RESULTS_DIR)}")
    _hsep()

    # Inventory
    files = [
        "primary_comparison", "monotonicity", "scaling",
        "noise_robustness", "bidirectional_validation",
        "edge_deployment", "window_ablation",
    ]
    print()
    print("  Available result files:")
    for f in files:
        path = os.path.join(RESULTS_DIR, f"{f}.json")
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    {GREEN}✓{RESET}  {f}.json  ({size:,} bytes)")
        else:
            print(f"    {YELLOW}─{RESET}  {f}.json  (not yet available)")
    print()

    results: List[CheckResult] = []
    for i, check_fn in enumerate(ALL_CHECKS, start=1):
        if check_ids and i not in check_ids:
            continue
        r = check_fn(verbose=verbose)
        _print_result(r, verbose)
        results.append(r)

    # Summary
    _hsep()
    print(f"{BOLD}  Summary{RESET}")
    _sep()

    passed   = [r for r in results if r.passed and not r.skipped]
    failed   = [r for r in results if not r.passed and not r.skipped]
    skipped  = [r for r in results if r.skipped]
    warnings = [r for r in results if r.warnings]

    for r in results:
        if r.skipped:
            status = _warn("SKIP")
        elif r.passed:
            status = _pass()
        else:
            status = _fail()
        print(f"  {status}  Check {r.check_id}: {r.name}")

    print()
    print(f"  Passed:  {len(passed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Skipped: {len(skipped)}  (result files not yet available)")
    if warnings:
        print(f"  Checks with warnings: {', '.join(str(r.check_id) for r in warnings)}")
    print()

    if failed:
        print(f"{RED}{BOLD}  ✗  {len(failed)} CHECK(S) FAILED — do not update the paper.{RESET}")
        print()
        print("  Failed checks and what to do:")
        for r in failed:
            print(f"    Check {r.check_id}: {r.name}")
            for line in r.failures[:3]:
                print(f"      · {line}")
            if len(r.failures) > 3:
                print(f"      … and {len(r.failures)-3} more (run --verbose for details)")
        print()
        print("  Common causes:")
        ids = {r.check_id for r in failed}
        if 1 in ids:
            print("    Check 1: dataset doesn't satisfy Assumption 1 — exclude it.")
        if 2 in ids or 3 in ids:
            print("    Checks 2/3: window-filling evals exceeded; "
                  "check epsilon/window_size config.")
        if 4 in ids:
            print("    Check 4: Two-Phase found a suboptimal k — "
                  "investigate plateau detection threshold.")
        if 5 in ids:
            print("    Check 5: forward/reverse disagree — "
                  "inspect individual trajectories.")
        if 6 in ids:
            print("    Check 6: serialised model sizes differ from paper projections — "
                  "update Table 6.")
        if 7 in ids:
            print("    Check 7: JSON aggregates are stale — re-run run_experiments.py.")
        _hsep()
        return 1
    elif skipped and not passed:
        print(f"{YELLOW}{BOLD}  ─  All checks skipped (no results yet).{RESET}")
        print("  Run the experiments first: python run_experiments.py all")
        _hsep()
        return 0
    else:
        print(f"{GREEN}{BOLD}  ✓  ALL CHECKS PASSED.{RESET}")
        if skipped:
            print(f"     ({len(skipped)} skipped — those experiments haven't run yet)")
        print("  Results appear internally consistent. Proceed to update_paper_values.py.")
        _hsep()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check all results before updating the paper."
    )
    parser.add_argument(
        "--check", type=int, nargs="+", metavar="N",
        help="Run only check(s) N (e.g. --check 2 3). Default: all."
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-run details for every check, not just failures."
    )
    args = parser.parse_args()
    sys.exit(run_checks(args.check, args.verbose))
