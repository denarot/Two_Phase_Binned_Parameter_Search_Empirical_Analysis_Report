"""
analysis.py
-----------
Loads JSON results from results/ and computes all derived statistics
needed for the paper:

  Table 1   — per-dataset / aggregate summaries (Section 7.2)
  Table 2   — paired t-tests + Cohen's d (Section 7.3)
  Appendix A — Phase 1 / Phase 2 evaluation breakdown with theory bounds

Usage
-----
  python analysis.py               # all tables
  python analysis.py --table 1     # Table 1 only
  python analysis.py --table 2     # Table 2 only
  python analysis.py --table A     # Appendix A only

Output is formatted for direct transcription into the paper.
All numbers are printed as mean ± std with significance flags.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Constants (must match run_experiments.py)
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
METHODS = ["grid", "random", "bayesian", "two_phase"]
METHOD_LABELS = {
    "grid":      "Grid Search",
    "random":    "Random Search",
    "bayesian":  "Bayesian Opt.",
    "two_phase": "Two-Phase (Ours)",
}
DATASETS = ["covertype", "mnist", "adult"]
ALPHA = 0.05          # significance level for t-tests
K_MIN = 10            # must match run_experiments.py K_MIN; override with --k-min if needed

EFFECT_THRESHOLDS = [
    (2.0, "Huge"),
    (0.8, "Large"),
    (0.5, "Medium"),
    (0.2, "Small"),
    (0.0, "Negligible"),
]

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> Optional[dict]:
    """Load a results JSON file; return None with a warning if missing."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        print(f"  [WARNING] {path} not found — run the corresponding experiment first.")
        return None
    with open(path) as f:
        return json.load(f)


def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def _header(title: str) -> None:
    print()
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _subheader(title: str) -> None:
    print()
    _sep("─")
    print(f"  {title}")
    _sep("─")


def _col(*values, widths: List[int]) -> None:
    """Print a single table row with fixed column widths."""
    parts = [str(v).ljust(w) for v, w in zip(values, widths)]
    print("  " + "  ".join(parts))


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d = (μ_a − μ_b) / σ_pooled.

    Uses the pooled standard deviation (equal-n assumption valid here since
    both arrays come from the same set of seeds).
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_pooled = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    # Zero pooled variance means both arrays are constant (e.g. all methods
    # use exactly the same eval count on every seed). d is undefined, not
    # infinite — return nan so the effect label prints "N/A" cleanly.
    if var_pooled < 1e-12:
        return float("nan")
    sigma_pooled = math.sqrt(var_pooled)
    if math.isnan(sigma_pooled):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sigma_pooled)


def _effect_label(d: float) -> str:
    if math.isnan(d):
        return "N/A"
    abs_d = abs(d)
    for threshold, label in EFFECT_THRESHOLDS:
        if abs_d >= threshold:
            return label
    return "Negligible"


def _paired_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """
    Paired t-test: returns (t_statistic, p_value, cohen_d).

    Uses scipy.stats.ttest_rel (paired / repeated-measures).
    Both arrays must be the same length (same seeds).

    Edge cases:
    - If differences are all zero (identical arrays), t=0, p=1.0, d=nan.
    - If only one seed (n=1), returns nan throughout.
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan"), float("nan")
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0, 0.0
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        t_stat, p_val = stats.ttest_rel(a, b)
    if not math.isfinite(t_stat):
        t_stat = float("nan")
    d = _cohens_d(a, b)
    return float(t_stat), float(p_val), d


def _fmt_p(p: float) -> str:
    if math.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def _fmt_ms(mean: float, std: float, decimals: int = 1) -> str:
    fmt = f".{decimals}f"
    return f"{mean:{fmt}} ±{std:{fmt}}"


def _sig_star(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_per_seed(data: dict, dataset: str, method: str, field: str) -> np.ndarray:
    """Pull a scalar field from every run dict for a given dataset/method."""
    runs = data[dataset][method]["runs"]
    return np.array([run[field] for run in runs], dtype=float)


# ---------------------------------------------------------------------------
# Table 1 — Primary Results Summary
# ---------------------------------------------------------------------------


def print_table1(data: dict) -> None:
    _header("TABLE 1 — Primary Comparison Results  (Section 7.2)")

    W = [14, 18, 14, 10, 12, 10, 10]
    _col("Dataset", "Method", "k̂ (mean±std)", "CV acc", "Evals (mean±std)",
         "Speedup", "Success%", widths=W)
    _sep()

    all_tp_evals: List[float] = []
    all_tp_speedups: List[float] = []
    all_tp_success: List[float] = []

    for ds in DATASETS:
        if ds not in data:
            _col(ds, "— data missing —", *[""] * 5, widths=W)
            continue

        first_ds_row = True
        for method in METHODS:
            m = data[ds][method]
            k_mean = m["k_hat_mean"]
            k_std = m["k_hat_std"]
            cv_mean = m["cv_accuracy_mean"]
            ev_mean = m["evaluations_mean"]
            ev_std = m["evaluations_std"]
            speedup = m["speedup"]
            success = m["success_rate"] * 100

            ds_col = ds if first_ds_row else ""
            first_ds_row = False

            _col(
                ds_col, METHOD_LABELS[method],
                f"{k_mean:.0f} ±{k_std:.0f}",
                f"{cv_mean:.4f}",
                f"{ev_mean:.1f} ±{ev_std:.1f}",
                f"{speedup:.1f}×",
                f"{success:.0f}%",
                widths=W,
            )

            if method == "two_phase":
                all_tp_evals.append(ev_mean)
                all_tp_speedups.append(speedup)
                all_tp_success.append(m["success_rate"])

        _sep("-")

    _subheader("Aggregate (Two-Phase Search)")
    if all_tp_evals:
        _col("Aggregate", METHOD_LABELS["two_phase"],
             "—",
             "—",
             f"{np.mean(all_tp_evals):.1f} ±{np.std(all_tp_evals):.1f}",
             f"{np.mean(all_tp_speedups):.1f}×",
             f"{np.mean(all_tp_success)*100:.0f}%",
             widths=W)
    else:
        print("  No Two-Phase data available.")


# ---------------------------------------------------------------------------
# Table 2 — Paired t-tests + Cohen's d  (Section 7.3)
# ---------------------------------------------------------------------------


def print_table2(data: dict) -> None:
    _header("TABLE 2 — Statistical Significance  (Section 7.3)")
    print("  Paired t-tests (α=0.05, two-tailed) across seeds.")
    print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    print()

    METRICS = [
        ("evaluations",  "total_evaluations",  "Evaluation count"),
        ("wall_clock",   "wall_clock_seconds",  "Wall-clock time (s)"),
        ("k_error",      None,                  "k* error |k̂ − k_grid|"),
    ]

    COMPARISONS = [
        ("two_phase", "grid",     "Two-Phase vs Grid Search"),
        ("two_phase", "random",   "Two-Phase vs Random Search"),
        ("two_phase", "bayesian", "Two-Phase vs Bayesian Opt."),
    ]

    W = [28, 22, 9, 9, 9, 10, 12]

    for ds in DATASETS:
        _subheader(f"Dataset: {ds}")

        if ds not in data:
            print(f"  — data missing for {ds} —")
            continue

        _col("Comparison", "Metric", "t-stat", "p-value", "Sig.", "Cohen's d",
             "Effect size", widths=W)
        _sep("-")

        grid_ks = _extract_per_seed(data, ds, "grid", "k_hat")

        for method_a, method_b, comp_label in COMPARISONS:
            first_row = True
            for _metric_key, run_field, metric_label in METRICS:

                if run_field is not None:
                    vals_a = _extract_per_seed(data, ds, method_a, run_field)
                    vals_b = _extract_per_seed(data, ds, method_b, run_field)
                else:
                    ks_a = _extract_per_seed(data, ds, method_a, "k_hat")
                    ks_b = _extract_per_seed(data, ds, method_b, "k_hat")
                    vals_a = np.abs(ks_a - grid_ks)
                    vals_b = np.abs(ks_b - grid_ks)

                t, p, d = _paired_ttest(vals_a, vals_b)

                comp_col = comp_label if first_row else ""
                first_row = False

                _col(
                    comp_col, metric_label,
                    f"{t:+.2f}" if not math.isnan(t) else "N/A",
                    _fmt_p(p),
                    _sig_star(p),
                    f"{d:+.2f}" if not math.isnan(d) else "N/A",
                    _effect_label(d),
                    widths=W,
                )
            _sep("-")

    _subheader("Cross-dataset aggregate (all seeds pooled)")
    _col("Comparison", "Metric", "t-stat", "p-value", "Sig.", "Cohen's d",
         "Effect size", widths=W)
    _sep("-")

    available_ds = [ds for ds in DATASETS if ds in data]
    if not available_ds:
        print("  No data available.")
        return

    for method_a, method_b, comp_label in COMPARISONS:
        first_row = True
        for _metric_key, run_field, metric_label in METRICS:
            vals_a_all, vals_b_all = [], []
            for ds in available_ds:
                grid_ks = _extract_per_seed(data, ds, "grid", "k_hat")
                if run_field is not None:
                    vals_a_all.extend(_extract_per_seed(data, ds, method_a, run_field))
                    vals_b_all.extend(_extract_per_seed(data, ds, method_b, run_field))
                else:
                    ks_a = _extract_per_seed(data, ds, method_a, "k_hat")
                    ks_b = _extract_per_seed(data, ds, method_b, "k_hat")
                    vals_a_all.extend(np.abs(ks_a - grid_ks))
                    vals_b_all.extend(np.abs(ks_b - grid_ks))

            a, b = np.array(vals_a_all), np.array(vals_b_all)
            if len(a) >= 2 and len(b) >= 2:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", RuntimeWarning)
                    t_raw, p = stats.ttest_ind(a, b, equal_var=False)
                t = float(t_raw) if math.isfinite(t_raw) else float("nan")
                d = _cohens_d(a, b)
            else:
                t, p, d = float("nan"), float("nan"), float("nan")

            comp_col = comp_label if first_row else ""
            first_row = False
            _col(
                comp_col, metric_label,
                f"{t:+.2f}" if not math.isnan(t) else "N/A",
                _fmt_p(p),
                _sig_star(p),
                f"{d:+.2f}" if not math.isnan(d) else "N/A",
                _effect_label(d),
                widths=W,
            )
        _sep("-")


# ---------------------------------------------------------------------------
# Appendix A — Phase 1 / Phase 2 breakdown with theory bounds
# ---------------------------------------------------------------------------


def print_appendix_a(data: dict) -> None:
    _header("APPENDIX A — Phase-by-Phase Evaluation Breakdown")
    print("  Theory bounds (Theorem 1 + Theorem 3):")
    print("    Phase 1: ≤ ⌈log₂(k* / k_min)⌉ + 1")
    print("    Phase 2: ≤ ⌈log₂(U − L)⌉")
    print("    Total  : Phase1 + Phase2  (cf. Theorem 4)")
    print()

    W = [14, 10, 14, 14, 14, 11, 11, 10]
    _col("Dataset", "Seed", "k̂", "Phase1 evals",
         "Phase2 evals", "P1 bound", "P2 bound", "Within?", widths=W)
    _sep()

    summary_rows: List[dict] = []

    for ds in DATASETS:
        if ds not in data:
            _col(ds, "—", "data missing", *[""] * 5, widths=W)
            continue

        tp = data[ds]["two_phase"]
        runs = tp["runs"]

        p1_evals_all, p2_evals_all = [], []
        p1_within_all, p2_within_all = [], []
        total_within_all = []

        for run in runs:
            seed = run["seed"]
            k_hat = run["k_hat"]
            p1 = run["phase1_evaluations"]
            p2 = run["phase2_evaluations"]

            traj = run["trajectory"]
            phase1_steps = [s for s in traj if s["phase"] == 1]

            k_star_proxy = max(k_hat, K_MIN + 1)
            p1_bound = math.ceil(math.log2(k_star_proxy / K_MIN)) + 1

            if len(phase1_steps) >= 2:
                U = phase1_steps[-1]["k"]
                L_bracket = phase1_steps[-2]["k"]
            else:
                U = k_hat + 1
                L_bracket = k_hat
            bracket_width = max(U - L_bracket, 1)
            p2_bound = math.ceil(math.log2(bracket_width)) if bracket_width > 1 else 0

            p1_ok = p1 <= p1_bound
            p2_ok = p2 <= p2_bound
            both_ok = p1_ok and p2_ok

            _col(
                ds if run == runs[0] else "",
                seed, k_hat,
                f"{p1} ≤{p1_bound}  {'OK' if p1_ok else 'FAIL'}",
                f"{p2} ≤{p2_bound}  {'OK' if p2_ok else 'FAIL'}",
                p1_bound, p2_bound,
                "OK" if both_ok else "FAIL",
                widths=W,
            )

            p1_evals_all.append(p1)
            p2_evals_all.append(p2)
            p1_within_all.append(p1_ok)
            p2_within_all.append(p2_ok)
            total_within_all.append(both_ok)

        _sep("-")
        _col(
            f"{ds} MEAN", "—",
            f"{tp['k_hat_mean']:.0f} ±{tp['k_hat_std']:.0f}",
            f"{np.mean(p1_evals_all):.1f} ±{np.std(p1_evals_all):.1f}",
            f"{np.mean(p2_evals_all):.1f} ±{np.std(p2_evals_all):.1f}",
            f"{np.mean(p1_within_all)*100:.0f}%",
            f"{np.mean(p2_within_all)*100:.0f}%",
            f"{np.mean(total_within_all)*100:.0f}%",
            widths=W,
        )
        _sep()

        summary_rows.append({
            "dataset":           ds,
            "p1_mean":           np.mean(p1_evals_all),
            "p1_std":            np.std(p1_evals_all),
            "p2_mean":           np.mean(p2_evals_all),
            "p2_std":            np.std(p2_evals_all),
            "p1_compliance":     np.mean(p1_within_all),
            "p2_compliance":     np.mean(p2_within_all),
            "total_compliance":  np.mean(total_within_all),
        })

    if summary_rows:
        _subheader("Aggregate across all datasets")
        _col("Dataset", "", "k̂", "Phase1", "Phase2",
             "P1 ok%", "P2 ok%", "Both ok%", widths=W)
        _sep("-")
        for row in summary_rows:
            _col(
                row["dataset"], "", "—",
                _fmt_ms(row["p1_mean"], row["p1_std"]),
                _fmt_ms(row["p2_mean"], row["p2_std"]),
                f"{row['p1_compliance']*100:.0f}%",
                f"{row['p2_compliance']*100:.0f}%",
                f"{row['total_compliance']*100:.0f}%",
                widths=W,
            )
        _sep("-")
        _col(
            "OVERALL", "", "—",
            _fmt_ms(np.mean([r["p1_mean"] for r in summary_rows]),
                    np.std([r["p1_mean"] for r in summary_rows])),
            _fmt_ms(np.mean([r["p2_mean"] for r in summary_rows]),
                    np.std([r["p2_mean"] for r in summary_rows])),
            f"{np.mean([r['p1_compliance'] for r in summary_rows])*100:.0f}%",
            f"{np.mean([r['p2_compliance'] for r in summary_rows])*100:.0f}%",
            f"{np.mean([r['total_compliance'] for r in summary_rows])*100:.0f}%",
            widths=W,
        )

    _subheader("Reading these numbers")
    print("  P1 bound  = ⌈log₂(k̂ / k_min)⌉ + 1   (Theorem 1: Phase 1 terminates here)")
    print("  P2 bound  = ⌈log₂(U − L)⌉            (Theorem 3: binary search on bracket)")
    print("  'Within?' = both Phase 1 and Phase 2 actual counts ≤ their bounds")
    print("  A 100% compliance rate confirms the algorithm runs within its")
    print("  theoretical budget on every seed.")


# ---------------------------------------------------------------------------
# Results inventory
# ---------------------------------------------------------------------------


def print_data_inventory() -> None:
    _header("Results Inventory")
    experiments = [
        ("primary_comparison",    "Primary comparison (Tables 1, 2, Appendix A)"),
        ("monotonicity",          "Monotonicity validation (Figure 0)"),
        ("scaling",               "Scaling validation (Table 3, Figure 3)"),
        ("noise_robustness",      "Noise robustness (Table 4)"),
        ("bidirectional_validation", "Bidirectional validation (Table 5)"),
        ("edge_deployment",       "Edge deployment (Table 6)"),
        ("window_ablation",       "Window ablation (Table 7)"),
    ]
    for name, desc in experiments:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        status = "OK present" if os.path.exists(path) else "FAIL missing"
        size = f"  ({os.path.getsize(path):,} bytes)" if os.path.exists(path) else ""
        print(f"  {status}  {name}.json{size}")
        print(f"           {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all_tables() -> None:
    data = _load("primary_comparison")
    print_data_inventory()
    print_table1(data if data else {})
    print_table2(data if data else {})
    print_appendix_a(data if data else {})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and print paper statistics from results/ JSON files."
    )
    parser.add_argument(
        "--table", choices=["1", "2", "A", "all"], default="all",
        help="Which table to print (default: all)"
    )
    parser.add_argument(
        "--k-min", type=int, default=None,
        help="k_min value used in experiments (default: 10). Override if you ran "
             "experiments with a different k_min — affects Appendix A theory bounds."
    )
    args = parser.parse_args()

    if args.k_min is not None:
        K_MIN = args.k_min

    data = _load("primary_comparison")
    d = data if data else {}

    if args.table == "1":
        print_data_inventory()
        print_table1(d)
    elif args.table == "2":
        print_data_inventory()
        print_table2(d)
    elif args.table == "A":
        print_data_inventory()
        print_appendix_a(d)
    else:
        run_all_tables()
