"""
update_paper_values.py
----------------------
Reads empirical results from results/ and prints a complete editing guide
for updating the paper — every projected value side-by-side with the
empirical measurement, organised by table.

Usage
-----
  python update_paper_values.py          # full guide, all tables
  python update_paper_values.py --table 1   # Table 1 only
  python update_paper_values.py --table 2   # Table 2 only
  ... (tables: 1, 2, 3, 4, 5, 6, 7, A)

Output format
-------------
  OLD  (projected)  ->  NEW  (empirical)  [FLAG if headline-level change]

Exit code
---------
  0 = no headline changes
  1 = one or more headline numbers changed materially (see HEADLINE ALERTS)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Projected values exactly as they appear in the paper (v4)
# Edit this dict if the paper is revised before empirical runs complete.
# ---------------------------------------------------------------------------

PROJECTED = {
    # Table 1 — per-dataset × method
    "t1": {
        "covertype": {
            "grid":      {"k": 150,       "acc": 0.924, "evals": 100,      "time_s": 4500, "speedup": 1.0, "success_pct": 100},
            "random":    {"k": "143±15",  "acc": 0.919, "evals": 50,       "time_s": 2250, "speedup": 2.0, "success_pct": 74},
            "bayesian":  {"k": "148±7",   "acc": 0.922, "evals": 30,       "time_s": 1800, "speedup": 2.5, "success_pct": 87},
            "two_phase": {"k": "149±3",   "acc": 0.924, "evals": "13±1",   "time_s": 585,  "speedup": 7.7, "success_pct": 96},
        },
        "mnist": {
            "grid":      {"k": 100,       "acc": 0.968, "evals": 100,      "time_s": 3200, "speedup": 1.0, "success_pct": 100},
            "random":    {"k": "92±18",   "acc": 0.964, "evals": 50,       "time_s": 1600, "speedup": 2.0, "success_pct": 68},
            "bayesian":  {"k": "98±6",    "acc": 0.967, "evals": 30,       "time_s": 1200, "speedup": 2.7, "success_pct": 90},
            "two_phase": {"k": "99±2",    "acc": 0.968, "evals": "12±1",   "time_s": 384,  "speedup": 8.3, "success_pct": 98},
        },
        "adult": {
            "grid":      {"k": 80,        "acc": 0.862, "evals": 100,      "time_s": 1200, "speedup": 1.0, "success_pct": 100},
            "random":    {"k": "72±12",   "acc": 0.858, "evals": 50,       "time_s": 600,  "speedup": 2.0, "success_pct": 72},
            "bayesian":  {"k": "78±5",    "acc": 0.861, "evals": 30,       "time_s": 450,  "speedup": 2.7, "success_pct": 85},
            "two_phase": {"k": "79±2",    "acc": 0.862, "evals": "11±1",   "time_s": 132,  "speedup": 9.1, "success_pct": 97},
        },
        "aggregate": {
            "two_phase": {"evals": "12±1", "speedup": 8.4, "success_pct": 97},
        },
    },
    # Table 2 — per-comparison × metric
    "t2": {
        ("two_phase", "grid",     "evaluations"): {"p": "<0.001", "d": 8.52, "interp": "Huge effect"},
        ("two_phase", "grid",     "wall_clock"):  {"p": "<0.001", "d": 7.83, "interp": "Huge effect"},
        ("two_phase", "grid",     "k_error"):     {"p": "0.127",  "d": 0.42, "interp": "Not significant (good: accuracy preserved)"},
        ("two_phase", "random",   "evaluations"): {"p": "<0.001", "d": 4.26, "interp": "Large effect"},
        ("two_phase", "random",   "k_error"):     {"p": "<0.001", "d": 2.18, "interp": "Large effect (Two-Phase more precise)"},
        ("two_phase", "bayesian", "evaluations"): {"p": "<0.001", "d": 2.87, "interp": "Large effect"},
        ("two_phase", "bayesian", "k_error"):     {"p": "0.041",  "d": 0.73, "interp": "Medium effect (Two-Phase more precise)"},
    },
    # Table 3 — scaling
    "t3": {
        90:   {"predicted": "6.5+3.2->10", "actual": "9.8±1.2",  "matches": True},
        190:  {"predicted": "7.6+4.2->12", "actual": "11.5±1.4", "matches": True},
        490:  {"predicted": "8.9+5.6->15", "actual": "14.2±1.8", "matches": True},
        990:  {"predicted": "10.0+6.6->17","actual": "16.3±2.1", "matches": True},
        1990: {"predicted": "11.0+7.6->19","actual": "18.1±2.3", "matches": True},
    },
    # Table 4 — noise
    "t4": {
        0.000: {"k": "149.2±2.1", "success_pct": 98, "grid_ref": 150},
        0.010: {"k": "148.5±3.8", "success_pct": 94, "grid_ref": 150},
        0.020: {"k": "146.8±5.2", "success_pct": 88, "grid_ref": 150},
        0.050: {"k": "143.1±8.7", "success_pct": 76, "grid_ref": 150},
    },
    # Table 5 — bidirectional
    "t5": {
        "covertype": {"k_fwd": 149, "k_rev": 151, "abs_diff": 2,  "rel_diff_pct": 1.3, "converged": True},
        "mnist":     {"k_fwd": 99,  "k_rev": 102, "abs_diff": 3,  "rel_diff_pct": 2.9, "converged": True},
        "adult":     {"k_fwd": 79,  "k_rev": 81,  "abs_diff": 2,  "rel_diff_pct": 2.5, "converged": True},
    },
    # Table 6 — edge deployment
    "t6": {
        "covertype": {
            "grid":      {"k": 150, "inf_ms": 0.31,  "mem_mb": 12.4, "fits_4mb": False},
            "two_phase": {"k": 13,  "inf_ms": 0.027, "mem_mb": 1.1,  "fits_4mb": True},
            "default":   {"k": 500, "inf_ms": 1.04,  "mem_mb": 41.3, "fits_4mb": False},
        },
        "mnist": {
            "grid":      {"k": 100, "inf_ms": 0.21,  "mem_mb": 68.2,  "fits_4mb": False},
            "two_phase": {"k": 12,  "inf_ms": 0.025, "mem_mb": 8.2,   "fits_4mb": False},
            "default":   {"k": 500, "inf_ms": 1.05,  "mem_mb": 341.0, "fits_4mb": False},
        },
        "adult": {
            "grid":      {"k": 80,  "inf_ms": 0.16,  "mem_mb": 0.8, "fits_4mb": True},
            "two_phase": {"k": 11,  "inf_ms": 0.022, "mem_mb": 0.1, "fits_4mb": True},
            "default":   {"k": 500, "inf_ms": 1.01,  "mem_mb": 5.0, "fits_4mb": False},
        },
    },
    # Table 7 — window ablation
    "t7": {
        3:  {"success_pct": 88, "fp_pct": 9.0, "evals": "11.2±1.8"},
        5:  {"success_pct": 96, "fp_pct": 3.0, "evals": "13.1±1.4"},
        10: {"success_pct": 98, "fp_pct": 0.5, "evals": "15.8±1.2"},
    },
    # Appendix A — Phase 1 / Phase 2 breakdown
    "appA": {
        "covertype": {"p1_theory": "≤10", "p1_actual": "9.2±1.1", "p2_theory": "≤7", "p2_actual": "6.8±0.9"},
        "mnist":     {"p1_theory": "≤10", "p1_actual": "8.9±1.2", "p2_theory": "≤7", "p2_actual": "6.5±0.8"},
        "adult":     {"p1_theory": "≤10", "p1_actual": "8.4±1.0", "p2_theory": "≤7", "p2_actual": "6.2±0.7"},
    },
    # Headline numbers (abstract / Section 2.5 / Section 9)
    "headlines": {
        "speedup_range":     "7–15×",
        "aggregate_speedup": 8.4,
        "success_rate":      0.97,
        "covertype_speedup": 7.6,
        "covertype_evals":   13,
    },
}

# Thresholds for flagging material headline changes
HEADLINE_SPEEDUP_TOLERANCE = 0.5    # flag if aggregate speedup changes by more than ±0.5×
HEADLINE_SUCCESS_TOLERANCE = 0.03   # flag if success rate shifts by more than ±3 pp
BUDGET_CHANGE_FLAG         = True   # always flag 4 MB compliance changes

RESULTS_DIR   = "results"
DATASETS      = ["covertype", "mnist", "adult"]
METHODS       = ["grid", "random", "bayesian", "two_phase"]
METHOD_LABELS = {
    "grid": "Grid Search", "random": "Random Search",
    "bayesian": "Bayesian Opt.", "two_phase": "Two-Phase (Ours)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> Optional[dict]:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _ms(mean, std=None, decimals=1) -> str:
    fmt = f".{decimals}f"
    if std is None or std == 0:
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} ±{std:{fmt}}"


def _pct(v: float) -> str:
    return f"{v*100:.0f}%"


def _sep(c="─", w=74): print(c * w)
def _header(t): print(); _sep("═"); print(f"  {t}"); _sep("═")
def _sub(t):    print(); _sep("─"); print(f"  {t}"); _sep("─")


def _row(label: str, old: str, new: str, flag: str = ""):
    flag_str = f"  ★ {flag}" if flag else ""
    print(f"  {label:<38}  OLD: {old:<22}  NEW: {new:<22}{flag_str}")


def _missing(table_name: str):
    print(f"  [WAITING] {table_name}.json not found in results/.")
    print(f"  Run the corresponding experiment first, then re-run this script.")


headline_alerts: list[str] = []


def _alert(msg: str):
    headline_alerts.append(msg)
    return "HEADLINE CHANGE"


# ---------------------------------------------------------------------------
# Table 1 — Primary Comparison
# ---------------------------------------------------------------------------


def print_table1(data: dict) -> None:
    _header("TABLE 1 — Primary Comparison  (Section 7.2)")

    for ds in DATASETS:
        _sub(f"Dataset: {ds}")
        if ds not in data:
            _missing("primary_comparison"); continue

        for method in METHODS:
            proj = PROJECTED["t1"][ds][method]
            emp  = data[ds][method]
            label_prefix = METHOD_LABELS[method]

            k_new   = _ms(emp["k_hat_mean"],        emp["k_hat_std"],        decimals=0)
            acc_new = _ms(emp["cv_accuracy_mean"],   emp["cv_accuracy_std"],  decimals=4)
            ev_new  = _ms(emp["evaluations_mean"],   emp["evaluations_std"],  decimals=1)
            t_new   = _ms(emp["wall_clock_mean"],    emp["wall_clock_std"],   decimals=1)
            spd_new = f"{emp['speedup']:.1f}×"
            suc_new = _pct(emp["success_rate"])

            _row(f"{label_prefix} — k̂ found",   str(proj["k"]),             k_new)
            _row(f"{label_prefix} — CV accuracy", str(proj["acc"]),          acc_new)
            _row(f"{label_prefix} — evaluations", str(proj["evals"]),        ev_new)
            _row(f"{label_prefix} — time (s)",    str(proj["time_s"]),       t_new)
            _row(f"{label_prefix} — speedup",     f"{proj['speedup']:.1f}×", spd_new)
            _row(f"{label_prefix} — success %",   f"{proj['success_pct']}%", suc_new)

        print()

    _sub("Aggregate — Two-Phase Search")
    proj_agg   = PROJECTED["t1"]["aggregate"]["two_phase"]
    tp_evals   = [data[ds]["two_phase"]["evaluations_mean"] for ds in DATASETS if ds in data]
    tp_speedups= [data[ds]["two_phase"]["speedup"]          for ds in DATASETS if ds in data]
    tp_success = [data[ds]["two_phase"]["success_rate"]     for ds in DATASETS if ds in data]

    if tp_evals:
        import numpy as np
        agg_ev  = np.mean(tp_evals);    agg_ev_std  = np.std(tp_evals)
        agg_spd = np.mean(tp_speedups); agg_suc     = np.mean(tp_success)

        flag_spd = ""
        if abs(agg_spd - proj_agg["speedup"]) > HEADLINE_SPEEDUP_TOLERANCE:
            flag_spd = _alert(
                f"Table 1 aggregate speedup: {proj_agg['speedup']:.1f}× -> {agg_spd:.1f}× "
                f"(Δ={agg_spd - proj_agg['speedup']:+.1f}×). "
                "Update abstract, Section 2.3, Section 2.5, and Section 9."
            )

        flag_suc = ""
        if abs(agg_suc - proj_agg["success_pct"] / 100) > HEADLINE_SUCCESS_TOLERANCE:
            flag_suc = _alert(
                f"Table 1 aggregate success rate: {proj_agg['success_pct']}% -> {agg_suc*100:.0f}% "
                f"(Δ={agg_suc*100 - proj_agg['success_pct']:+.0f} pp). "
                "Update abstract and Section 2.5."
            )

        _row("Two-Phase — evaluations", str(proj_agg["evals"]),      _ms(agg_ev, agg_ev_std, 1))
        _row("Two-Phase — speedup",     f"{proj_agg['speedup']}×",   f"{agg_spd:.1f}×", flag_spd)
        _row("Two-Phase — success %",   f"{proj_agg['success_pct']}%", _pct(agg_suc), flag_suc)
    else:
        _missing("primary_comparison")


# ---------------------------------------------------------------------------
# Table 2 — Statistical Significance
# ---------------------------------------------------------------------------


def print_table2(data: dict) -> None:
    _header("TABLE 2 — Statistical Significance  (Section 7.3)")
    print("  Note: Table 2 values are derived from analysis.py, not stored in")
    print("  primary_comparison.json directly. Run `python analysis.py --table 2` for")
    print("  the full per-dataset breakdown. This section shows cross-dataset aggregates.")
    print()

    if not data:
        _missing("primary_comparison"); return

    import numpy as np
    from scipy import stats as _stats

    METRICS_MAP = [
        ("evaluations", "total_evaluations",  "Evaluations"),
        ("wall_clock",  "wall_clock_seconds",  "Wall-clock (s)"),
        ("k_error",     None,                  "k* error |k̂−k_grid|"),
    ]
    COMPARISONS = [
        ("two_phase", "grid",     "Two-Phase vs Grid Search"),
        ("two_phase", "random",   "Two-Phase vs Random Search"),
        ("two_phase", "bayesian", "Two-Phase vs Bayesian Opt."),
    ]

    available_ds = [ds for ds in DATASETS if ds in data]

    for method_a, method_b, comp_label in COMPARISONS:
        _sub(comp_label)
        for metric_key, run_field, metric_label in METRICS_MAP:
            proj_key = (method_a, method_b, metric_key)
            proj = PROJECTED["t2"].get(proj_key)
            if proj is None:
                continue

            vals_a, vals_b = [], []
            for ds in available_ds:
                grid_ks = np.array([r["k_hat"] for r in data[ds]["grid"]["runs"]])
                if run_field:
                    vals_a.extend([r[run_field] for r in data[ds][method_a]["runs"]])
                    vals_b.extend([r[run_field] for r in data[ds][method_b]["runs"]])
                else:
                    ks_a = np.array([r["k_hat"] for r in data[ds][method_a]["runs"]])
                    ks_b = np.array([r["k_hat"] for r in data[ds][method_b]["runs"]])
                    vals_a.extend(abs(ks_a - grid_ks))
                    vals_b.extend(abs(ks_b - grid_ks))

            a, b = np.array(vals_a, float), np.array(vals_b, float)
            if len(a) >= 2 and len(b) >= 2:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = _stats.ttest_ind(a, b, equal_var=False)
                var_pool = (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
                d = (np.mean(a) - np.mean(b)) / math.sqrt(var_pool) if var_pool > 1e-12 else float("nan")
                p_str  = "<0.001" if p < 0.001 else f"{p:.3f}"
                d_str  = f"{d:.2f}" if math.isfinite(d) else "N/A"
                new_str = f"p={p_str}, d={d_str}"
            else:
                new_str = "insufficient data"

            old_str = f"p={proj['p']}, d={proj['d']:.2f}"
            _row(metric_label, old_str, new_str)


# ---------------------------------------------------------------------------
# Table 3 — Scaling Validation
# ---------------------------------------------------------------------------


def print_table3() -> None:
    _header("TABLE 3 — Scaling Validation  (Section 7.4)")
    data = _load("scaling")
    if data is None:
        _missing("scaling"); return

    for n_val, proj in PROJECTED["t3"].items():
        k_max = n_val + 10
        key = str(k_max)
        if key not in data:
            print(f"  N={n_val} (k_max={k_max}): not found in scaling.json")
            continue
        emp = data[key]
        ev_new      = _ms(emp["evaluations_mean"], emp["evaluations_std"], 1)
        matches_new = "OK Yes" if emp["within_bound"] else "FAIL No"
        _row(f"N={n_val} — actual evals",   proj["actual"],         ev_new)
        _row(f"N={n_val} — within theory?", str(proj["matches"]),   matches_new)


# ---------------------------------------------------------------------------
# Table 4 — Noise Robustness
# ---------------------------------------------------------------------------


def print_table4() -> None:
    _header("TABLE 4 — Noise Robustness  (Section 7.5)")
    data = _load("noise_robustness")
    if data is None:
        _missing("noise_robustness"); return

    for sigma, proj in PROJECTED["t4"].items():
        key = str(float(sigma))
        if key not in data:
            key = next((k for k in data if abs(float(k) - sigma) < 1e-6), None)
        if key is None:
            print(f"  σ={sigma}: not found in noise_robustness.json")
            continue
        emp    = data[key]
        k_new  = _ms(emp["k_hat_mean"], emp["k_hat_std"], 1)
        suc_new = _pct(emp["success_rate"])
        _row(f"σ={sigma:.3f} — k̂",        str(proj["k"]),            k_new)
        _row(f"σ={sigma:.3f} — success %", f"{proj['success_pct']}%", suc_new)


# ---------------------------------------------------------------------------
# Table 5 — Bidirectional Validation
# ---------------------------------------------------------------------------


def print_table5() -> None:
    _header("TABLE 5 — Bidirectional Validation  (Section 7.6)")
    data = _load("bidirectional_validation")
    if data is None:
        _missing("bidirectional_validation"); return

    import numpy as np
    for ds, proj in PROJECTED["t5"].items():
        if ds not in data:
            print(f"  {ds}: not found in bidirectional_validation.json"); continue
        emp       = data[ds]
        runs      = emp["runs"]
        fwd_ks    = [r["k_forward"]  for r in runs]
        rev_ks    = [r["k_reverse"]  for r in runs]
        rel_diffs = [r["k_rel_diff"] for r in runs]

        fwd_new  = _ms(np.mean(fwd_ks),    np.std(fwd_ks),    0)
        rev_new  = _ms(np.mean(rev_ks),    np.std(rev_ks),    0)
        rel_new  = f"{np.mean(rel_diffs)*100:.1f}%"
        conv_new = "OK Yes" if np.mean(rel_diffs) < 0.10 else "FAIL No (>10%)"

        _row(f"{ds} — k_forward",  str(proj["k_fwd"]),             fwd_new)
        _row(f"{ds} — k_reverse",  str(proj["k_rev"]),             rev_new)
        _row(f"{ds} — rel. diff",  f"{proj['rel_diff_pct']:.1f}%", rel_new)
        _row(f"{ds} — converged?", "OK Yes",                        conv_new)
        print()


# ---------------------------------------------------------------------------
# Table 6 — Edge Deployment
# ---------------------------------------------------------------------------


def print_table6() -> None:
    _header("TABLE 6 — Edge Deployment  (Section 7.7)")
    data = _load("edge_deployment")
    if data is None:
        _missing("edge_deployment"); return

    config_labels = {
        "grid":      "Grid Search k*",
        "two_phase": "Two-Phase k̂",
        "default":   "Default 500",
    }

    for ds, proj_ds in PROJECTED["t6"].items():
        if ds not in data:
            print(f"  {ds}: not found in edge_deployment.json"); continue
        _sub(f"Dataset: {ds}")

        for config, proj in proj_ds.items():
            agg_key = f"agg_{config}"
            if agg_key not in data[ds]:
                print(f"  {config}: aggregate key '{agg_key}' not found"); continue
            emp = data[ds][agg_key]
            lbl = config_labels[config]

            mem_new = _ms(emp["size_mb_mean"], emp["size_mb_std"], 2)
            inf_new = f"{emp['inference_ms_mean']:.3f}"
            bud_emp = emp["budget_compliance_rate"] == 1.0
            bud_new = "Yes OK" if bud_emp else "No"

            flag_bud = ""
            if bud_emp != proj["fits_4mb"]:
                direction = "GAINED" if bud_emp else "LOST"
                flag_bud = _alert(
                    f"Table 6 {ds}/{config}: 4MB budget compliance {direction} "
                    f"(was {'Yes' if proj['fits_4mb'] else 'No'}, now {bud_new}). "
                    "Update Section 7.7 narrative and abstract edge-deployment claim."
                )

            _row(f"{lbl} — memory (MB)",    f"{proj['mem_mb']:.1f}",  mem_new)
            _row(f"{lbl} — inference (ms)", f"{proj['inf_ms']:.3f}",  inf_new)
            _row(f"{lbl} — fits 4MB?",      "Yes OK" if proj["fits_4mb"] else "No", bud_new, flag_bud)
        print()


# ---------------------------------------------------------------------------
# Table 7 — Window Size Ablation
# ---------------------------------------------------------------------------


def print_table7() -> None:
    _header("TABLE 7 — Window Size Ablation  (Section 7.8)")
    data = _load("window_ablation")
    if data is None:
        _missing("window_ablation"); return

    for w, proj in PROJECTED["t7"].items():
        key = str(w)
        if key not in data:
            print(f"  w={w}: not found in window_ablation.json"); continue
        emp    = data[key]
        suc_new = _pct(emp["success_rate"])
        fp_new  = _pct(emp["false_positive_rate"])
        ev_new  = _ms(emp["evaluations_mean"], emp["evaluations_std"], 1)

        _row(f"w={w} — success %",      f"{proj['success_pct']}%", suc_new)
        _row(f"w={w} — false pos rate", f"{proj['fp_pct']:.1f}%",  fp_new)
        _row(f"w={w} — mean evals",     str(proj["evals"]),         ev_new)
        print()


# ---------------------------------------------------------------------------
# Appendix A — Phase 1 / Phase 2 Breakdown
# ---------------------------------------------------------------------------


def print_appendix_a(data: dict) -> None:
    _header("APPENDIX A — Phase 1 / Phase 2 Evaluation Breakdown")

    if not data:
        _missing("primary_comparison"); return

    import numpy as np
    for ds, proj in PROJECTED["appA"].items():
        if ds not in data:
            print(f"  {ds}: not found"); continue
        _sub(f"Dataset: {ds}")

        runs  = data[ds]["two_phase"]["runs"]
        p1_vals = [r["phase1_evaluations"] for r in runs]
        p2_vals = [r["phase2_evaluations"] for r in runs]

        p1_new = _ms(np.mean(p1_vals), np.std(p1_vals), 1)
        p2_new = _ms(np.mean(p2_vals), np.std(p2_vals), 1)

        _row(f"{ds} — Phase 1 actual", proj["p1_actual"], p1_new)
        _row(f"{ds} — Phase 2 actual", proj["p2_actual"], p2_new)


# ---------------------------------------------------------------------------
# Headline locations guide (always printed)
# ---------------------------------------------------------------------------


def print_location_guide() -> None:
    _header("PAPER LOCATION GUIDE — Where to make each change")
    print("""
  After updating each table value in Section 7, check these prose locations:

  ABSTRACT (Section 1)
  ─────────────────────
  - "7–15× speedup over grid search"
      -> Update range to match actual min/max speedup across datasets (Table 1).
  - "within 0.01 accuracy of the grid search optimum in over 95% of runs"
      -> Update threshold/percentage to match aggregate success rate (Table 1 agg.).
  - "predicts a 7–15× speedup" -> change "predicts" to measured past-tense verb.

  SECTION 2.3 (AutoML pipelines bullet)
  ──────────────────────────────────────
  - "The projected 7.6× speedup on Covertype"
      -> Replace 7.6× with actual Covertype speedup from Table 1.
  - "Two-Phase Search requires approximately 13"
      -> Replace 13 with actual mean evaluation count for Covertype.

  SECTION 2.5 (Two Audiences)
  ────────────────────────────
  - Any reference to specific speedup multiples or success percentages.

  SECTION 7 INTRO
  ────────────────
  - "Hardware specifications will be reported in the final submission."
      -> Replace with actual CPU model and RAM.
  - Remove "Note on numerical values" disclaimer block (top of Section 7).
  - Remove all per-table "(projected values, pending empirical validation)" captions.

  SECTION 7.7 (Edge Deployment narrative)
  ────────────────────────────────────────
  - "Two-Phase k-hat fits 4 MB flash budget on 2/3 datasets; default 500 trees fails all three"
      -> Update 2/3 fraction based on Table 6 budget compliance.

  SECTION 9 (Conclusion)
  ───────────────────────
  - Any "predicts" / "projected" language -> past-tense empirical claims.
  - Update aggregate speedup and success rate numbers.

  APPENDIX B (Revision checklist)
  ─────────────────────────────────
  - Mark empirical items as resolved once experiments are complete.
    """)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all() -> None:
    primary = _load("primary_comparison")
    d = primary if primary else {}

    print_table1(d)
    print_table2(d)
    print_table3()
    print_table4()
    print_table5()
    print_table6()
    print_table7()
    print_appendix_a(d)
    print_location_guide()

    _header("HEADLINE ALERTS  (values requiring abstract/conclusion updates)")
    if headline_alerts:
        for i, alert in enumerate(headline_alerts, 1):
            print(f"\n  [{i}] {alert}")
        print()
        sys.exit(1)
    else:
        print("\n  No headline-level changes detected.")
        print("  (Run again after all experiments complete for a final check.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print old-vs-new replacement guide for all paper tables."
    )
    parser.add_argument(
        "--table", choices=["1", "2", "3", "4", "5", "6", "7", "A", "all"],
        default="all",
        help="Which table to print (default: all)"
    )
    args = parser.parse_args()

    primary = _load("primary_comparison")
    d = primary if primary else {}

    dispatch = {
        "1":   lambda: print_table1(d),
        "2":   lambda: print_table2(d),
        "3":   print_table3,
        "4":   print_table4,
        "5":   print_table5,
        "6":   print_table6,
        "7":   print_table7,
        "A":   lambda: print_appendix_a(d),
        "all": run_all,
    }
    dispatch[args.table]()

    if args.table != "all" and headline_alerts:
        _header("HEADLINE ALERTS")
        for i, alert in enumerate(headline_alerts, 1):
            print(f"\n  [{i}] {alert}")
        sys.exit(1)
