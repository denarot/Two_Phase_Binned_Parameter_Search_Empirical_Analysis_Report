"""
generate_figures.py
-------------------
Generates four publication-quality figures for:
  "Two-Phase Search for Optimal Random Forest Tree Count:
   Seven Theorems Proving O(log N) Optimality"

Figures produced
----------------
  Figure 0  — Monotonicity Validation          (Section 7.1)
  Figure 1  — Convergence Trajectory           (Section 7.2)
  Figure 2  — Pareto Frontier                  (Section 7.3)
  Figure 3  — Log Scaling                      (Section 7.4)

Output
------
  figures/figure0_monotonicity.{pdf,png}
  figures/figure1_convergence.{pdf,png}
  figures/figure2_pareto.{pdf,png}
  figures/figure3_logscaling.{pdf,png}

Usage
-----
  python generate_figures.py          # all four figures
  python generate_figures.py --fig 0  # Figure 0 only
  python generate_figures.py --fig 1  # Figure 1 only
  python generate_figures.py --fig 2  # Figure 2 only
  python generate_figures.py --fig 3  # Figure 3 only

Missing data
------------
  If a results file is absent, the corresponding figure is saved as a
  placeholder with a clear "data not yet available" annotation so the
  pipeline stays runnable end-to-end before experiments complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Global style  (publication quality: ~3.5-inch column width, 10/8pt fonts)
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures"
RESULTS_DIR = "results"

# Colour palette — consistent across all four figures
C = {
    "grid":      "#888888",   # grey
    "random":    "#2166ac",   # blue
    "bayesian":  "#d6604d",   # orange-red
    "two_phase": "#1a9641",   # green
    "phase1":    "#4393c3",   # sky blue  (trajectory phase 1)
    "phase2":    "#d73027",   # red       (trajectory phase 2)
    "theory":    "#000000",   # black     (theoretical bound line)
    "ref":       "#aaaaaa",   # light grey (grid search reference line)
}

METHOD_LABELS = {
    "grid":      "Grid Search",
    "random":    "Random Search",
    "bayesian":  "Bayesian Opt.",
    "two_phase": "Two-Phase (Ours)",
}

MARKER_STYLES = {
    "grid":      ("s", C["grid"]),
    "random":    ("^", C["random"]),
    "bayesian":  ("D", C["bayesian"]),
    "two_phase": ("o", C["two_phase"]),
}

DATASET_LABELS = {
    "covertype": "Covertype",
    "mnist":     "MNIST",
    "adult":     "Adult",
}


def _apply_style() -> None:
    """Apply publication rcParams once before any figure is created."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          8,
        "axes.labelsize":     10,
        "axes.titlesize":     10,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.frameon":     False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "lines.linewidth":    1.4,
        "lines.markersize":   5,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,   # embed fonts as TrueType (not Type 3)
        "ps.fonttype":        42,
    })


def _col_width_fig(n_panels: int, height: float = 2.4) -> Tuple[float, float]:
    """Return (width, height) in inches for n_panels side-by-side subplots."""
    return (3.5 * n_panels, height)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load(name: str) -> Optional[Dict]:
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save_fig(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved → {path}")


def _placeholder(name: str, message: str, n_panels: int = 3) -> None:
    """Save a placeholder figure when data is not yet available."""
    fig, axes = plt.subplots(1, n_panels, figsize=_col_width_fig(n_panels))
    if n_panels == 1:
        axes = [axes]
    for ax in axes:
        ax.text(0.5, 0.5, message, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#999999",
                wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    _save_fig(fig, name)
    plt.close(fig)
    print(f"  (Placeholder — {message})")


# ---------------------------------------------------------------------------
# Plateau detection helper (mirrors Assumption 2)
# ---------------------------------------------------------------------------


def _find_plateau_onset(k_values: List[float], mean_curve: List[float],
                        epsilon: float = 0.005, w: int = 3) -> Optional[float]:
    """
    Return the k at which the windowed gradient first drops below epsilon.
    Used to place the vertical dashed line in Figure 0.
    """
    for i in range(w, len(mean_curve)):
        window = [mean_curve[i - j] - mean_curve[i - j - 1] for j in range(w)]
        g = sum(window) / w
        if g < epsilon:
            return float(k_values[i])
    return float(k_values[-1])


# ---------------------------------------------------------------------------
# Figure 0 — Monotonicity Validation
# ---------------------------------------------------------------------------


def figure0_monotonicity() -> None:
    """
    Three subplots (one per dataset).

    - Mean accuracy ± 1 std band across all seeds
    - Log-scaled x-axis (tree count k)
    - Vertical dashed line at empirical plateau onset k_p
    """
    data = _load("monotonicity")
    if data is None:
        _placeholder("figure0_monotonicity",
                     "Awaiting monotonicity.json\n"
                     "(run: python run_experiments.py monotonicity)")
        return

    datasets = [ds for ds in ["covertype", "mnist", "adult"] if ds in data]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=_col_width_fig(n), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        r = data[ds_name]
        ks   = np.array(r["k_values"],    dtype=float)
        mean = np.array(r["mean_curve"],  dtype=float)
        std  = np.array(r["std_curve"],   dtype=float)

        ax.plot(ks, mean, color=C["two_phase"], linewidth=1.6, zorder=3)
        ax.fill_between(ks, mean - std, mean + std,
                        color=C["two_phase"], alpha=0.18, zorder=2)

        k_p = _find_plateau_onset(ks.tolist(), mean.tolist())
        ax.axvline(k_p, color="#555555", linestyle="--", linewidth=1.0, zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel("Tree count $k$", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("CV Accuracy", fontsize=10)
        ax.set_title(DATASET_LABELS.get(ds_name, ds_name), fontsize=10, pad=4)

        y_range = mean.max() - mean.min() if mean.max() > mean.min() else 0.01
        ax.annotate(f"$k_p={k_p:.0f}$",
                    xy=(k_p, mean.min() + y_range * 0.15),
                    xytext=(k_p * 1.25, mean.min() + y_range * 0.05),
                    fontsize=7, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))

        ylo = max(0.0, mean.min() - std.max() - 0.02)
        yhi = min(1.0, mean.max() + std.max() + 0.02)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(int(k)) for k in ks],
                           fontsize=7, rotation=45, ha="right")

    fig.suptitle("Figure 0: Monotonicity Validation", fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figure0_monotonicity")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Convergence Trajectory
# ---------------------------------------------------------------------------


def figure1_convergence() -> None:
    """
    Three subplots (one per dataset), showing the seed-42 Two-Phase trajectory.

    - Phase 1 points in sky blue, Phase 2 in red
    - Vertical dashed line at Phase 1 → Phase 2 transition
    - Horizontal dashed line at grid search optimum accuracy (seed 42)
    - Connected evaluation path in light grey
    """
    data = _load("primary_comparison")
    if data is None:
        _placeholder("figure1_convergence",
                     "Awaiting primary_comparison.json\n"
                     "(run: python run_experiments.py primary)")
        return

    datasets = [ds for ds in ["covertype", "mnist", "adult"] if ds in data]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=_col_width_fig(n), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds = data[ds_name]

        tp_run = next(
            (r for r in ds["two_phase"]["runs"] if r["seed"] == 42),
            ds["two_phase"]["runs"][0],
        )
        traj = tp_run["trajectory"]

        steps  = [t["step"]     for t in traj]
        accs   = [t["accuracy"] for t in traj]
        phases = [t["phase"]    for t in traj]

        grid_run = next(
            (r for r in ds["grid"]["runs"] if r["seed"] == 42),
            ds["grid"]["runs"][0],
        )
        grid_opt_acc = grid_run["cv_accuracy"]

        p1_steps = [s for s, ph in zip(steps, phases) if ph == 1]
        p1_accs  = [a for a, ph in zip(accs,  phases) if ph == 1]
        p2_steps = [s for s, ph in zip(steps, phases) if ph == 2]
        p2_accs  = [a for a, ph in zip(accs,  phases) if ph == 2]

        ax.plot(steps, accs, color="#cccccc", linewidth=0.8, zorder=1)
        ax.scatter(p1_steps, p1_accs, color=C["phase1"], zorder=4,
                   s=28, label="Phase 1", marker="o")
        ax.scatter(p2_steps, p2_accs, color=C["phase2"], zorder=4,
                   s=28, label="Phase 2", marker="s")

        if p1_steps and p2_steps:
            transition_x = p1_steps[-1] + 0.5
            ax.axvline(transition_x, color="#555555", linestyle="--",
                       linewidth=0.9, zorder=2)
            ax.text(transition_x + 0.05,
                    min(accs) + (max(accs) - min(accs)) * 0.05,
                    "P1→P2", fontsize=6.5, color="#555555", va="bottom")

        ax.axhline(grid_opt_acc, color=C["ref"], linestyle=":", linewidth=1.0,
                   zorder=1, label=f"Grid opt. ({grid_opt_acc:.3f})")

        ax.set_xlabel("Evaluation step", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("CV Accuracy", fontsize=10)
        ax.set_title(DATASET_LABELS.get(ds_name, ds_name), fontsize=10, pad=4)
        ax.set_xlim(0.5, len(traj) + 0.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ylo = min(accs) - 0.02
        yhi = max(max(accs), grid_opt_acc) + 0.02
        ax.set_ylim(max(0.0, ylo), min(1.0, yhi))

        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle("Figure 1: Convergence Trajectory (seed 42)", fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figure1_convergence")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Pareto Frontier
# ---------------------------------------------------------------------------


def _pareto_frontier(times: np.ndarray, accs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (times, accs) of non-dominated points.

    A point (t, a) is dominated if there exists (t', a') with t'≤t and a'≥a
    (strictly better on at least one axis).
    """
    points = sorted(zip(times, accs), key=lambda p: p[0])
    frontier_t, frontier_a = [], []
    max_acc_so_far = -np.inf
    for t, a in points:
        if a >= max_acc_so_far:
            frontier_t.append(t)
            frontier_a.append(a)
            max_acc_so_far = a
    return np.array(frontier_t), np.array(frontier_a)


def figure2_pareto() -> None:
    """
    Three subplots (one per dataset).

    - Scatter all 10 runs per method as (wall_clock_seconds, cv_accuracy)
    - Pareto frontier through non-dominated points across all methods
    """
    data = _load("primary_comparison")
    if data is None:
        _placeholder("figure2_pareto",
                     "Awaiting primary_comparison.json\n"
                     "(run: python run_experiments.py primary)")
        return

    datasets = [ds for ds in ["covertype", "mnist", "adult"] if ds in data]
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=_col_width_fig(n), sharey=False)
    if n == 1:
        axes = [axes]

    methods = ["grid", "random", "bayesian", "two_phase"]

    legend_handles = [
        mlines.Line2D([], [], marker=MARKER_STYLES[m][0], color=MARKER_STYLES[m][1],
                      linestyle="None", markersize=5, label=METHOD_LABELS[m])
        for m in methods
    ]
    legend_handles.append(
        mlines.Line2D([], [], color="#333333", linestyle="-",
                      linewidth=1.0, label="Pareto frontier")
    )

    for ax, ds_name in zip(axes, datasets):
        ds = data[ds_name]
        all_times, all_accs = [], []

        for method in methods:
            marker, color = MARKER_STYLES[method]
            runs = ds[method]["runs"]
            times = np.array([r["wall_clock_seconds"] for r in runs])
            accs  = np.array([r["cv_accuracy"]        for r in runs])

            ax.scatter(times, accs, marker=marker, color=color,
                       s=22, zorder=4, alpha=0.85, edgecolors="none")

            all_times.extend(times.tolist())
            all_accs.extend(accs.tolist())

        ft, fa = _pareto_frontier(np.array(all_times), np.array(all_accs))
        if len(ft) > 1:
            ax.plot(ft, fa, color="#333333", linewidth=1.0, zorder=3,
                    linestyle="-")
        elif len(ft) == 1:
            ax.scatter(ft, fa, color="#333333", s=30, zorder=3)

        ax.set_xscale("log")
        ax.set_xlabel("Wall-clock time (s)", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("CV Accuracy", fontsize=10)
        ax.set_title(DATASET_LABELS.get(ds_name, ds_name), fontsize=10, pad=4)

        t_arr = np.array(all_times)
        a_arr = np.array(all_accs)
        ax.set_xlim(t_arr.min() * 0.7, t_arr.max() * 1.4)
        ax.set_ylim(max(0.0, a_arr.min() - 0.03), min(1.0, a_arr.max() + 0.03))

    axes[0].legend(handles=legend_handles, loc="lower right",
                   fontsize=7, markerscale=1.1)

    fig.suptitle("Figure 2: Pareto Frontier — Accuracy vs. Wall-Clock Time",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "figure2_pareto")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Log Scaling
# ---------------------------------------------------------------------------


def figure3_logscaling() -> None:
    """
    Single log-log plot.

    1. Theorem 4 bound: 2·⌈log₂(N)⌉ + 1  (solid black)
    2. Empirical mean ± std from scaling.json (green points + error bars)
    3. Grid search reference: N/10 evaluations (dashed grey)

    X-axis: N = k_max − k_min  (log scale)
    Y-axis: evaluation count   (log scale)
    """
    data = _load("scaling")

    # N values corresponding to SCALING_KMAXES = [100, 200, 500, 1000, 2000], K_MIN=10
    default_ns = [90, 190, 490, 990, 1990]
    N_theory  = np.array(default_ns, dtype=float)
    # Theorem 4: 2·⌈log₂(N)⌉ + 1
    theory_evals = np.array(
        [2 * math.ceil(math.log2(n)) + 1 for n in N_theory], dtype=float
    )
    grid_evals = N_theory / 10.0

    fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.0))

    ax.plot(N_theory, theory_evals, color=C["theory"], linewidth=1.4,
            linestyle="-", zorder=3,
            label=r"Theorem 4: $2\lceil\log_2 N\rceil + 1$")

    ax.plot(N_theory, grid_evals, color=C["ref"], linewidth=1.0,
            linestyle="--", zorder=2,
            label=r"Grid search: $N/10$ evals")

    if data is not None:
        ns_emp, means_emp, stds_emp = [], [], []
        for k_max_str, r in sorted(data.items(), key=lambda x: int(x[0])):
            ns_emp.append(r["n"])
            means_emp.append(r["evaluations_mean"])
            stds_emp.append(r["evaluations_std"])

        ns_emp    = np.array(ns_emp,    dtype=float)
        means_emp = np.array(means_emp, dtype=float)
        stds_emp  = np.array(stds_emp,  dtype=float)

        ax.errorbar(ns_emp, means_emp, yerr=stds_emp,
                    fmt="o", color=C["two_phase"], markersize=5,
                    capsize=3, capthick=0.8, elinewidth=0.8, zorder=4,
                    label="Two-Phase (empirical)")

        for n, mean, r in zip(ns_emp, means_emp, data.values()):
            if not r["within_bound"]:
                ax.annotate("!", xy=(n, mean), fontsize=8, color="red",
                            ha="center", va="bottom")
    else:
        ax.text(0.5, 0.6,
                "Scaling results not yet available.\n"
                "Run: python run_experiments.py scaling",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#aaaaaa", style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Search space size $N = k_{\max} - k_{\min}$", fontsize=10)
    ax.set_ylabel("Evaluation count", fontsize=10)
    ax.set_title("Figure 3: Evaluation Count vs. Search Space Size",
                 fontsize=10, pad=4)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, which="both", color="#eeeeee", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save_fig(fig, "figure3_logscaling")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Batch runner + CLI
# ---------------------------------------------------------------------------


def generate_all() -> None:
    _apply_style()
    print("\nGenerating Figure 0 — Monotonicity Validation...")
    figure0_monotonicity()
    print("\nGenerating Figure 1 — Convergence Trajectory...")
    figure1_convergence()
    print("\nGenerating Figure 2 — Pareto Frontier...")
    figure2_pareto()
    print("\nGenerating Figure 3 — Log Scaling...")
    figure3_logscaling()
    print(f"\nAll figures saved to {FIGURES_DIR}/")


_FIGURE_FNS = {
    "0": figure0_monotonicity,
    "1": figure1_convergence,
    "2": figure2_pareto,
    "3": figure3_logscaling,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate publication figures for the Two-Phase Search paper."
    )
    parser.add_argument(
        "--fig", choices=["0", "1", "2", "3"],
        default=None,
        help="Which figure to generate (default: all four)"
    )
    args = parser.parse_args()

    _apply_style()

    if args.fig is None:
        generate_all()
    else:
        labels = {
            "0": "Monotonicity Validation",
            "1": "Convergence Trajectory",
            "2": "Pareto Frontier",
            "3": "Log Scaling",
        }
        print(f"\nGenerating Figure {args.fig} — {labels[args.fig]}...")
        _FIGURE_FNS[args.fig]()
