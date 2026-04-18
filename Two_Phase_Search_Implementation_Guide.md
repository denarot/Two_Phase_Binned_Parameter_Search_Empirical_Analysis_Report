# Two-Phase Search — Implementation Guide: Prompt-by-Prompt

**Purpose:** This document describes the code to be generated for the Two-Phase Search paper's experimental validation. Each section is a self-contained prompt that can be pasted into a chat. Prompts are ordered by dependency — each builds on the outputs of the previous ones. The outputs of each prompt are specified so you can verify completeness before moving to the next.

**Prerequisites:** Upload Two_Phase_Search_Paper_Clean_v4.docx and the Project Briefing to the chat before starting.

**Environment:** Python 3.8+, scikit-learn 1.2.0+, numpy, scipy, matplotlib. Optional: scikit-optimize (for Bayesian optimisation baseline).

---

## Prompt 1: Core Algorithm Implementation

**What to ask:**

> Implement the Two-Phase Search algorithm from Algorithm 1 in Section 5.2 of the uploaded paper. Write a Python function `two_phase_search(X, y, k_min=10, k_max=1000, epsilon=0.001, window_size=5, cv_folds=5, random_state=42)` that:
>
> 1. Takes training data X, y and search parameters as input
> 2. Uses scikit-learn RandomForestClassifier with 5-fold cross-validation as the evaluation oracle f-tilde(k)
> 3. Implements Phase 1 (exponential bracketing) using the doubling schedule k_i = k_min * 2^i
> 4. Implements Phase 2 (binary refinement) on the discrete integer domain
> 5. Uses the windowed average gradient criterion from Assumption 2 for plateau detection
> 6. Records the full evaluation trajectory: a list of (step_number, phase, k_value, accuracy, wall_clock_seconds) at every evaluation in both phases
> 7. Returns a result object or dictionary containing: k_hat (the found optimal tree count), best_accuracy, total_evaluations, phase1_evaluations, phase2_evaluations, bracket_L, bracket_U, trajectory (the full list from point 6), and total_wall_clock_seconds
>
> Include a warm-start optimisation: when Phase 2 evaluates a k smaller than a previously evaluated k', reuse a subset of the previously fitted trees rather than retraining from scratch.
>
> Save the file as `two_phase_search.py`.

**Expected output:** A single Python file containing the `two_phase_search` function and any helper functions it needs. Verify it runs on a small synthetic dataset before proceeding.

**Verification test:** Run `two_phase_search(X, y, k_min=10, k_max=200)` on the Iris dataset (small, fast) and confirm it returns a result with k_hat between 10 and 200, total_evaluations < 20, and a non-empty trajectory list.

---

## Prompt 2: Baseline Implementations

**What to ask:**

> Using the same evaluation oracle (scikit-learn RandomForestClassifier with 5-fold cross-validation) as the Two-Phase Search implementation in `two_phase_search.py`, implement three baseline methods in a file called `baselines.py`:
>
> 1. `grid_search(X, y, k_min=10, k_max=1000, step=10, cv_folds=5, random_state=42)` — evaluates every step-th k from k_min to k_max. Returns the same result structure as two_phase_search: k_hat, best_accuracy, total_evaluations, trajectory, total_wall_clock_seconds.
>
> 2. `random_search(X, y, k_min=10, k_max=1000, n_evaluations=50, cv_folds=5, random_state=42)` — samples n_evaluations random k values uniformly from [k_min, k_max]. Returns the same result structure.
>
> 3. `bayesian_search(X, y, k_min=10, k_max=1000, n_iterations=30, cv_folds=5, random_state=42)` — uses Gaussian Process with Matérn kernel to model the accuracy surface and select evaluation points. Use scikit-optimize's gp_minimize or implement a simple GP-UCB loop. Returns the same result structure.
>
> All three must record the full (step_number, k_value, accuracy, wall_clock_seconds) trajectory in the same format as two_phase_search, so results are directly comparable.
>
> Save as `baselines.py`.

**Expected output:** A single Python file with three functions sharing the same interface and return structure as the Two-Phase Search implementation.

**Verification test:** Run all three on Iris with k_max=200. Grid search should return ~19 evaluations (with step=10), random search should return 50, Bayesian should return 30.

---

## Prompt 3: Dataset Preparation

**What to ask:**

> Create a file `datasets.py` that provides a function `load_dataset(name)` returning (X_train, X_test, y_train, y_test) for three datasets:
>
> 1. **Covertype** — use sklearn.datasets.fetch_covtype(). Use the first 50,000 samples for tractability (the full 580k is too slow for 10-seed experiments). Fixed 80/20 train-test split.
>
> 2. **MNIST** — use sklearn.datasets.fetch_openml('mnist_784'). Flatten to tabular. Use 10,000 samples for tractability. Fixed 80/20 split.
>
> 3. **Adult** — use sklearn.datasets.fetch_openml('adult', version=2). Handle categorical features (one-hot encode or ordinal encode). Fixed 80/20 split.
>
> The train-test split must be deterministic (random_state=42) so all methods see the same data. The function should cache the loaded data so it's only fetched once per session.
>
> Also create a `get_dataset_names()` function that returns `['covertype', 'mnist', 'adult']`.
>
> Save as `datasets.py`.

**Expected output:** A single Python file. Verify each dataset loads correctly and that the shapes are reasonable (print X_train.shape, y_train.shape for each).

**Note on sample sizes:** The paper says "Covertype (580k samples, 54 features)" but running 10 seeds × 4 methods × full dataset will take days. Using 50k samples preserves the structure while keeping each experiment under a few minutes. Document the actual sample size used in the final paper — a footnote is sufficient. If you have access to substantial compute and want to run on the full dataset, remove the subsampling. The algorithm's properties (O(log N) evaluations, convergence rate) do not depend on dataset size.

---

## Prompt 4: Experimental Harness

**What to ask:**

> Create the main experimental harness in `run_experiments.py` that orchestrates all experiments described in Section 7 of the paper. It should:
>
> 1. Import from `two_phase_search.py`, `baselines.py`, and `datasets.py`
>
> 2. Run each experiment as a separate function that can be called independently:
>    - `run_monotonicity_validation()` — Section 7.1 / Figure 0
>    - `run_primary_comparison()` — Section 7.2 / Table 1 + Figure 1
>    - `run_scaling_validation()` — Section 7.4 / Table 3 + Figure 3
>    - `run_noise_robustness()` — Section 7.5 / Table 4
>    - `run_bidirectional_validation()` — Section 7.6 / Table 5
>    - `run_edge_deployment()` — Section 7.7 / Table 6
>    - `run_window_ablation()` — Section 7.8 / Table 7
>
> 3. Each function should:
>    - Run 10 independent trials with seeds 42–51
>    - Save raw results to a JSON file in a `results/` directory (one file per experiment)
>    - Print a summary table to stdout
>    - Return the results dictionary
>
> 4. Include a `run_all()` function that calls everything in order and saves a master results file
>
> 5. Include a `if __name__ == '__main__'` block that accepts a command-line argument for which experiment to run (or 'all')
>
> Here are the specific parameters for each experiment:
>
> **Monotonicity validation (Section 7.1):** For each dataset, train Random Forests at k ∈ {10, 20, 50, 100, 200, 300, 500, 750, 1000} and record OOB accuracy. 10 seeds. Validate that the curve is monotonically non-decreasing with <0.005 variation in the plateau region.
>
> **Primary comparison (Section 7.2):** Run all four methods (grid search with step=10, random search with 50 evaluations, Bayesian optimisation with 30 iterations, Two-Phase Search with k_min=10, k_max=1000, ε=0.001, w=5) on all three datasets. 10 seeds each. Record: k* found, accuracy, evaluation count, wall-clock time. Compute success rate as % of runs where |k_method − k_grid| ≤ 10.
>
> **Scaling validation (Section 7.4):** Run Two-Phase Search with k_max ∈ {100, 200, 500, 1000, 2000}, k_min=10. Covertype only. 10 seeds. Record evaluation count and compare against Theorem 4 prediction.
>
> **Noise robustness (Section 7.5):** Inject clipped Gaussian noise at σ ∈ {0.000, 0.010, 0.020, 0.050} into the evaluation oracle. Covertype only. 10 seeds. Record k* found and success rate.
>
> **Bidirectional validation (Section 7.6):** Run forward (k_min→k_max) and a reverse variant (k_max→k_min with exponential halving then binary refinement). All three datasets. 10 seeds. Record k* from each direction and compute |k_forward − k_reverse| / max(k_f, k_r).
>
> **Edge deployment (Section 7.7):** For each dataset, train Random Forests at three tree counts: grid search optimum k*, Two-Phase k-hat, and 500 (default). Measure serialised model size (joblib + sys.getsizeof or pickle size in MB) and inference time (mean over 1000 predictions in ms/sample). Check 4 MB budget compliance.
>
> **Window ablation (Section 7.8):** Run Two-Phase Search with w ∈ {3, 5, 10}, ε=0.001, on Covertype. 10 seeds. Record success rate, false positive rate (premature plateau detection), and evaluation count.
>
> Save as `run_experiments.py`.

**Expected output:** A single Python file that can run any experiment independently via command line. Each experiment saves JSON results and prints a summary.

**Verification test:** Run `python run_experiments.py monotonicity` on one dataset with 2 seeds instead of 10 to verify the pipeline works end-to-end before committing to the full run.

---

## Prompt 5: Statistical Analysis

**What to ask:**

> Create `analysis.py` that loads the JSON results from `results/` and computes the derived statistics needed for the paper:
>
> 1. **Table 2 statistics (Section 7.3):** Paired t-tests at α = 0.05 for Two-Phase vs. each baseline on three metrics (evaluations, wall-clock time, k* error). Compute Cohen's d = (μ₁ − μ₂) / σ_pooled for each comparison. Classify effect size: small (0.2), medium (0.5), large (0.8), huge (2.0+).
>
> 2. **Aggregate statistics for Table 1:** Compute per-dataset and aggregate means ± std for evaluation count, speedup relative to grid search, and success rate.
>
> 3. **Appendix A breakdown:** Extract Phase 1 and Phase 2 evaluation counts separately from the trajectory data. Compute means ± std and compare against the theoretical predictions: Phase 1 ≤ ⌈log₂(k*/k_min)⌉ + 1, Phase 2 ≤ ⌈log₂(U − L)⌉.
>
> 4. **Print all results** in a format that can be directly transcribed into the paper tables.
>
> Save as `analysis.py`.

**Expected output:** A Python file that reads from `results/` and prints formatted table contents for Tables 1, 2, and Appendix A.

---

## Prompt 6: Figure Generation

**What to ask:**

> Create `generate_figures.py` that loads results from `results/` and produces four publication-quality figures matching the specifications in the paper:
>
> **Figure 0 (Section 7.1): Monotonicity Validation**
> - Three subplots (one per dataset)
> - X-axis: tree count k (log scale), from 10 to 1000
> - Y-axis: OOB accuracy
> - Plot mean accuracy across 10 seeds as a solid line with shaded ±1 std band
> - Vertical dashed line at the empirical plateau onset k_p for each dataset
> - Save as `figures/figure0_monotonicity.pdf` and `.png`
>
> **Figure 1 (Section 7.2): Convergence Trajectory**
> - Three subplots (one per dataset)
> - X-axis: evaluation step number (1 through T_total)
> - Y-axis: cross-validation accuracy at the k tested at that step
> - Show one representative run (seed 42) as connected points
> - Vertical dashed line at Phase 1 → Phase 2 transition
> - Horizontal dashed line at grid search optimum accuracy
> - Colour Phase 1 points differently from Phase 2 points
> - Save as `figures/figure1_convergence.pdf` and `.png`
>
> **Figure 2 (Section 7.3): Pareto Frontier**
> - Three subplots (one per dataset)
> - X-axis: total wall-clock training time (seconds, log scale)
> - Y-axis: best accuracy achieved
> - Plot all 10 runs per method as individual points with distinct markers: grid search (square, grey), random search (triangle, blue), Bayesian (diamond, orange), Two-Phase (circle, green)
> - Draw the Pareto frontier through non-dominated points
> - Save as `figures/figure2_pareto.pdf` and `.png`
>
> **Figure 3 (Section 7.4): Log Scaling**
> - Single plot
> - X-axis: search space size N (log scale): 90, 190, 490, 990, 1990
> - Y-axis: evaluation count
> - Three elements: (1) theoretical prediction from Theorem 4 as solid black line, (2) empirical mean ± std as green points with error bars, (3) grid search reference (evaluations = N/10) as dashed grey line
> - Log-log axes to show the linear-vs-logarithmic divergence
> - Save as `figures/figure3_logscaling.pdf` and `.png`
>
> Use matplotlib with a clean, publication-appropriate style (no grid, minimal chartjunk, legible at column width ~3.5 inches). Use consistent colours across all figures. Font size 10pt for axis labels, 8pt for tick labels.
>
> Save as `generate_figures.py`.

**Expected output:** A Python file that reads from `results/` and saves eight files (four PDFs, four PNGs) to `figures/`.

**Verification test:** Run after Prompt 4's experiments have completed. Check that all four figures render correctly and match the specifications in the paper.

---

## Prompt 7: Reverse Search Implementation (for Table 5)

**What to ask:**

> The bidirectional convergence experiment (Section 7.6, Table 5) requires a reverse variant of Two-Phase Search that starts from k_max and searches downward. Add a function `two_phase_search_reverse(X, y, k_min=10, k_max=1000, epsilon=0.001, window_size=5, cv_folds=5, random_state=42)` to `two_phase_search.py` that:
>
> 1. Phase 1 (Reverse Exponential): Start at k_max and halve: k_i = k_max / 2^i. Detect when the accuracy begins dropping below the plateau (i.e., the negative gradient exceeds ε). This brackets the plateau onset from above.
>
> 2. Phase 2 (Binary Refinement): Same as the forward version, binary search within the bracket [L, U].
>
> 3. Returns the same result structure as the forward version.
>
> The convergence test in Section 7.6 compares k_forward and k_reverse: |k_forward − k_reverse| / max(k_f, k_r) should be < 0.10.

**Expected output:** An updated `two_phase_search.py` with both forward and reverse functions.

---

## Prompt 8: Results-to-Paper Pipeline

**What to ask:**

> Create `update_paper_values.py` that reads the JSON results from `results/` and the analysis output, and prints a complete replacement guide — every value in the paper that needs to change, with the old projected value and the new empirical value side by side. Organise by table:
>
> - Table 1: for each dataset × method, print the old and new values for k* found, accuracy, evaluations, time, speedup, success %
> - Table 2: for each comparison × metric, print old and new p-value, Cohen's d, interpretation
> - Table 3: for each search space size, print old and new actual evaluations
> - Table 4: for each noise level, print old and new k* and success rate
> - Table 5: for each dataset, print old and new forward k*, reverse k*, difference
> - Table 6: for each dataset × configuration, print old and new tree count, inference time, memory, budget compliance
> - Table 7: for each window size, print old and new success rate, FP rate, evaluations
> - Appendix A: for each dataset, print old and new Phase 1 and Phase 2 evaluation counts
>
> Also flag any headline numbers that changed materially:
> - Did the aggregate speedup shift from 8.4×?
> - Did the aggregate success rate shift from 97%?
> - Did any dataset's Two-Phase k-hat change the 4 MB budget compliance from Table 6?
>
> This output serves as the editing guide for updating the paper.
>
> Save as `update_paper_values.py`.

**Expected output:** A script that produces a complete old-vs-new comparison for every projected value in the paper, plus a flag for any headline-level changes.

---

## Prompt 9: End-to-End Execution Script

**What to ask:**

> Create `run_all.sh` — a bash script that runs the entire pipeline from start to finish:
>
> ```
> #!/bin/bash
> set -e
>
> echo "=== Step 1: Running all experiments ==="
> python run_experiments.py all
>
> echo "=== Step 2: Running statistical analysis ==="
> python analysis.py
>
> echo "=== Step 3: Generating figures ==="
> python generate_figures.py
>
> echo "=== Step 4: Generating paper update guide ==="
> python update_paper_values.py
>
> echo "=== Complete ==="
> echo "Results in results/"
> echo "Figures in figures/"
> echo "Review update_paper_values.py output for paper edits"
> ```
>
> Also create a `requirements.txt` listing all Python dependencies with version pins.
>
> Save as `run_all.sh` and `requirements.txt`.

**Expected output:** Two files. The bash script should run the full pipeline. Requirements.txt should list: scikit-learn, numpy, scipy, matplotlib, scikit-optimize (if used for Bayesian baseline), joblib.

---

## Prompt 10: Verification and Sanity Checks

**What to ask:**

> Create `verify_results.py` that runs a series of sanity checks on the results in `results/` to catch problems before updating the paper:
>
> 1. **Monotonicity check:** For each dataset, verify that Figure 0 data is monotonically non-decreasing (within the 0.005 tolerance stated in Section 7.1).
>
> 2. **Evaluation count check:** For each Two-Phase run, verify total evaluations ≤ 2 * ⌈log₂(k_max/k_min)⌉ + 2 (the theoretical upper bound from Theorem 4, with margin).
>
> 3. **Phase count check:** Verify Phase 1 evals ≤ ⌈log₂(k_max/k_min)⌉ + 1 and Phase 2 evals ≤ ⌈log₂(U − L)⌉ for every run. Flag any violations.
>
> 4. **Accuracy preservation check:** For each Two-Phase run, verify |accuracy_twophase − accuracy_gridsearch| < 0.01 (the ε-optimality claim from the abstract).
>
> 5. **Bidirectional convergence check:** Verify |k_forward − k_reverse| / max(k_f, k_r) < 0.10 for all datasets.
>
> 6. **Budget compliance check:** Verify Table 6 memory values and 4 MB compliance flags are consistent.
>
> 7. **Success rate check:** Verify aggregate success rate computation is correct.
>
> Print PASS/FAIL for each check with details on any failures. If any check fails, the paper values should not be updated until the failure is understood.
>
> Save as `verify_results.py`.

**Expected output:** A verification script that produces a clear PASS/FAIL report. All checks should pass before proceeding to update the paper.

---

## Summary: Prompt Order and Dependencies

```
Prompt 1: two_phase_search.py          (no dependencies)
Prompt 2: baselines.py                 (depends on: oracle interface from Prompt 1)
Prompt 3: datasets.py                  (no dependencies)
Prompt 7: reverse search addition      (depends on: Prompt 1)
Prompt 4: run_experiments.py           (depends on: Prompts 1, 2, 3, 7)
Prompt 9: run_all.sh + requirements    (depends on: Prompts 4, 5, 6)
  --- RUN EXPERIMENTS HERE ---
Prompt 5: analysis.py                  (depends on: results from Prompt 4)
Prompt 6: generate_figures.py          (depends on: results from Prompt 4)
Prompt 8: update_paper_values.py       (depends on: results from Prompt 4, analysis from Prompt 5)
Prompt 10: verify_results.py           (depends on: results from Prompt 4)
```

**Recommended execution order:**
1. Prompts 1, 2, 3, 7 (in any order — these are independent code files)
2. Prompt 4 (harness that wires them together)
3. Prompt 9 (execution script)
4. **Run the experiments** (this is the long step — may take hours depending on hardware)
5. Prompts 5, 6, 8, 10 (post-processing — all depend on results existing)
6. Prompt 10 first (verify before updating)
7. Then Prompts 5, 6, 8 (analysis, figures, paper update guide)

---

## File Inventory (after all prompts are complete)

```
two_phase_search.py          Core algorithm (forward + reverse)
baselines.py                 Grid search, random search, Bayesian optimisation
datasets.py                  Dataset loading and preparation
run_experiments.py           Experimental harness (7 experiments)
analysis.py                  Statistical analysis (Tables 2, Appendix A)
generate_figures.py          Figure generation (Figures 0–3)
update_paper_values.py       Old-vs-new value comparison for paper editing
verify_results.py            Sanity checks on experimental results
run_all.sh                   End-to-end execution script
requirements.txt             Python dependencies
results/                     JSON output from experiments
figures/                     PDF and PNG output from figure generation
```
