# Two-Phase Search Paper — Completion Checklist

**Last updated:** April 2026  
**Current paper version:** v4 (Two_Phase_Search_Paper_Clean_v4.docx)  
**Companion document:** Two_Phase_Search_Research_Journal_v3.docx

---

## Phase A: Experimental Code (prerequisite for everything else)

- [ ] **A1. Write the Two-Phase Search implementation in Python**
  - Implement Algorithm 1 (Section 5.2) against scikit-learn RandomForestClassifier
  - Parameters: k_min, k_max, ε, w as inputs
  - Must record full evaluation trajectory: (k_i, f-tilde(k_i)) at every step of both phases (required for Figure 1 and Appendix A)
  - Must record wall-clock time per evaluation (required for Figure 2)
  - Must record Phase 1 and Phase 2 evaluation counts separately (required for Appendix A)

- [ ] **A2. Write the baseline implementations**
  - Grid search: evaluate every Δ-th k in [k_min, k_max]
  - Random search: 50 uniformly sampled k values
  - Bayesian optimisation: 30 iterations, Gaussian Process with Matérn kernel (use scikit-optimize or similar)

- [ ] **A3. Write the experimental harness**
  - 10 independent runs per configuration (seeds 42–51)
  - Fixed 80/20 train-test split per dataset
  - Record: k* found, accuracy, evaluation count, wall-clock time, model memory footprint (serialised size in MB), inference time (ms/sample over 1000 samples)
  - Document hardware: CPU model, RAM, scikit-learn version

- [ ] **A4. Prepare datasets**
  - Covertype (580k samples, 54 features)
  - MNIST (use scikit-learn fetch; flatten to tabular)
  - Adult (UCI repository)

---

## Phase B: Run Experiments

- [ ] **B1. Monotonicity validation (Figure 0, Section 7.1)**
  - Plot OOB accuracy vs. k ∈ {10, 20, 50, 100, 200, 300, 500, 750, 1000} for all three datasets
  - Verify monotonically non-decreasing with <0.005 accuracy variation in plateau region across 10 seeds
  - If any dataset fails, exclude from main results and document why
  - Record empirical k_p (plateau onset) for each dataset

- [ ] **B2. Primary method comparison (Table 1, Section 7.2)**
  - Run all four methods on all three datasets
  - Record all metrics listed in A3
  - Compute success rate: % of runs finding k* within 10 trees of grid search optimum

- [ ] **B3. Convergence trajectory data (Figure 1, Section 7.2)**
  - Extract from the same runs as B2 — no additional experiments needed
  - Requires the full (k_i, f-tilde(k_i)) trajectory from A1

- [ ] **B4. Statistical significance (Table 2, Section 7.3)**
  - Paired t-tests on the 10 runs from B2
  - Compute Cohen's d for evaluations, wall-clock time, and k* error
  - Two-Phase vs. grid search, vs. random search, vs. Bayesian optimisation

- [ ] **B5. Pareto frontier data (Figure 2, Section 7.3)**
  - Extract from the same runs as B2 — no additional experiments needed
  - Plot accuracy vs. wall-clock time for all four methods

- [ ] **B6. O(log N) scaling validation (Table 3 and Figure 3, Section 7.4)**
  - Run Two-Phase Search with k_max ∈ {100, 200, 500, 1000, 2000}, k_min = 10
  - 10 runs per search space size
  - Compare empirical evaluation counts against Theorem 4 predictions
  - Plot log-log with theoretical prediction line and grid search reference

- [ ] **B7. Noise robustness (Table 4, Section 7.5)**
  - Inject clipped Gaussian noise at σ ∈ {0.000, 0.010, 0.020, 0.050}
  - Covertype dataset only
  - Record k* found and success rate at each noise level

- [ ] **B8. Bidirectional convergence (Table 5, Section 7.6)**
  - Run forward (k_min → k_max) and reverse (k_max → k_min) search
  - All three datasets
  - Compute |k_forward − k_reverse| / max(k_f, k_r)

- [ ] **B9. Edge deployment impact (Table 6, Section 7.7)**
  - Measure serialised model size (MB) and inference time (ms/sample) for: grid search k*, Two-Phase k-hat, and default 500 trees
  - All three datasets
  - Check 4 MB flash budget compliance

- [ ] **B10. Window size ablation (Table 7, Section 7.8)**
  - w ∈ {3, 5, 10} on Covertype
  - Record success rate, false positive rate, mean evaluations

- [ ] **B11. Phase-by-phase breakdown (Appendix A)**
  - Extract from the same runs as B2/B6
  - Report Phase 1 and Phase 2 evaluation counts separately per dataset

---

## Phase C: Generate Figures

- [ ] **C1. Figure 0 — Monotonicity validation plots** (from B1 data)
- [ ] **C2. Figure 1 — Convergence trajectory** (from B3 data)
- [ ] **C3. Figure 2 — Pareto frontier** (from B5 data)
- [ ] **C4. Figure 3 — Log scaling** (from B6 data)

---

## Phase D: Update Paper with Empirical Results

- [ ] **D1. Replace projected values in Tables 1–7 and Appendix A with empirical values**
- [ ] **D2. Insert generated Figures 0–3 into the paper**
- [ ] **D3. Remove the opening "Note on numerical values" block**
- [ ] **D4. Remove all per-table "(projected values, pending empirical validation)" captions**
- [ ] **D5. Update abstract**: change "predicts a 7–15× speedup" to actual measured speedup
- [ ] **D6. Update conclusion**: change "projected" language to past-tense empirical claims
- [ ] **D7. Update Section 7 intro**: add hardware specifications (CPU model, RAM)
- [ ] **D8. Review whether any headline numbers shifted materially** — if the 7–15× speedup, 97% success rate, or 8.4× aggregate changed, adjust the abstract, Section 2.5, and Section 9 accordingly

---

## Phase E: Pre-Submission Polish

- [ ] **E1. Final proofread for consistency** — check all cross-references (Table N, Figure N, Theorem N, Section N) are correct
- [ ] **E2. Verify Appendix B checklist** — all 19 items should now show "Resolved" with no "pending" qualifiers
- [ ] **E3. Confirm Section 6.8 summary table** — verify bounds in the summary match the theorem statements exactly
- [ ] **E4. Format for target venue** — apply AutoML workshop or NeurIPS HPO workshop template
- [ ] **E5. Write acknowledgments section** if applicable
- [ ] **E6. Compile references** — verify all citations (Breiman 2001, Oshiro 2012, Probst & Boulesteix 2017, Pham-Quoc 2024, Daghero 2022, Alkhoury & Welke 2024/2025, Bergstra & Bengio 2012, Snoek 2012, Hutter 2011, Bergstra 2011, Li 2017, Domhan 2015, Maclaurin 2015) are present and complete

---

## Phase F: Submission and Release

- [ ] **F1. Post preprint to arXiv**
- [ ] **F2. Create GitHub repository** — paper PDF, reproducible experiment code, README
- [ ] **F3. Build and release Python package** (pip install) — two_phase_search or rfk_search
- [ ] **F4. Submit to target venue** (AutoML workshop or NeurIPS HPO workshop)
- [ ] **F5. Update briefing document and research journal** for post-submission records

---

## Explicitly Deferred (Future Work, not blocking submission)

- [ ] FPGA / microcontroller physical validation (Section 8.3)
- [ ] ε sensitivity ablation (Section 8.3)
- [ ] Multi-parameter extension to mtry, nodesize (Section 8.3)
- [ ] Distributional output / confidence interval extension (Research Journal Section 3.3)
- [ ] Stumpy Forest follow-up paper (Research Journal Section 3.2)
