# PROJECT BRIEFING: Two-Phase Search Paper — Status Summary (April 2026)

## What the paper is

A theoretical computer science / machine learning paper proving that a Two-Phase Search algorithm (exponential bracketing + binary refinement) finds the optimal Random Forest tree count k* in O(log N) evaluations — provably matching the information-theoretic lower bound. The core claim is 7–15× speedup over grid search with an ε-optimal guarantee. The paper argues this is the only sub-grid-search method that provides a provable optimality guarantee on the single-parameter monotonic search problem.

## Current paper version

The uploaded file **Two_Phase_Search_Paper_Clean_v4.docx** is the current draft. A companion document (Two_Phase_Search_Research_Journal_v3.docx) exists separately for provenance and design philosophy notes — it is not part of the paper.

## Paper structure (Section → content map)

- **Section 1**: Abstract (complete)
- **Section 2**: Introduction and Practical Motivation
  - 2.1: Pragmatist's valid objection (acknowledges the "this is a non-problem" position)
  - 2.2: Edge computing as primary motivating application (Pham-Quoc 2024, Daghero 2022)
  - 2.3: Additional contexts (production SLAs, AutoML pipelines, research benchmarking, distributed hyperparameter search)
  - 2.4: Why OOB monitoring is not a full substitute
  - 2.5: **The Optimality Guarantee: Two Audiences, Two Benefits** — frames the contribution around the optimality guarantee (Theorem 7) rather than just the speedup; distinguishes compute-constrained practitioners (who get a strict improvement over random search/BO) from compute-rich practitioners (who get either direct replacement or a hybrid bracket-then-grid-search workflow)
- **Section 3**: Related Work (four subsections: RF tree count, HPO methods, edge deployment, binary/exponential search)
- **Section 4**: Problem Setup and Assumptions (Assumptions 1–4: monotonic-then-plateau, plateau threshold, discrete Lipschitz, bounded noise)
- **Section 5**: The Two-Phase Search Algorithm
  - 5.1: Algorithm overview (Phase 1 and Phase 2)
  - 5.2: Pseudocode (Algorithm 1, 20 steps)
  - 5.3: Correctness and termination + implementation note on warm-start reuse
- **Section 6**: Theoretical Results (seven theorems + Section 6.8 summary table)
- **Section 7**: Experiments (all values currently projected, pending empirical validation)
  - 7.1: Monotonicity validation — Figure 0 specification (pending generation)
  - 7.2: Primary results — Table 1 + **Figure 1 specification** (convergence trajectory, pending generation)
  - 7.3: Statistical significance — Table 2 + **Figure 2 specification** (Pareto frontier, pending generation)
  - 7.4: Complexity validation — Table 3 + **Figure 3 specification** (log scaling, pending generation)
  - 7.5: Noise robustness — Table 4
  - 7.6: Bidirectional convergence — Table 5
  - 7.7: Edge deployment impact — Table 6
  - 7.8: Window size ablation — Table 7
- **Section 8**: Discussion, Limitations, and Future Work
  - 8.1: When this algorithm is and is not needed (softened to acknowledge both audiences from 2.5)
  - 8.2: Limitations (monotonicity assumption, single hyperparameter, noise model, ε choice)
  - 8.3: Future work (multi-parameter extension, FPGA validation, ε sensitivity ablation, non-RF extensions)
- **Section 9**: Conclusion
- **Appendix A**: Phase-by-phase evaluation breakdown (projected values)
- **Appendix B**: 19-item revision checklist (all theoretical items resolved; empirical items pending)

## What has been completed

### Theoretical framework (all complete)
- Noise model unified to bounded noise |η| ≤ σ throughout — Gaussian assumption removed
- Discrete vs. continuous domain tension resolved across all seven proofs
- Theorem 6 factor-of-2 carried through explicitly
- Theorem 7 comparison model justified via adversarial argument
- Monotonicity assumption grounded with citations (Oshiro 2012, Probst & Boulesteix 2017)
- "Golden Number" terminology removed; "Two-Phase Search" used throughout
- Plateau threshold ε formalised as Assumption 2

### Paper structure (all complete)
- Abstract
- Related Work section (four subsections)
- Formal algorithm pseudocode (Algorithm 1, 20 steps)
- Section 2.5: Two Audiences, Two Benefits (optimality guarantee framing)
- Distributed coordination overhead bullet in Section 2.3
- Warm-start implementation note after Section 5.3
- Section 8.1 softened to acknowledge both audiences
- Abstract sentence on optimality guarantee uniqueness

### Experimental design (specifications written, data pending)
- Figure 0 specification: monotonicity validation plot for all three datasets
- Figure 1 specification: convergence trajectory showing Phase 1/Phase 2 regimes
- Figure 2 specification: Pareto frontier — accuracy vs. wall-clock time for all four methods
- Figure 3 specification: log scaling — evaluation count vs. N on log-log plot with theory overlay
- Table 1–7 structures defined with projected values
- Appendix A structure defined with projected values
- All projection disclaimers added (opening note + per-table captions)

## What remains to be done

### Blocking submission (must complete)
1. **Write experimental code** — implement Algorithm 1, three baselines, and the experimental harness in Python against scikit-learn. Code must record full evaluation trajectories (k_i, f-tilde(k_i)) at every step, wall-clock time per evaluation, Phase 1/Phase 2 counts separately, and model memory footprint.
2. **Run all experiments** — Tables 1–7 and Appendix A contain projected values. Empirical data needed from three datasets (Covertype, MNIST, Adult), 10 runs per configuration (seeds 42–51).
3. **Generate Figures 0–3** — specifications are written in the paper (Sections 7.1, 7.2, 7.3, 7.4); code needs to produce the plots.
4. **Replace projected values with empirical values** — update Tables 1–7 and Appendix A.
5. **Remove projection disclaimers** — delete the opening "Note on numerical values" block and all per-table "(projected values, pending empirical validation)" captions.
6. **Update abstract and conclusion** — change "predicts a 7–15× speedup" to measured values; change "projected" language to past-tense empirical claims.
7. **Add hardware specifications** — document CPU model and RAM in Section 7 intro.
8. **Check headline numbers** — if the 7–15× speedup, 97% success rate, or 8.4× aggregate shifted materially, update the abstract, Section 2.5, Section 7.7 narrative, and Section 9.
9. **Format for target venue** — apply workshop template (AutoML or NeurIPS HPO workshop).
10. **Compile references** — verify all 15 citations are present and complete.

### Not blocking submission (future work)
- FPGA / microcontroller physical validation (Section 8.3)
- ε sensitivity ablation (Section 8.3)
- Multi-parameter extension to mtry, nodesize (Section 8.3)
- Distributional output extension (research journal only, not in paper)
- Stumpy Forest follow-up paper (research journal only)
- Python package release on PyPI (post-submission)

## Key numbers to remember (all projected, pending validation)
- Covertype: 99 grid search evaluations → ~13 Two-Phase evaluations (7.6× speedup)
- Aggregate across datasets: mean 12 ±1 evaluations, 8.4× speedup, 97% success rate
- Edge result: Two-Phase k-hat fits 4 MB flash budget on 2/3 datasets; default 500 trees fails all three
- Window size recommendation: w = 5 (3% false positive rate, 13.1 ±1.4 mean evaluations)

## Target venues
- AutoML workshop or NeurIPS HPO workshop first
- NeurIPS/ICML main track after empirical validation and FPGA validation

## Figure data requirements summary (for experimental code)

| Figure | Data source | Key requirement |
|--------|-------------|-----------------|
| Figure 0 | Separate monotonicity sweep | OOB accuracy at 9 tree counts × 3 datasets × 10 seeds |
| Figure 1 | Same runs as Table 1 | Full (k_i, f-tilde(k_i)) trajectory at every step of both phases |
| Figure 2 | Same runs as Table 1 | Per-run (wall-clock time, best accuracy) for all four methods |
| Figure 3 | Same runs as Table 3 | Per-run evaluation count at 5 search space sizes |

## What NOT to include in this chat
- Research journal material (provenance, analogies, design philosophy) — this is catalogued separately and not needed for the experimental phase
- Earlier transcript material about Stumpy Forests, stratified feature sampling, undervolting, or gaming analogies — all catalogued in the research journal
- Pre-revision versions of the paper — v4 is authoritative
