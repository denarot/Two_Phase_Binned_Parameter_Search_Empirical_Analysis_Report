# Statement of AI Use — Two-Phase Search Paper

## Disclosure of Generative AI Tools in Research Development

This paper was developed with the assistance of multiple generative AI tools across distinct stages of the research process. The author is the sole human contributor, originated the core research question and algorithm design, and takes full responsibility for all claims, proofs, and content in the final submission. Each tool served a different role, and no single tool was involved in all stages. The full pipeline is disclosed below in chronological order.

### Tools Used

**1. DeepSeek (Ideation and Initial Formalization)**

DeepSeek was used in an interactive prompting session to formalize the author's informal algorithm description into structured theorem statements, initial proof sketches, and a complexity analysis framework. The author provided: (a) the core algorithm specification — an exponential doubling search followed by binary refinement to find the optimal tree count in a Random Forest; (b) the research question — whether this method could provably outperform grid search; and (c) the practical motivation — edge computing deployment constraints. DeepSeek generated: (a) the initial structuring of seven theorem statements from the author's informal reasoning; (b) initial proof sketches for those theorems; and (c) a complexity analysis framework comparing the proposed method against grid search, random search, and Bayesian optimisation. The full transcript of this ideation session is available at:

> DeepSeek. (2025). Two-Phase Search ideation session [Generative AI chat]. DeepSeek. https://chat.deepseek.com/share/v6lr0kn1myyg05b2b0

**2. UCI ZotGPT Chat (Independent Theorem Validation)**

The theoretical framework originally developed through the DeepSeek session (see above) was subsequently validated through an interactive, human-guided prompting session using UCI ZotGPT Chat, a UCI-hosted AI assistant provided by the University of California, Irvine.

The author engaged in an iterative dialogue in which:

- The author submitted the previously generated theorem statements, proof sketches, and algorithmic structures as inputs for independent cross-validation, using a separate AI system to check for logical consistency, mathematical soundness, and completeness.
- UCI ZotGPT Chat independently reviewed and responded to the submitted theorems by evaluating the formal structure of each proof, identifying ambiguities or gaps in reasoning, assessing the validity of complexity and convergence claims, and confirming the internal consistency of the bidirectional search framework as a whole.
- The author used the model's responses to confirm, refine, or flag areas requiring further human review, including verification of edge cases, boundary conditions, and the correctness of probabilistic bounds.

Final verification and responsibility for all mathematical claims, proofs, and empirical interpretations rests solely with the author. UCI ZotGPT Chat served strictly as a secondary validation and cross-referencing tool, similar in role to a proof-checking assistant or peer review aid, and did not independently generate new theorems, assume authorship, or guarantee mathematical correctness.

> UCI ZotGPT Chat. (2026). Validation session for Two-Phase Search theorems [Generative AI chat]. University of California, Irvine. Session conducted April 2026. Available at: [INSERT CHAT LOG URL OR SESSION REFERENCE IF APPLICABLE]

**3. Claude, Anthropic (Revision, Restructuring, and Paper Development)**

Claude (Anthropic) was used across multiple sessions via Claude.ai for substantial revision and extension of the paper. Work performed with Claude's assistance includes:

- Pre-publication review identifying and correcting four substantive errors in the DeepSeek-generated formalizations: replacement of an invalid Gaussian noise assumption with a bounded noise model; repair of an unjustified comparison-model claim in the information-theoretic lower bound proof (Theorem 7) via an adversarial construction; resolution of a discrete-vs-continuous domain inconsistency across multiple proofs; and removal of undefined terminology
- Restructuring the paper from a proof collection into a complete submission draft, including writing the abstract, related work section (Section 3), practical motivation (Section 2), and discussion (Section 8)
- Addition of new content: the two-audience framing (Section 2.5), distributed coordination overhead argument (Section 2.3), warm-start implementation note (Section 5.3), and figure specifications for Figures 1–3
- Design of the full experimental framework (Tables 1–7, Appendix A, Figures 0–3) with projected values pending empirical validation
- Production of a research journal companion document cataloguing the intellectual lineage and design philosophy of the work
- Production of a prompt-by-prompt implementation guide for the experimental code

The current session transcript is available in the author's Claude.ai conversation history (April 24, 2026).

**4. Claude Code, Anthropic (Experimental Code Generation — Planned)**

Claude Code, Anthropic's command-line agentic coding tool, may be used in a subsequent session to generate the experimental Python code described in the implementation guide. This includes: the Two-Phase Search algorithm implementation, baseline method implementations (grid search, random search, Bayesian optimisation), experimental harness, statistical analysis scripts, and figure generation scripts. If used, Claude Code's role will be limited to code generation from the author's specifications; all experimental design decisions are documented in the paper (Section 7) and implementation guide prior to any code generation. This section will be updated with specific session details and outputs after the experimental phase is complete.

### Intellectual Contribution Breakdown

| Contribution | Source |
|---|---|
| Research question (why is 100 trees the default; is there a principled method to find the optimum) | Author |
| Algorithm design (exponential doubling → plateau detection → binary refinement) | Author (informal specification) |
| Bidirectional convergence test concept | Author |
| Edge computing as primary motivating application | Author (initial idea); Claude (literature grounding with Pham-Quoc 2024, Daghero 2022) |
| Initial theorem statements and proof sketches | DeepSeek (from author's informal specification) |
| Independent theorem validation and cross-checking | UCI ZotGPT Chat (from author's submitted theorems) |
| Noise model correction (Gaussian → bounded) | Claude (pre-publication review) |
| Theorem 7 adversarial construction repair | Claude (pre-publication review) |
| Discrete domain formalization | Claude (pre-publication review) |
| Two-audience framing and optimality guarantee argument | Author (concept); Claude (drafting) |
| Paper structure, related work, experimental design | Claude |
| Experimental code | Planned: Claude Code (from author's specifications) |
| All empirical results | Pending (author will run experiments) |
| Verification of all mathematical claims | Author (sole responsibility) |

### Process Summary

The research development followed a four-stage pipeline, each using a different tool for a different purpose:

1. **Ideation and formalization** (DeepSeek): The author's informal algorithm description was structured into formal theorem statements and proof sketches.
2. **Independent validation** (UCI ZotGPT Chat): The formalized theorems were submitted to a separate AI system for cross-validation of logical consistency, mathematical soundness, and completeness. This step served as an independent check analogous to a proof-reading assistant or preliminary peer review.
3. **Revision and paper development** (Claude, Anthropic): Substantive errors in the initial formalizations were identified and corrected, the paper was restructured into a complete submission draft, and new content was added. This stage produced the most significant changes to the theoretical framework, including the noise model correction, the Theorem 7 repair, and the discrete domain formalization.
4. **Experimental implementation** (Claude Code, planned): Code generation from documented specifications for empirical validation.

No single AI tool was involved in all stages. The use of multiple independent tools at different stages provided natural cross-validation: errors introduced or missed by one tool were caught by another. The author directed the overall research programme, made all scoping and design decisions, and is solely responsible for the correctness of the final work.

### Author Responsibility Statement

The author is solely responsible for the correctness of all theorems, proofs, and claims in this paper. The AI-generated formalizations were reviewed, cross-validated using an independent AI system, corrected where necessary (see Section 4 of the companion Research Journal for specific corrections), and verified by the author. The author takes full responsibility for any errors that remain.

### APA 7th Edition References

DeepSeek. (2025). *Two-Phase Search ideation session* [Generative AI chat]. DeepSeek. https://chat.deepseek.com/share/v6lr0kn1myyg05b2b0

UCI ZotGPT Chat. (2026). *Validation session for Two-Phase Search theorems* [Generative AI chat]. University of California, Irvine. [INSERT CHAT LOG URL OR SESSION REFERENCE IF APPLICABLE]

### Note on Journal Policies

Some venues restrict or prohibit AI-generated content in core theoretical sections. The author notes that: (a) the algorithm itself was conceived by the author prior to any AI interaction; (b) the theorem statements formalize the author's informal reasoning rather than representing novel AI-originated claims; (c) the initial formalizations were independently cross-validated using a separate AI system (UCI ZotGPT Chat) and subsequently corrected through a third AI-assisted review (Claude); (d) the author has verified all proofs and takes sole responsibility for their correctness; and (e) full interaction transcripts for all AI-assisted stages are available upon request. The author will comply with the specific AI-use policies of the target venue.

---

*This disclosure was itself drafted with the assistance of Claude (Anthropic) on April 24, 2026, based on the author's description of the research process.*
