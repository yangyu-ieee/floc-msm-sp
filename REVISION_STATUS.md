# FLOC-MSM / SIGPRO-D-26-02622 — revision status and evidence map

**Last reconciled:** 16 November 2026  
**Manuscript:** *Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective*  
**Journal:** Elsevier *Signal Processing*  
**Manuscript ID:** SIGPRO-D-26-02622

This file is a navigation/status record, not the manuscript, response letter, or proof of final portal submission.

## 1. Current status

The R1 clean main manuscript is documented as **23 printed pages**. A **94-page Editorial Manager assembly/preview** was inspected; its first page carries the manuscript ID and Editorial Manager footer, and page 3 names an attached supplementary item, Supplementary_Material_v73.pdf.

This establishes that an R1 assembly/preview artifact exists. It does **not** establish that the final portal submission step was completed, which response letter was uploaded, or that the named supplementary file can be independently inspected.

**Current disposition: selected public research/evidence audits are substantially complete; editorial finalization remains unverified.** Do not describe the response candidate as author-approved or the R1 as finally submitted based on Git alone.

## 2. Authoritative artifacts and version boundaries

| Artifact | Role | Authority / limitation |
|---|---|---|
| [Corrected v70 archive](revision_v70_20261010/README.md) | Frozen corrected code and selected archived results | Reproducibility snapshot, not the manuscript or complete submission package. Do not rewrite it during documentation cleanup. |
| [Reconciled response candidate](revision_audits/2026-11-14_R1_response_to_reviewers_reconciled_CANDIDATE.md) | Full R1.1–R1.5 and R3.1–R3.6 response candidate | Best available Git review starting point; not author-approved, not portal-verified, and not independently checked against the exact final supplement. |
| [Response alignment closeout](revision_audits/2026-11-14_R1_reviewer_response_alignment_closeout.md) | Response-to-clean-manuscript alignment audit | Controls the documented correction of stale response values and page locations; preserves unresolved editorial gates. |
| [Supplement/public gate audit](revision_audits/2026-11-14_R1_supplement_and_public_gate_audit.md) | Public numerical gate and supplement-source boundary | 28 selected clean-R1 core table cells checked; full S1–S18 supplement not independently verified. |
| [Audit index](revision_audits/README.md) | Audit navigation and overlap map | Use this to find topic-specific audits without treating every suggested future edit as an instruction to change the assembled R1. |

### Strict version separation

- Root-level scripts are the **historical pre-revision submission**. The affected original alpha=1.5 CMS sampler used cos(U) instead of cos((1-alpha)U); the affected original alpha=1.5 results and significance claims were withdrawn.
- revision_v70_20261010 is the corrected, versioned code/results snapshot.
- The 23-page clean R1 manuscript, the 94-page Editorial Manager assembly, the separately named Supplementary_Material_v73.pdf, and the response-letter source copies are **not all present in this Git repository**. Do not treat the code archive as a complete editorial submission package.

## 3. Verified core numerical record

The independent clean-R1-to-archive audit checked **28 mean/standard-deviation cells**:

- Table 1: 4 objectives × 4 label budgets = 16 cells, matched to the designated main_table_reconciliation_20261009 run.
- Table 3: 3 validation-selected robust-loss families × 4 budgets = 12 cells, matched to the archived validation-selection records. The selected variants are checked against validation accuracy, not held-out test accuracy.

The independent live-source arithmetic reported **28/28 matches**. The standalone Python checker audit_public_R1_core_table_gate.py is available, but its execution from a complete local clone was not claimed in the audit. This is a selected public-core check, **not** a full manuscript or Supplementary Materials audit.

### Corrected alpha=1.5 core Table 1

| Total labels | MSE mean accuracy | p-moment loss (p=1.2) | Difference |
|---:|---:|---:|---:|
| 50 | 93.6% | 94.1% | +0.44 pp |
| 100 | 94.5% | 96.0% | +1.52 pp |
| 200 | 95.7% | 97.0% | +1.24 pp |
| 500 | 97.0% | 98.5% | +1.44 pp |

The p-moment loss has the highest mean at 50 and 200 labels; L1 has the highest mean at 100 and 500. The observed p-moment-versus-MSE differences are descriptive, **not a statistically established five-seed advantage**. The smallest attainable two-sided exhaustive sign-flip value with five nonzero pairs is 0.0625. The sign-flip calculations are descriptive diagnostics unless the relevant exchangeability/symmetry assumption is justified; absence of p<0.05 is not evidence of equivalence.

The original affected alpha=1.5 results and their significance claims must remain withdrawn. Do not restore old values from the root scripts or from the earlier main_alpha15 archive in order to improve the apparent result.

## 4. Scientific interpretation and claim boundaries

1. **Theory:** Proposition 1 and Corollaries 1–2 are applications of established symmetric-alpha-stable moment conditions, with the stated bounded-Jacobian assumption where used. They are not a new stable-distribution theorem and do not establish a general convergence guarantee.
2. **Residual law:** Do not equate the injected alpha-stable noise with the trained residual distribution. Per-sequence normalization and a finite-parameter model bound the processed finite-input domain; the audits do not establish that learned residuals follow an exact non-degenerate alpha-stable law.
3. **Primary replication hierarchy:** the corrected core benchmark uses one pretrained checkpoint per objective and five matched downstream data-generation/fine-tuning seeds. Those five seeds do not estimate variance across independently pretrained encoders. Within a label budget and seed, downstream data are matched across objectives; test draws differ across label budgets, and the 500-label condition also uses a different fine-tuning epoch budget.
4. **Learnable p:** the parameter is included in the optimizer, but the raw objective has a unit/scale dependence that prevents interpreting unconstrained exponent optimization as a scale-free estimate of the noise-tail index. The archived p(t) trajectory is missing. Do not claim demonstrated adaptive tail-index learning.
5. **Gradient interpretation:** lower residual-score concentration does not imply smaller network gradients. The stagewise audit reports lower top-0.1% score share for p=1.2 than MSE in 20/20 seed–epoch snapshots but higher held-out raw parameter-gradient q0.9 in 19/20. These are descriptive diagnostics, not training-update distributions or a gradient-tail theorem.
6. **Masking:** the primary exact-count 50% token mask is not equivalent to 50% raw-sample concealment because Conv1D receptive fields overlap. Random-token versus contiguous-block results are 7/8 higher means plus one displayed tie, not eight wins. The separate 25%/75% endpoints use Bernoulli masks; the scratch control also varies between runs, so those margins do not isolate a pure mask-ratio effect.
7. **Multi-alpha:** the corrected sweep is descriptive rather than a strict common-random-number comparison of losses because pretraining random streams are not reset between objective arms. The alpha=1 targeted Cauchy run is separately valid under its exact alpha=1 configuration; its general-alpha runner must not be reused for non-Cauchy values.
8. **Student-t:** this is a separate stress test using Bernoulli masks and distribution-specific streams, not a one-factor Student-t-versus-S-alpha-S comparison. The small five-seed contrasts remain descriptive.
9. **External data:** UCR results support heterogeneous rankings rather than a universal loss selector. NOISEX's 5 dB recorded-waveform mixing is not sensor- or sample-rate-calibrated physical noise. CWRU is a load-held-out stress diagnostic with overlapping windows and near-four-class-chance absolute accuracy; its historical loss ranking is additionally confounded by non-reset model/optimizer RNG streams between objectives.
10. **Prior art:** STAR-MAE (2026) is an adjacent masked-audio method with distribution-aware loss reweighting. It limits broad novelty claims but is not a directly comparable baseline in the current raw one-dimensional fixed-backbone experiment.

The detailed source checks and exact qualifications are indexed in revision_audits/README.md. No current audit justifies another GPU sweep by default.

## 5. Missing evidence and blocked checks

The public v70 tree intentionally omits several inputs required for full source-level auditing:

- manuscript/sp_main.tex;
- manuscript/sp_supplementary.tex or supplement/sp_supplementary.tex;
- the UCR and NOISEX supplementary table fragments;
- SOURCE_PROVENANCE.json;
- results/diagnostics/epoch20_score_samples.npz;
- the exact historical source/hash for the dedicated exact-count random-token masking sensitivity arm.

Consequently, code/audit_supplement_tables.py cannot initialize against the public v70 tree, and code/audit_release.py cannot run as a whole-release gate there. This is a declared public-package dependency limitation, not evidence that the clean-R1 Table 1 or Table 3 values are wrong. The public-core checker is deliberately narrower.

Still unverified:

- the actual bytes and final contents of Supplementary_Material_v73.pdf, including all S1–S18 rows, figures, captions and locators;
- the canonical response letter selected by the author or attached in Editorial Manager;
- whether the final portal submission action was completed;
- whether any additional editor/Reviewer 2 instruction exists. The accessible author-side response copies contain Reviewer 1 and Reviewer 3 sections; absence of Reviewer 2 in those copies is not proof that no such instruction exists.

Do not fill these gaps by inferring from filenames, the 94-page assembly, or a response candidate.

## 6. Exact-duplicate inventory and preservation decision

A Git Blob SHA audit of the 233 files under revision_v70_20261010/ found **20 exact-content duplicate groups**:

- 2 source-copy groups: the main corrected runner is byte-identical to the file named provenance/mask_strategy_contiguous_block_v2_runner.py; the corrected UCR runner is byte-identical to its provenance-path copy.
- 3 diagnostic copies: a config, a results JSON, and run metadata are duplicated between the parent diagnostics directory and local_geometry/.
- 14 results.json / results_partial.json pairs are byte-identical.
- 1 three-path trajectory group: the p=1 FLOM trajectory and L1 trajectory copies are byte-identical, consistent with the mathematical identity of the p=1 objective and L1.

These are archive redundancies, not 20 independent experiments or 20 scientific contradictions. Some are useful provenance aliases; the results_partial.json names are misleading where the bytes exactly equal the corresponding complete results.json.

**Preservation decision:** do not delete or rename any of these files in v70. The archive is frozen and its SHA256 manifest covers 232 non-manifest files. Deleting aliases would invalidate that manifest and may break old paths. If a future v71 release is created, canonicalize the aliases there, preserve a mapping from old paths to canonical files, and regenerate the manifest.

## 7. Repository maintenance findings

- The root README's old Quick Start incorrectly prefixed root-level historical scripts with code/; those paths do not exist. The root scripts also contain machine-specific paths and/or obsolete experimental logic, so they should not be presented as a reproducible entrypoint.
- The 18 Markdown audit documents are topic-specific evidence records. Several deliberately repeat the same caveats in a broader summary and a later focused audit. They are retained; the new audit index marks overview/detail/supersession relationships instead of deleting source evidence.
- The codex/revision-v70-20261010-public-release branch and main are diverged (at audit time, main was 76 commits ahead and one commit behind that branch). Do not force-sync or merge the release branch as part of documentation cleanup.

## 8. Next actions — ordered by value

1. Obtain the exact final Supplementary_Material_v73.pdf (or later submitted version) and verify S1–S18 plus every response locator against it.
2. Confirm from the author/Editorial Manager which response letter is canonical and whether the final submission step is complete; check whether any Reviewer 2/editor instruction is outstanding.
3. Preserve the corrected Table 1/3 values and the withdrawn original-sampler results boundary in every author-controlled response copy.
4. If a new editable manuscript revision is requested, prioritize concise protocol qualifications from the focused audits; do not append every suggested paragraph or start new GPU experiments by default.
5. Keep v70 immutable. Consider exact-duplicate cleanup only in a separately versioned future release.
