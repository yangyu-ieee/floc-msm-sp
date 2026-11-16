# R1 audit index — FLOC-MSM / SIGPRO-D-26-02622

This directory contains post-assembly, mostly read-only audits. The notes are evidence records, not a queue of paragraphs that must all be pasted into the assembled R1. The frozen code/results archive is [revision_v70_20261010](../revision_v70_20261010/README.md); the controlling status summary is [REVISION_STATUS.md](../REVISION_STATUS.md).

## 1. Editorial and source-of-truth gates

| File | Role / finding | Status |
|---|---|---|
| [Response candidate](2026-11-14_R1_response_to_reviewers_reconciled_CANDIDATE.md) | Full R1.1–R1.5 and R3.1–R3.6 response candidate, corrected against clean R1 numbers and printed page locations | Best Git review starting point; not author-approved |
| [Response alignment closeout](2026-11-14_R1_reviewer_response_alignment_closeout.md) | Identifies stale values in older drafts; documents candidate corrections, reviewer-quote coverage and remaining portal/supplement gates | Controlling response-alignment audit |
| [Supplement/public gate](2026-11-14_R1_supplement_and_public_gate_audit.md) | Separates 28/28 checked clean-R1 core cells from the unavailable full S1–S18 source package | Controlling public-core/supplement boundary |

## 2. Core benchmark, seeds and statistical interpretation

| File | Role / finding | Relationship |
|---|---|---|
| [Seed scope and inference](2026-11-13_R1_seed_scope_and_inference_audit.md) | Main five-seed hierarchy, regenerated test sets across budgets, exact sign-flip enumeration and its exchangeability boundary | Primary detailed inference audit |
| [Randomized-family task](2026-11-13_R1_family_randomized_task_and_signflip_audit.md) | Randomized-family task changes class semantics; tiny mean margins and five-run sign-flip resolution | Separate protocol; not a fixed-template replication |
| [Multi-alpha provenance](2026-11-13_R1_multialpha_provenance_and_rank_audit.md) | Multi-alpha source recovery, unpaired pretraining streams and ranking resolution | Descriptive sweep; not strict objective-isolated comparison |
| [Validation selection](2026-11-13_R1_validation_selection_fragility_audit.md) | Validation-only tuning provenance, ties/near-ties and cases where held-out test ranking differs | Distinct from fixed-parameter Table 1 |
| [Student-t fairness](2026-11-13_R1_student_t_fairness_and_signflip_audit.md) | Student-t stress-test generator, Bernoulli mask and five-seed contrast limitations | Separate noise-family stress test |
| [UCR and moment window](2026-11-13_R1_UCR_and_moment_window_audit.md) | UCR dataset-cluster estimands, equal-dataset versus test-size weighting, stable-moment window distinctions and Cauchy replication | Broader external-statistics/moment audit |

## 3. Model, loss and masking interpretation

| File | Role / finding | Relationship |
|---|---|---|
| [Learnable p objective](2026-11-13_R1_learnable_p_objective_scale_and_trace_audit.md) | Learnable p is in the optimizer, but objective-scale dependence and missing p(t) trajectory block a learned-tail-index claim | Separate adaptive-p diagnostic |
| [Stagewise score/gradient](2026-11-13_R1_stagewise_score_gradient_scale_audit.md) | Residual-score concentration does not order raw network-gradient norms; Huber and L1 scaling caveats | Detailed gradient diagnostic |
| [Preprocessing scale](2026-11-13_R1_preprocessing_relative_noise_scale_audit.md) | Global-RMS preprocessing changes effective impulsive-noise scale, so it is not a normalization-only ablation | Separate Table 2 interpretation |
| [Mask position and coverage](2026-11-13_R1_mask_position_and_coverage_audit.md) | Token-mask fraction differs from raw-sample concealment; endpoint runs and scratch-control variation limit mask-ratio attribution | Detailed masking geometry |
| [Local geometry](2026-11-13_R1_local_geometry_fixed_mask_and_slice_audit.md) | Fixed-mask local-loss geometry and slice-level diagnostics | Diagnostic only; not a new optimizer result |
| [Pretraining control and STAR-MAE](2026-11-13_R1_pretraining_control_and_STARMAE_prior_art.md) | Core objective arms share a pretraining realization; one checkpoint per objective; STAR-MAE is adjacent 2026 prior art | Prior-art boundary and pretraining-control detail |

## 4. External-data evidence and interpretation

| File | Role / finding | Relationship |
|---|---|---|
| [External evidence overview](2026-11-13_R1_external_evidence_audit.md) | NOISEX mixing, win-count versus mean ranking, CWRU load split/window dependence and post-normalization bounds | Earlier overview |
| [External evidence RNG extension](2026-11-13_R1_external_evidence_cwru_rng_and_noisex_scope.md) | Adds CWRU objective-RNG confounding and seedwise stability of NOISEX mean winners | Later focused extension; use this for expanded RNG/seed caveats |
| [Quality overview](2026-11-13_R1_quality_audit.md) | Cross-cutting summary of five-pair inference, masking geometry and STAR-MAE literature | Overview; detailed claims live in the focused notes above |

## 5. Companion audit scripts

The Python files in this directory are source-level/data-level checkers. Their presence does not imply that every script has been run from a complete local clone. Individual notes state which checks were independently recomputed from GitHub JSON, which scripts were executed, and which remain unexecuted.

Important narrow checker:
- [Public R1 core table gate](audit_public_R1_core_table_gate.py) checks the 28 selected Table 1/Table 3 core cells from the public archive. It does **not** verify the complete Supplementary Materials.

Do not use the public v70 code/audit_release.py or code/audit_supplement_tables.py as full-package gates without their missing private manuscript/supplementary inputs. See the [public-gate audit](2026-11-14_R1_supplement_and_public_gate_audit.md).

## 6. Overlap and preservation policy

- The quality audit is a cross-cutting overview; it intentionally overlaps with the focused seed, masking and prior-art audits.
- The external-evidence audit is the earlier overview; the CWRU/NOISEX extension adds RNG fairness and seedwise winner stability. Do not count them as independent datasets or experiments.
- The response-alignment closeout references and extends the supplement/public-gate audit; the former controls response-letter consistency, the latter controls the scope of public numerical/supplement verification.
- Suggested text in individual audits is prospective. Do not paste every suggestion into a submitted R1 or rebuttal. Preserve the frozen R1/v70 evidence and obtain author approval before making editorial changes.
- Do not delete overlapping audits merely to reduce file count. They retain separate source checks and chronology; use this index to avoid repeating the same audit.
