# 26SP R1: corrected multi-alpha source recovery, stochastic-control limits and Cauchy sampler boundary

Date: 2026-11-13. Status: retrospective, data-free audit of the frozen 2026-10-10 R1 evidence. **No change to the frozen `revision_v70_20261010` tree, the submitted R1, or experimental result JSON.**

## Source restoration is exact, although the historical runner has a missing relative path

The archived wrapper is `revision_v70_20261010/code/provenance/run_corrected_multialpha_sweep.py`. Its declared input was `revision_v70_20261010/baseline_submission_candidate_2026-06-30/code/floc_tsp_loss_compare_fixed.py`, which is **not** present in the released GitHub tree. Consequently, invoking this historic wrapper directly from the current public checkout would fail during `read_bytes()` before training.

The byte-identical historical file **is** present at repository-root `floc_tsp_loss_compare_fixed.py`. Independently calculated SHA-256 values from the live repository (not from filenames) reconcile *all four* provenance levels:

| Stage | SHA-256 | Archived expected hash matches? |
| --- | --- | --- |
| Root historical source | `ba1d0a36301e344bd47fabb3ab939cf63693913285e80d353476d9c4dd22af88` | Yes, `frozen_source_sha256` |
| Existing historic wrapper | `71fc2aff3bb4abe46d67d5c8be8ada3e826b5bd2881d2aa0f59e2d3e13669e05` | Yes, `runner_sha256` |
| Corrected `stable_noise.py` | `5a507b3c297957368c09b23223227a8b397640fab6ad2b97ceef920e856a5aa4` | Yes, `stable_noise_sha256` |
| Reconstructed corrected, in-memory executed source | `e73efeb31fa72d898e48ff7ee016e9245255930dce6d83291afb10adebadb31b` | Yes, `transformed_source_sha256` |

Reconstruction reproduces the archived wrapper's *exact* two substitutions to the old sampler expression and its three replacements of the alpha loops, plus one `from stable_noise import from_cms_variates` line. The executable-source hash matches archived `results/multialpha/corrected_sweep/run_metadata.json`. This **recovers code identity**; it does not imply we executed the costly GPU experiment again or measured new scientific results.

A **non-mutating, SHA-guarded, portable replacement entry** has been added under `revision_audits/verify_and_rerun_multialpha.py`:

- Default invocation `python revision_audits/verify_and_rerun_multialpha.py` uses only the Python standard library to verify original/runner/sampler/transformed source hashes, generator alpha propagation, eight recorded rank differences and the target Cauchy alpha domain. It performs **no retraining**.
- Optional `--execute --output-dir /new/folder` requires a CUDA-capable environment, executes only code with an exact archived transformed hash, and writes results in a **new** folder; it refuses to write inside frozen v70 or to any existing folder. **No rerun was performed as part of this audit**.
- Source hashes require byte-preserving checkout; Windows autocrlf conversion can alter bytes and must be disabled for exact hash verification.

## The alpha parameter is correctly applied to all signal stages

Original runner passes `alpha` to `gen_unlabeled(10000,128,alpha)`, to `eval_downstream(...,alpha=alpha)`, and within evaluation to both `gen_cls(n_train,128,alpha)` and `gen_cls(n_test,128,alpha)`. Thus the corrected sweep at alpha=1.5 and alpha=1.0 **does not** accidentally fix the downstream noise to alpha=1.5.

Student-t extension `revision_v70_20261010/code/floc_msm_student_t.py` analogously uses its CLI `nu` in the noise function for all three signal stages.

## Important fairness difference from the corrected primary core benchmark

The corrected primary core runner `code/floc_msm_main_corrected.py` restores the same PyTorch pretraining seed before each loss, and creates a fresh DataLoader with the same independent shuffle-generator seed. This improves within-configuration comparability.

In contrast, the original multi-alpha runner seeds NumPy/Torch **once** in its outer wrapper, then for each alpha creates one shared unlabelled dataset and one **shared stateful** shuffling DataLoader; the loop
`for loss_name in LOSSES: weights[loss_name]=pretrain_with_loss(pt_loader,loss_name,alpha,20)`
does **not** restart Torch or loader RNG before each loss. Initialization, mask/dropout streams and minibatch shuffle thus drift across objectives. The downstream loop resets the same seed for each loss/label-budget pair, so downstream training/test draws are paired **conditional on different, not seed-matched SSL checkpoints**.

Consequently, multi-alpha is a **descriptive regime map**, not a strict loss-isolated objective comparison or repeated-SSL statistical test. This caveat is additional to (not a replacement for) the already disclosed single-checkpoint-per-loss issue.

## Empirical ranking resolution, reconstructed from 5×500 test correctness counts

| α | Labels | Highest mean | Runner-up | Mean advantage (pp) | Additional correct classifications out of 2,500 |
| --- | ---: | --- | --- | ---: | ---: |
| 1.5 | 50 | Charbonnier | L1 | 0.88 | 22 |
| 1.5 | 100 | Charbonnier | p=1.2 | **0.08** | **2** |
| 1.5 | 200 | p=1.2 | L1 | 0.36 | 9 |
| 1.5 | 500 | p=1.5 | Charbonnier | **0.08** | **2** |
| 1.0 | 50 | Charbonnier | p=1.2 | 2.20 | 55 |
| 1.0 | 100 | p=1.5 | p=1.2 | 0.28 | 7 |
| 1.0 | 200 | Charbonnier | p=1.2 | 0.76 | 19 |
| 1.0 | 500 | L1 | Charbonnier | 0.36 | 9 |

All numbers were reconstructed from the live frozen `results/multialpha/corrected_sweep/results.json` arrays and checked against `phase_map_summary.json`. There are **two near-ties only two correct classifications apart**, and five of eight top-two gaps are at most 0.36 percentage points. The best loss changes by regime as described in unmarked R1 Section 4.5, but do **not** interpret these as a stable alpha-to-p selection rule or universal ranking.

A notable case is alpha=1.0, n=100 where p=1.5 has the highest mean, although p=1.5 is not below the raw alpha-stable index alpha=1. Under the *idealized unnormalized alpha-stable residual* assumption, neither the population p-moment condition nor the q=2 derivative-variance condition would hold for p=1.5. Here max-abs normalization changes the data distribution and the residual need not be stable, so there is **no logical contradiction**; it reinforces the distinction between idealized motivation and empirical loss selection.

## A separate alpha=1 targeted runner has a subtle conditional safety boundary

`revision_v70_20261010/code/alpha1_training_dynamics/targeted_alpha10_fullseed.py` still spells the original, flawed general-alpha CMS formula. However the archived `results/cauchy_level/config.json` **confirms alpha=1.0 and 10 seeds**, and at alpha=1 the exponent `(1-alpha)/alpha` is exactly zero, giving

`sin(U)*(cos(U)/W)^0 / cos(U)^1 = tan(U)`,

which equals the correct symmetric Cauchy draw. Thus the originally corrected alpha=1 targeted results are not invalidated by this syntax. **Do not reuse this runner with `--alpha` other than exactly `1.0`**, although its CLI currently permits arbitrary floats: that would silently reintroduce the wrong stable transform for non-Cauchy alpha.

For any future non-Cauchy target, use `stable_noise.from_cms_variates` with a newly documented source version and independent rerun, never silently reinterpret the existing archive. Leave the frozen runner unchanged for traceability.

## Recommended future paper edit (Section 4.5, no additional numerical work)

> The corrected multi-alpha sweep uses alpha-specific synthetic pretraining, labeled training and test corruption. Unlike the primary core comparison, the sweep does not restart the same pretraining initialization and minibatch-shuffle random streams for each loss. It therefore provides descriptive rankings conditional on separately trained encoders rather than a strictly common-random-number comparison of objectives. Some highest-mean margins are very small (0.08 percentage points at alpha=1.5 for 100 and 500 labels), so they do not identify an alpha-dependent optimal loss. The independent ten-run Cauchy experiment is distinct and is evaluated with Welch tests because the loss-specific runs are not paired.

**Disposition:** High priority to fix the published *reproduction entrypoint*, medium priority to clarify Supplementary Fig. S5 multi-alpha ranking precision. No reason to withdraw or revise frozen multi-alpha experiment numbers.