# 26SP R1 targeted follow-up: UCR evidence units and alpha=1 moment window

Date: 2026-11-13
Status: post-R1 prospective quality audit, **not** an amendment of the submitted/assembled manuscript.
Repository baseline: frozen `revision_v70_20261010/`.
Inspected:
- `code/ucr_all_losses_corrected.py`
- `results/ucr_corrected/config.json` and `results/ucr_corrected/dataset_diagnostics.json`
- `code/alpha1_training_dynamics/targeted_alpha10_fullseed.py`
- `results/cauchy_level/config.json`
- `SIGPRO-D-26-02622_R1.pdf` (assembled R1 manuscript, primary text)
No experiment rerun, no manuscript amendment, and no claim of a newly proven neural-network convergence result.

## A. UCR effective experimental units

In the UCR runner, for each named dataset:
1. The *official* train/test splits are loaded separately.
2. CMS corruption is drawn **once** for the complete train split (seed = `20261007+20000+2*dataset_index`) and **once** for the complete test split (seed offset +1). The same corrupted matrices `X_train_n, X_test_n` are used for every loss and all downstream seeds.
3. An unlabeled pretraining pool `X_pt` is constructed **only from the official training split**, by `np.tile(X_train_n, (n_aug,1))` with `n_aug=max(1, 3000//len(X_train_n))`, adding independently drawn Gaussian jitter at SD 0.02 to each row. Under max-abs mode it is normalized per augmented sequence. Thus `X_pt` is **not 3000 independently observed new signals**, and the originally drawn CMS corruption in the underlying training rows is reused across Gaussian-jitter copies.
4. **One pre-trained encoder per loss and dataset** is fitted on that pretraining pool. Five downstream seeds generate separate per-class few-shot selections and random training initializations, but **not** independent CMS-corruption or pretraining replications.
5. The reported accuracy SD over five downstream seeds therefore concerns conditional few-shot selection/fine-tuning variation on a **fixed** corrupted test set. It does not estimate variability across independent corruption realizations or independent SSL pretraining runs. The benchmark still supports descriptive external-task comparisons, and this is **not** test leakage.

Illustrative *source training record* counts and augmented pretraining rows implied by the archived dataset diagnostics and the runner's rule:

| UCR dataset | Official train records | Gaussian-jitter copies per source record | Pretraining rows |
|---|---:|---:|---:|
| CBF | 30 | 100 | 3000 |
| GunPoint | 50 | 60 | 3000 |
| ItalyPowerDemand | 67 | 44 | 2948 |
| ECG200 | 100 | 30 | 3000 |
| FaceAll | 560 | 5 | 2800 |
| FordA | 3601 | 1 | 3601 |

Copies are not identical after jitter; this is about **independent underlying source sequences and independent CMS noise realizations**, not uniqueness of final array values.

Suggested methods text for a future editable revision:
> For each UCR dataset, the original training and test splits are independently contaminated once with a fixed-seed CMS noise realization. The same contaminated test set is used across all losses and downstream seeds. The unlabeled pretraining pool is assembled by repeating contaminated training examples with additional Gaussian-jitter augmentation, without using test examples. We train one encoder per loss and dataset; five fine-tuning seeds vary the per-class labeled subset and downstream optimization, not the corruption realization or pretraining initialization. Accordingly, reported standard deviations quantify downstream variation under this fixed dataset/noise protocol rather than unconditional robustness across noise draws.

Do not call the UCR protocol a "multiple independently corrupted noises" evaluation without reruns.

## B. Distinguish population-loss and score-moment windows under SαS hypothesis

For a nondegenerate symmetric α-stable **residual** R, `0<alpha<2`, with p in (1,2):
- finite population p-moment objective iff `p<alpha`;
- finite residual-score second moment iff `2*(p-1)<alpha`, or `p<1+alpha/2`.

These are two different moment criteria. For any `0<alpha<2`, there is an interval where **expected p-loss is infinite but score second moment is finite**:

`max(1,alpha) < p < 1+alpha/2` (also `p=alpha` may be included when `alpha>1`: expected p-loss is still infinite there).

At nominal `alpha=1` and `p=1.2`:
- `E|R|^1.2 = infinity`;
- `E|psi_1.2(R)|^2 < infinity` because `2(1.2-1)=0.4<1`.

At `alpha=1.5`, `p=1.2`, both idealized criteria are satisfied. The above is a **direct implication of the classical stable moment rule**, not a new theorem. It is relevant because the alpha=1 targeted empirical comparison can perform well even though, under the *hypothetical unnormalized alpha-stable residual model*, its expected p-loss would diverge. **However, in the manuscript's actual max-abs-normalized finite-network protocol, the learned residual is bounded at each fixed parameter value. No claim that its expected p-loss diverges or that this abstract decoupling explains the empirical alpha=1 result is justified.**

Suggested theoretical clarification if future editing is permitted:
> The p-moment objective and the residual-score variance impose different moment requirements. Under an idealized SαS residual, p=1.2 at α=1 would yield infinite expected p-loss but finite second moment of its residual score. This observation follows directly from the classical stable moment criterion and is not a convergence theorem. Our max-abs-normalized learned residuals do not satisfy that SαS assumption; the observation therefore clarifies the scope of the theoretical comparison rather than predicting the reported Cauchy-level accuracies.

## C. Cross-check of the Cauchy-level replication hierarchy

The `targeted_alpha10_fullseed.py` script re-generates data and retrains encoders separately for ten loss-specific runs, explicitly using `LOSS_SEED_OFFSET`; accordingly those runs are **not matched pairs in the sense of common random streams**. The R1 corrective Welch comparisons are appropriate to this experiment's independent-stream description. Its clipping and masking choices also differ from the main fixed-count/encoder-only clipped benchmark (Bernoulli token masks and clip over full pretraining parameter list in this targeted script); the two studies should not be presented as a directly matched comparison across α.

## D. Independent Cauchy Welch/Holm replication: valid R1 p-values, obsolete paired statistics in the frozen summary

This subsequent audit (2026-11-13) verified the strongest numerical inferential claim of the present clean R1 §4.5 against all 10 per-loss pipeline seeds in `results/cauchy_level/results.json`. The Cauchy generator at **exactly** α=1 produces `tan(U)` despite the historical general-α CMS formula, so the frozen α=1 results are not invalidated by that historical expression.

**Crucial provenance distinction:** `code/alpha1_training_dynamics/targeted_alpha10_fullseed.py` gives each loss distinct seed offsets for **both** new pretraining data/checkpoints and new downstream training/test realizations. The seed index is only a grouping label; the loss experiments are not common-random-number **pairs**. Nonetheless the original runner's `summarize()` calls `paired()`, which uses `scipy.stats.ttest_rel` to write `results/cauchy_level/summary.json`'s `best_floc_vs_mse` p-values. **Those historic paired p-values are inappropriate for interpreting these loss-specific independent-stream experiments.**

The separate **frozen** `code/audit_cauchy_unpaired_welch.py` correctly recomputes `scipy.stats.ttest_ind(..., equal_var=False)` from the raw archived accuracies and applies Holm over the four budget-wise p=1.2-versus-MSE comparisons. The clean submitted R1 refers explicitly to this corrected Welch inference, not to the old paired tests.

An independent fresh SciPy computation of each contrast from the ten archived accuracy values (recovered also as correct counts out of 500) produced:

| Total labeled | p=1.2 minus MSE mean (pp) | Welch two-sided raw p | Holm-adjusted p | Unadjusted 95% Welch CI for mean difference (pp) |
| ---: | ---: | ---: | ---: | --- |
| 50 | +14.92 | 1.0909×10⁻⁵ | 2.1818×10⁻⁵ | [9.94, 19.90] |
| 100 | +12.64 | 4.4102×10⁻⁷ | 1.3231×10⁻⁶ | [9.40, 15.88] |
| 200 | +13.84 | 2.6161×10⁻⁸ | 1.0465×10⁻⁷ | [10.87, 16.81] |
| 500 | +5.72 | 2.4720×10⁻⁵ | 2.4720×10⁻⁵ | [3.60, 7.84] |

**Every corrected Holm p < 0.001, as the clean R1 asserts.** The listed 95% CIs are *unadjusted* and are NOT simultaneous/multiplicity-corrected intervals. The comparison describes the joint synthetic data-generator-and-training pipelines under independent loss-specific random streams; it does not hold the same 500 test observations fixed across losses.

Additional independent Welch comparisons of p=1.2 against L1 and Charbonnier at all four budgets yield unadjusted p-values > 0.05, so these ten-run data do not establish superiority over those robust alternatives. Failure to reject is not an equivalence demonstration.

**Operational fix:** never use `results/cauchy_level/summary.json`'s old `paired()` p-values or regenerate that summary as though they were valid matched comparisons. Leave the frozen file intact. Cite the corrected four-comparison audit or use `revision_audits/audit_cauchy_welch_holm_and_legacy_summary.py` with SciPy to recompute the correct numbers.

This is a provenance/statistical-source warning, **not a numerical error in the current R1 Holm-adjusted claim**. The same ten SSL runs recur across budgets, so four budget effects are related; Holm controls familywise error under arbitrary dependence when the marginal p-values are valid, but the results do not independently replicate across four unrelated datasets.

## Disposition

- No new numerical error asserted.
- Recommend a one-paragraph precision upgrade to UCR protocol in next editable source.
- Recommend a two-sentence clarification of distinct moment windows only if it helps readers; avoid presenting it as a new theorem or causal explanation.
- Preserve R1 and frozen v70 evidence unchanged.
