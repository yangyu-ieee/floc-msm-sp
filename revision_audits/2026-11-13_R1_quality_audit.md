# 26SP R1: post-assembly audit of seed inference, masking geometry, and literature

Date: 2026-11-13
Status: **evidence audit and prospective editorial recommendations**, not an author-approved manuscript revision.
Scope: SIGPRO-D-26-02622R1, "Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective".
Code baseline: the **frozen** revision_v70_20261010 snapshot, especially
- code/floc_msm_main_corrected.py (Conv1D/token masking/main evaluation);
- results/main_table_reconciliation_20261009/results.json (all seed accuracies);
- revision_v70_20261010/README.md (protocol and non-GPU-replication caveats).
No GPU experiment was run, no archived results were modified, and this note does **not** assert submission-system status.

## A. Actual five-pair inference, beyond the discreteness bound

Enumeration of all 32 sign assignments applied to the five paired downstream-seed accuracy differences (FLOM-based p-moment p=1.2 minus MSE):

| Labels | Mean accuracy difference, percentage points | Positive pairs | Two-sided sign-flip p |
| --- | ---: | ---: | ---: |
| 50 | +0.44 | 3/5 | 0.5625 |
| 100 | +1.52 | 4/5 | 0.1875 |
| 200 | +1.24 | 4/5 | 0.1875 |
| 500 | +1.44 | 4/5 | 0.1875 |

All reproduce the source seed-level archive; see audit_main_seed_and_mask_geometry.py. The R1 manuscript correctly states the minimum possible two-sided p-value is 0.0625; reporting **actual** p-values in a next revision would be more informative. No comparison establishes conventional p<0.05 significance. A sign-flip enumeration is a test of paired differences under its sign-exchangeability/symmetry assumptions, not an inference over independently retrained pretraining encoders. Each loss has one pretrained checkpoint in this benchmark. Dependence between budget conditions also rules out interpreting them as independent replications.

Suggested sentence (Section 4.2, if further editing is appropriate):
> Enumerating all 32 sign flips of the five matched fine-tuning-seed differences gives two-sided p-values of 0.5625, 0.1875, 0.1875, and 0.1875 at 50, 100, 200, and 500 labels, respectively. These conditional, exploratory comparisons neither establish a downstream performance advantage nor measure variability across pretraining checkpoints.

## B. Exact masking geometry: nominal ratio is not raw-sample occlusion

The main tokenizer is Conv1D on a 128-sample sequence with kernel=8, stride=4, padding=2, yielding 32 overlapping tokens. By direct enumeration of convolution receptive fields:
- 4 edge raw-sample positions feed **one** token each.
- 124 raw positions feed **two** tokens each.
- No raw sample is uncovered by the tokenizer.

Define a raw coordinate as **fully concealed at the tokenizer output** if **every** token whose convolution receptive field contains it is replaced by the mask token. This is a structural visibility definition. It is **not** a measure of information-theoretic entropy, downstream difficulty, or causal relevance.

Exactly 16 of 32 randomly selected tokens masked:
- for a position covered by two tokens, P(both masked)=(16/32)(15/31)=15/62;
- for a position covered by one token, P(masked)=16/32=1/2;
- expected fully concealed coordinates = 124*(15/62)+4*(1/2)=**32/128 = 25.00%**.

A uniform contiguous 16-token block has 17 permissible starts. At 15 starts it fully conceals 60 raw coordinates, at the two edge starts 62. Thus the mean is (15*60+2*62)/17 = **1024/17 ≈ 60.2353/128 = 47.06%**.

For the separate Bernoulli(0.5) token-mask diagnostics, expected full concealment is 124*(1/2)^2+4*(1/2)=**33/128 = 25.78%**. These numbers concern possible direct dependence at the tokenizer, not reconstruction quality.

Consequence: comparison of exact-16 random-token and contiguous-block masking at the same token-level ratio **does not hold raw-sample concealment constant**. Their performances cannot be attributed purely to mask topology independent of effective information exposure. The current R1 manuscript already correctly notes overlap; adding the quantitative statement would make the limitation significantly clearer. No extra training is necessary for the calculation, but no performance causal attribution follows.

Suggested Section 4.4 paragraph (prospective):
> Owing to convolutional receptive-field overlap, equal token-mask ratios do not imply equal raw-sample concealment. Our kernel-8, stride-4, padding-2 tokenizer maps 128 samples to 32 tokens. Under exact 16-of-32 random-token masking, the expected number of raw positions covered exclusively by masked tokens is 32 (25.0%); for a contiguous 16-token block, it is 60.24 (47.1%) averaged across its 17 possible start positions. Hence the random-versus-block comparison jointly changes mask geometry and effective raw-sample exposure. Its accuracy contrast is descriptive and cannot isolate a topology-only effect.

Do **not** claim "25% of information is hidden"; raw visibility here is only a receptive-field support criterion.

## C. A directly relevant 2026 literature neighbor

Jiayu Xiong, Jing Wang, Wanlong Wang, Xiaosen Lyu, Jianlong Kwan, and Jun Xue,
"Masked autoencoders for spatio-temporal audio representations: Theory and optimization",
*Pattern Recognition*, vol. 175, article 113133, July 2026.
DOI: https://doi.org/10.1016/j.patcog.2026.113133
Publisher page: https://www.sciencedirect.com/science/article/pii/S0031320326000968
Public code: https://github.com/Jiayu-Xiong/STAR-MAE

The official publisher abstract describes Distribution-aware Loss Reweighting (DLR), a training-only module suppressing noisy reconstruction gradients while upweighting informative signals in audio masked autoencoders. This is nearer to the **robust reconstruction objective** motivation than general signal MAE citations and was not found in the 35-item reference list of the inspected R1 PDF.

Distinction:
- STAR-MAE combines audio spatiotemporal representation design/structured masking with learned distribution-aware reconstruction-loss reweighting.
- 26SP fixes a simple signal-MSM pipeline and compares established pointwise reconstruction penalties in controlled heavy-tailed/synthetic and external tests; it treats stable-residual moment arguments as conditional and explicitly tests limits of gradient interpretation.
- There is **no direct STAR-MAE vs. 26SP experimental comparison**; neither superiority over DLR nor novelty of the broad concept of robust masked reconstruction should be asserted.

Suggested Related Work sentence (prospective):
> Reconstruction weighting has also been studied directly in masked audio learning: Xiong et al. (2026) introduced a distribution-aware loss-reweighting module in STAR-MAE to suppress noisy reconstruction contributions. Their adaptive, audio-oriented approach differs from our fixed pointwise loss-family comparison under impulsive corruption. We therefore do not claim to introduce robustness-aware masked reconstruction in general, nor to outperform adaptive reweighting methods not evaluated here.

## Disposition and limits

1. **No numerical contradiction** found in R1 Table 1 after auditing the v70 archived seed-level results.
2. **One quantified architectural confound** is available for the mask-strategy paragraph with no GPU work.
3. **One new originality-boundary citation** is recommended for the next permissible manuscript revision.
4. **Do not alter** the frozen v70 reproducibility snapshot or the author/submission-system PDFs as a side effect of this audit.
5. Confirm author approval and portal submission status separately; the locally assembled R1 PDF alone does not establish either.

Independent checker: [audit_main_seed_and_mask_geometry.py](./audit_main_seed_and_mask_geometry.py).
