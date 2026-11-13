# 26SP R1: score-concentration floors, Huber's effective quadratic region, and stagewise raw gradients

Audit date: 2026-11-13. This note is prospective, not a modification of the frozen v70 evidence or submitted R1. All results below come from archived **held-out** diagnostics, with no new model training.

## 1. Checked provenance and R1 claims

Sources:
- `revision_v70_20261010/code/r1_stagewise_diagnostics.py` and `code/r1_actual_residual_diagnostics.py`.
- `revision_v70_20261010/results/diagnostics/stagewise/{results.json,stagewise_gradient_tail_summary.json}`.
- `revision_v70_20261010/results/diagnostics/actual_residuals/{results.json,residual_quantile_aggregation_audit.json}`.
- The unmarked R1 paper, Section 4.3 / Figure 2 (assembled PDF pages 78–81), correctly identifies separately trained models, raw unscaled gradients, loss scaling, encoder-only training clipping, Bernoulli diagnostic masking, and absence of global convergence claims.

The stagewise record has **100** held-out snapshot records: 5 losses × 5 model/data seeds × 4 epochs (1/5/10/20). Each record uses **128,000 scalar residuals** (1,000 held-out signals × 128 positions) and **8 held-out batches** for the gradient q90. The epoch-20 q90 is the average of five separately computed within-seed quantiles, not a quantile pooled over forty correlated batches.

## 2. New empirical audit: score share and raw gradient magnitudes point in different directions

For each epoch, independently reaggregated archived records confirm the following five-seed mean 90th-percentile **unclipped** norm over encoder, decoder **and** mask-token parameters. The ratio is the mean of q90 values for one objective divided by the MSE mean, not a paired mean of ratios.

| Epoch | MSE q90 | p=1.2 q90 | p=1.2/MSE q90 | p=1.2/MSE top-0.1%-score-share ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.043295 | 0.092473 | 2.136× | 0.125× |
| 5 | 0.034075 | 0.071813 | 2.107× | 0.114× |
| 10 | 0.033217 | 0.059669 | 1.796× | 0.099× |
| 20 | 0.032818 | 0.063028 | 1.921× | 0.087× |

Among the 20 matched seed–epoch comparisons, p=1.2 has a **smaller** squared-score top-0.1% share than MSE in **20/20**, but a **larger** held-out batch-gradient q90 in **19/20**. Huber's q90 is lower than MSE in **20/20**, but that is not evidence that its optimizer is more stable. These 20 observations reuse five seeds across four epochs and are *not* 20 independent replicates or a significance test.

At epoch 20:
- MSE q90 = **0.0328184668**;
- Huber q90 = **0.0161782038** (0.4930× MSE);
- p=1.2 q90 = **0.0630281742** (1.9205×);
- L1 q90 = **0.0693390281** (2.1128×);
- Charbonnier q90 = **0.0745019385** (2.2701×).

The assembled R1 manuscript already gives all five epoch-20 q90 values correctly. This audit does not challenge their transcription; it makes the cross-seed directional mismatch explicit.

## 3. Why the L1 top-score share is automatically 0.1%

The diagnostic score function is `psi_L1(r)=sign(r)`. Its squared magnitude is **one at every nonzero residual**. In a vector of 128,000 nonzero residuals, the top `ceil(0.001*128000)=128` coordinates contain exactly (128/128000=0.001), or **0.100%**, of the total squared score mass. This number is the *combinatorial lower bound* for an equal-sized top fraction; it is essentially predetermined by the L1 score shape, rather than an empirical sign that a trained encoder has learned a less impulsive residual distribution.

Every one of the archived **20 stagewise L1 snapshots** and **five final held-out L1 snapshots** reports exactly 0.001. Compare this to:
- `p=1.2`, final actual-residual top-share **0.27757%**;
- Charbonnier, **0.11495%** (near saturation at score magnitude 1);
- MSE, **3.18220%**;
- Huber, **2.52195%**.

Likewise, a bounded Charbonnier score with small epsilon often lies close to ±1 for residuals larger than epsilon=0.01, so a small normalized score-share is partly imposed by the loss geometry. **This metric describes the weighting rule and observed residual profile jointly; it is not independently evidence of network-level convergence or superior classification.**

## 4. Huber delta=1 largely remains in its quadratic branch on held-out normalized residuals

Archived Huber and MSE loss conventions are
`L_MSE(r)=r^2` and `L_Huber,delta=1(r)=0.5*r^2` for `|r|<=1`, transitioning to linear beyond the unit threshold.

All **100 stagewise snapshot records across losses** have residual `abs_q99<1` (range **0.75047–0.89149**). The **25 final actual-residual records** also have `abs_q99<1` (range **0.75014–0.82847**). Therefore at these trained checkpoints, approximately **99% or more** of measured held-out residual coordinates fall in the **quadratic part of the Huber=1 loss**, with only the extreme upper tail possibly triggering Huber's linear branch. These are distributional observations, not counts of training-time residuals.

Consequently, Huber's epoch-20 raw gradient q90 of 0.01618 vs MSE's 0.03282 (ratio **0.493**) is consistent with the factor-of-two difference in the objectives' small-residual derivatives, **but the observed gradient ratio is across distinct trained models and does not prove that this factor alone caused it**.

This matters for fair interpretation: Huber delta=1 in the core trial may operate largely as a rescaled quadratic loss after max-abs normalization. The independently validation-selected Huber delta=0.25 or 0.5 settings in Table 3 are different protocols, and their superiority/weakness cannot be inferred by mixing those tables.

## 5. A further diagnostic/protocol boundary: eval-mode dropout and no gradient clipping

Within `evaluate(...)`, the code explicitly calls `enc.eval()` and `dec.eval()` and computes the held-out parameter gradients *after* a no-grad residual pass, using identical masked input batches. Dropout is disabled for the held-out gradient evaluation. Gradients are recomputed for all encoder, decoder and learned mask-token parameters without clipping.

In contrast, during pretraining the encoder and decoder are in `train()` mode (dropout enabled), the optimizer uses Adam, and `torch.nn.utils.clip_grad_norm_(enc.parameters(),1.0)` clips encoder gradients only. Therefore the displayed q90 is a **dropout-off, unclipped, held-out derivative diagnostic** of checkpoints trained under clipping, *not* the distribution of actual training updates. Its q90 is over only eight batches per checkpoint and cannot estimate a population gradient-tail index.

## 6. Manuscript recommendation, preserving the current correct claims

The existing R1 Section 4.3 already warns against identifying a Jacobian mechanism or convergence guarantee. Do not rewrite its main quantitative summary or claim a new theory. If an editable R1 is requested, one compact addition would improve interpretability:

> The score-concentration statistic is partly determined by the loss derivative itself: for nonzero residuals the L1 squared-score share in the largest 0.1% of coordinates is exactly 0.1%, independently of the learned residual magnitudes. Likewise, the held-out residual 99th percentiles lie below the Huber threshold delta=1, so this Huber setting operates in the quadratic branch for approximately 99% of measured coordinates. Hence both normalized score-share and raw-gradient-norm comparisons depend strongly on loss scaling and threshold conventions; the stagewise gradients, evaluated with dropout disabled and without clipping, are not the gradients applied by the optimizer during stochastic training.

A minimal rebuttal response is:

> We agree that lower residual-score concentration does not imply smaller network gradients. Our archived five-seed stagewise diagnostics show p=1.2 has a lower top-0.1% squared-score share than MSE in 20/20 seed–epoch comparisons but a higher held-out raw parameter-gradient q90 in 19/20; these repeated snapshots are descriptive, and our paper already refrains from asserting convergence or gradient-stability superiority.

## 7. Disposition

**No code error or incorrect main-paper numerical statement detected.** This is an interpretation and comparability audit. It makes stronger the manuscript's negative transfer result and prevents overclaiming from L1's mechanically minimal score share or Huber's near-quadratic regime. Do not request additional GPU training by default.

Companion standard-library checker: `revision_audits/audit_stagewise_score_gradient_scale.py`. Archive arithmetic and code guards were independently checked using the GitHub connector; the newly committed script has not yet completed a full local checkout execution.
