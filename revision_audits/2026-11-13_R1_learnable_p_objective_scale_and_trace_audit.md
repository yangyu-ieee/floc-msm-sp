# 26SP R1: learnable-p is implemented, but raw objective induces unit-dependent p gradients

Date: 2026-11-13. Scope: post-submission audit of the **single** archived learnable-p pretraining run. **Do not modify the frozen v70 tree, submitted R1, or archived accuracy values.**

## Evidence checked

- Frozen `revision_v70_20261010/code/floc_msm_main_corrected.py`; source SHA-256 `880fbd11139d81d5abe462cd5190a28c973dee1c54e7932d7f629a9c7b541104`, matching `results/main_alpha15/config.json` and `MANIFEST_SHA256.txt`.
- `revision_v70_20261010/results/main_alpha15/{results.json,config.json,pretraining_trajectory_floc_learnable.json,stdout.txt,run_metadata.json}`.
- The manuscript references this exploratory run in Section 4.5 and Note S11 without treating it as the main confirmed finding.

## 1. No absent-gradient bug: p really is in the optimizer

The source creates an unconstrained trainable scalar `p_raw` and appends it to the Adam parameter group alongside the encoder, decoder and mask token.

The objective exponent is parameterized by

`p(u)=0.5+1.5*sigmoid(u)`, initialized with `u=log((1.5-0.5)/(2.0-1.5))=log(2)`, giving `p(u_0)=1.5` at the initial scalar `u_0=log(2)` (whereas `p(0)=1.25`).

Hence `p` remains strictly in **(0.5, 2)** and receives backpropagated gradients from `mean(abs(r)**p)`; the code imposes **no** separate tail-index constraint `p<alpha`, no validation criterion for updating p, and no explicit regularizer in p. The encoder-only gradient clipping at norm 1.0 does **not** clip the scalar exponent's gradient. This is a correctly implemented joint scalar-parameter training loop for the stated raw objective, not a failure to register p.

Important caveat: `p<1` is allowed, where the derivative with respect to a residual has singular magnitude as `r->0` in the formal unregularized formula. This *theoretical possibility* does not establish an observed NaN/gradient bug in the saved run, which completed 20 epochs.

## 2. Analytical issue: the reconstruction loss alone cannot identify a scale-free optimal exponent

For any nonzero residual coordinate,

`L(r,p)=|r|^p,` with `dL/dp = |r|^p * log(|r|).`

Consequently:

- For `0<|r|<1`, increasing p decreases the raw numeric loss, and gradient descent pushes p upward.
- For `|r|>1`, decreasing p decreases the raw numeric loss.
- For `|r|=1`, the p gradient is zero. At exactly zero residual, the loss itself is zero for positive p.

A constant rescaling of a signal and its residual, `r -> c*r`, changes the derivative to
`d|c*r|^p/dp = |c*r|^p[log c + log|r|]`.
For example, a residual of magnitude 0.5 favors increasing p; the same residual expressed in units that multiply it by four has magnitude 2 and favors decreasing p.

This is a **units/scale degeneracy** in direct minimization over p, independent of any claimed residual α-stable law. The optimal exponent of the unnormalized joint objective is not automatically a robust-tail selector. Altering p changes the numerical definition and scale of the optimized loss itself, so training-loss reductions cannot independently identify reconstruction improvement.

The max-abs-normalized data and previously archived held-out residual q99 values below 1 for *other* model checkpoints make this mechanism worth highlighting, but **we do not have the learnable-p run's own residual histogram**. One must not infer that every learnable-p residual was below 1, or that all its p-gradient steps were positive.

The allowed p range extends above the nominal `alpha=1.5` and below one. Even under the *idealized*, unnormalized α-stable residual hypothesis, the finite p-moment condition `p<alpha` is not enforced. In the actual max-abs-normalized finite-network setting, the learned residual is not presumed α-stable, so violation of a nominal idealized moment window is not by itself an experimental error.

## 3. The required p(t) reproducibility artifact is missing

The code collects an epoch-level `p_history`, returns it as `weights['_p_history']`, and then writes a NumPy artifact `p_trajectory_floc_learnable.npy` inside its run `LOG_DIR`.

However, the frozen `revision_v70_20261010/results/main_alpha15/` directory **does not contain** that `.npy` file, and `MANIFEST_SHA256.txt` does not list any `p_trajectory_` entry. The frozen stdout contains only the aggregate classification rows, not the console's earlier `p: first -> final` diagnostic.

The saved `pretraining_trajectory_floc_learnable.json` contains **only mean objective values**, with the first epoch at **0.1034750948** and the twentieth at **0.0484054978** (−53.22%). Since p and the encoder/decoder are co-optimized, this curve cannot determine either the learned final exponent or the share of the improvement due to each component.

**Do not reconstruct or invent final p from the train-loss trajectory.** The historical experiment may have saved its full p trace outside the released archive, but that trace is not presently verifiable from the frozen GitHub snapshot. If the original file is later retrieved, place a hash-verified copy in a *new audit/provenance location* with explicit author approval; do not silently mutate the frozen v70.

## 4. Recomputed downstream evidence — one SSL checkpoint only

The learnable-p loss was SSL-pretrained **once** for this experiment, and the saved main table then reports five matched downstream fine-tuning seeds **conditional on that one checkpoint**.

| Total labels | Learnable p, five-seed mean | Fixed p=1.2 mean | Learnable minus fixed (pp) |
| ---: | ---: | ---: | ---: |
| 50 | **92.16%** | 94.40% | **−2.24** |
| 100 | **95.36%** | 95.92% | **−0.56** |
| 200 | **95.96%** | 96.68% | **−0.72** |
| 500 | **98.00%** | 98.12% | **−0.12** |

All means and paired integer correct-count differences were independently reconciled from `results.json`; differences are small at higher budgets. In contrast to comparisons with fixed p=1.2, learnable p does exceed MSE in mean at budgets **100 (+0.40 pp), 200 (+0.12 pp), and 500 (+0.68 pp)**, while falling below MSE at 50 (−2.12 pp). Therefore the defensible claim is **not** that adaptive p universally fails relative to every fixed loss. It is that this single, unregularized training protocol does not demonstrate a reliable advantage over a well-chosen fixed p.

No repeat-pretraining variance or inference for adaptive exponent estimation is available.

## 5. Recommended treatment for a future editable response

The current article already relegates the run to a supplemental exploratory note. Do not expand the claim or launch a new adaptive-loss research line in this revision.

Suggested concise English wording, if the reviewer asks:

> In our exploratory learnable-exponent trial, the exponent was optimized jointly with the encoder and decoder through p=0.5+1.5 sigmoid(u), initialized at p=1.5. Directly minimizing the raw objective |r|^p is intrinsically scale-dependent: its p derivative is |r|^p log|r|, so increasing p reduces the loss for residual magnitudes below one, whereas the opposite holds above one. This single-checkpoint trial therefore cannot be interpreted as learning a tail-optimal exponent, and its mean downstream accuracies do not surpass the fixed p=1.2 control at any tested label budget. The archived pretraining-loss curve alone does not identify the final exponent, because its per-epoch p trajectory was not included in the frozen release.

One scientifically precise alternative future design would choose p using an **independent, scale-calibrated validation objective** or an explicitly standardized residual criterion, with repeated SSL pretraining. **This is a future-work boundary, not a request for new R1 experiments.**

## 5A. New theory-scope audit: adaptive-exponent gradient needs a stricter moment window

This result is a direct mathematical consequence of classical symmetric α-stable power-law tails, **not** a new theorem of our paper. It is a boundary check between R1 Proposition 1/Corollary 2 and the exploratory learnable-p experiment; the former holds p **fixed** and differentiates the reconstruction residual with respect to **network parameters**, whereas the latter differentiates with respect to the **exponent parameter**.

Take one non-degenerate symmetric SαS residual `R`, with `0<alpha<2`, and let `p=p(u)>0`, with finite scalar `u` and `0<|p'(u)|<infinity` (true for the interior of the experiment's sigmoid parameterization).

For the raw loss `ell(u,R)=|R|^{p(u)}`,

`d ell / du = p'(u) |R|^{p(u)} log|R|`, for `R!=0`.

For any `q>0`, since `p'(u)` is a fixed finite nonzero factor and a symmetric α-stable density has tails asymptotic to a positive constant times `|r|^(-alpha-1)`, the integrability at infinity is equivalent to

`integral_1^infinity r^{p*q-alpha-1} (log r)^q dr < infinity`.

This integral converges **if and only if `q*p < alpha`**. At equality, `r^(-1) (log r)^q` diverges. Near zero, the SαS density is bounded and `r^{pq}|log r|^q` is integrable for `pq>0`, so it adds no extra restriction. Therefore

`E |d ell/du|^q < infinity  <=>  q*p(u) < alpha`.

Compare the fixed-p **residual score** from R1 Proposition 1:

`E |d ell/dR|^q < infinity  <=>  q*(p-1) < alpha`, for `p>1`.

These conditions are distinct. A fixed-p neural-network **parameter gradient** also requires the finite-N, bounded-Jacobian assumptions of R1 Corollary 2 and receives only a **sufficient** moment condition because Jacobian cancellation or vanishing can preserve moments even when residual-score moments diverge.

### Explicit boundary cases

| Hypothetical SαS tail alpha | p | Fixed-p score q=2: 2(p−1)<alpha | Adaptive exponent q=1: p<alpha | Adaptive exponent q=2: 2p<alpha |
| ---: | ---: | --- | --- | --- |
| 1.5 | 1.5 | **finite** (1<1.5) | **infinite** at threshold (1.5=1.5) | **infinite** (3>1.5) |
| 1.5 | 1.2 | **finite** (0.4<1.5) | **finite** (1.2<1.5) | **infinite** (2.4>1.5) |
| 1.0 | 1.2 | **finite** (0.4<1.0) | **infinite** (1.2>1.0) | **infinite** (2.4>1.0) |

Thus at the actual nominal `alpha=1.5` and initial learnable `p=1.5`, the idealized exponent-gradient **first absolute moment** already diverges, while the **fixed-p residual-score second moment** remains finite. No inference of actual empirical instability follows from this hypothetical example.

### Why this does not invalidate the submitted study

- The *actual* input sequence is max-abs-normalized: `||y||_infinity < 1` whenever the unnormalized sample values are finite. A fixed finite-parameter, continuous encoder/decoder applied to the bounded input domain and finitely many token-mask patterns has **bounded** outputs and hence bounded residual coordinates. Thus the actual fixed-model residual cannot be a non-degenerate SαS variable, and its positive moments are finite. Finite trained weights and usual fixed finite-parameter differentiation also do not permit concluding any population gradient divergence from the idealized stable law.
- The submitted R1 Proposition 1 / Corollary 2 already carefully states the **fixed-p** idealized-residual scope; it **does not claim** that an adaptive-p exponent gradient obeys `q(p−1)<alpha`.
- The scale dependence from §2 is an *additional* reason why unregularized exponent minimization does not identify a tail-optimal exponent: even when moments are finite, `log|R|` flips the descent direction across unit magnitude.
- The archived `p_trajectory_floc_learnable.npy` is missing; a theoretical condition must **not** be used to invent a learned exponent or an empirical divergence event.

**Verdict:** no correction required for R1's current fixed-p theorem. If discussing adaptive-p future work, explicitly distinguish exponent-gradient (`q*p<alpha`) from residual-score (`q*(p−1)<alpha`) conditions **under the idealized SαS assumption**. Do not promote this classical moment-tail deduction into an independent novelty or GPU experiment.

Literature provenance: J. P. Nolan, *Stable Distributions: Models for Heavy Tailed Data* (2020), stable-tail and moments section; accessible secondary verification: [Álvarez-Esteban et al., symmetric SαS tails, TEST (2023)](https://link.springer.com/article/10.1007/s11749-023-00894-7) and the [stabledist manual's tail-asymptotic formula](https://search.r-project.org/CRAN/refmans/stabledist/help/StableDistribution.html).

## 6. Audit file and disposition

The no-GPU companion script `revision_audits/audit_learnable_p_scale_and_provenance.py` performs source-hash and loss-gradient checks, requires that the missing p trajectory remain absent from the frozen snapshot, reconstructs all four learnable-vs-fixed mean contrasts, and enumerates finite-sample five-seed paired sign flips using integer test-correct counts.

**Assessment:** no implementation failure demonstrated; one **real provenance gap** for the exponent trajectory and one **mathematically intrinsic scale degeneracy** in the adaptive objective. These make any claim of learned noise-tail adaptation unsupportable from the released run. The fixed-p core evidence and submitted manuscript need not be altered. The audit script is newly authored and **has not yet run from a complete local repository clone**; archived JSON arithmetic and code conditions were checked separately via the connected repository.
