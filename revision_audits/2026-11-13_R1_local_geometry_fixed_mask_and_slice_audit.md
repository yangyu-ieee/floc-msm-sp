# 26SP R1: fixed-mask local-geometry audit — 60 rising slices, 12 checkpoints, no Hessian claim

Date: 2026-11-13. Scope: post-submission read-only audit and next-round manuscript guidance; do not rewrite historical v70 or the submitted R1.

## 1. Source/archived material

- `revision_v70_20261010/code/r1_loss_landscape.py`, the actual R3.1 matched-seed local landscape runner.
- `revision_v70_20261010/results/diagnostics/local_geometry/config.json`, `results.json`, and `local_landscape_endpoint_summary.json`.
- Unmarked assembled R1 Section 4.3 (PDF page 80), which reports 60 sampled slices, median two-endpoint increase 3.15%, and 2.10–4.89% range, without global/Hessian conclusions.

The source constructs four losses (MSE, L1, Charbonnier, p=1.2), **three** trained models per loss (three shared-seed clean-data/corruption protocols), and **five** isotropically sampled full-parameter directions per trained model. This gives **12 distinct trained checkpoints** and 60 directional slices, NOT 60 independent trainings or 60 independent empirical replicates.

Source controls for evaluation are stronger than the short manuscript sentence makes explicit:
- `enc.eval(); dec.eval()` switches off dropout at evaluation.
- The local objective uses the **first 128 held-out signals** from a 1,000-signal validation set, with identical data across directions for each seed.
- A fixed mask is generated from `79000+seed` **before** enumerating the five directions and five radii; the mask is reused across each direction and radius for a given seed. The same dataset and fixed-mask seed are applied across loss choices at that seed.
- For each direction, an independent seeded `torch.randn` vector is normalized by its global parameter Euclidean norm, and parameter values are replaced by `theta + radius * ||theta||_2 * direction`; the original fitted weights are restored after all directions.
- Radii are `[-0.10,-0.05,0,+0.05,+0.10]`, measured as a *global-parameter-L2-relative perturbation* rather than an infinitesimal or layer-normalized displacement.
- All loss values are recomputed on exactly the same fixed minibatch/held-out objective; there is no independently resampled stochastic mask or dropout at different radii. No hidden variable-randomness confound was found in this local experiment.

## 2. Full archived-curve recomputation

The frozen JSON contains **12 distinct checkpoint keys** × **5 direction curves** = **60** curves, each with 5 positive finite objective values. The zero-radius objective value is **bitwise-identical across the five directions belonging to each checkpoint**, as expected for a deterministic evaluation.

The stronger pointwise monotonicity check passed for all 60:

`L(-0.10) > L(-0.05) > L(0) < L(+0.05) < L(+0.10)`.

This **extends** the existing archived summary checker, which tested only the two outer endpoint values against the center.

Define the symmetric mean finite-radius excess (percent) as

`E(h) = 100*([L(+h)+L(-h)]/(2*L(0)) - 1)`.

Across 60 separate directions:

| Descriptive measure | Exact recomputation |
| --- | ---: |
| E(0.05), median over directions | **0.797759%** |
| E(0.05), observed range | **0.521913%–1.209089%** |
| E(0.10), median over directions | **3.152563%** |
| E(0.10), observed range | **2.098184%–4.889536%** |
| Median over directions of `E(0.10)/E(0.05)` | **4.011269** |
| Observed range of `E(0.10)/E(0.05)` | **3.7552–4.2320** |
| Curves monotone on both sides, including ±0.05 | **60/60** |
| Checkpoint centers consistent across their five directions | **12/12** |

The E(0.10) median and range independently reproduce the frozen published rounded values (3.15%, 2.10%–4.89%). `median(E(0.10)/E(0.05))` is **not** the same as `median(E(0.10))/median(E(0.05))`; the values above are explicitly defined.

## 3. Scientific interpretation: finite-radius directional symmetry, not a curvature theorem

For a twice-differentiable local objective, the symmetric excess along a fixed direction has leading term `(h^2/2) v^T H v`, so doubling a small `h` would produce roughly a fourfold excess. The empirical median ratio **4.01** is consistent with approximately quadratic *even-in-radius* behavior **over those two chosen finite steps and those sampled directions**.

That consistency is descriptive and does **not** prove that:
- the fitted checkpoint is a stationary/local minimum on the population or training objective;
- the full Hessian is positive semidefinite;
- every parameter-space direction rises, or the landscape is convex;
- the curvature ratio characterizes Hessian eigenvalues or robustness to larger perturbations;
- these 60 curves are independent trials (the underlying 12 checkpoints share paired seed protocols).

The relative radius 5%/10% of the **entire parameter vector norm** is not a limit approaching zero. Changes in architecture scaling or different layer-normalized perturbation rules can alter these slice profiles. The global-L2 direction convention must remain explicit.

## 4. Recommended paper treatment

The unmarked R1 Section 4.3 already states the appropriate descriptive scope and correctly reports rounded endpoints. The source-based audit strengthens, rather than invalidates, the existing experiment.

If another revision is invited, one optional compact sentence can make the repeat-measure design and fixed-measurement control explicit:

> The 60 local slices comprise five random directions for each of 12 independently trained loss–seed checkpoints; a fixed set of 128 validation sequences and one fixed mask are reused at every displacement, with dropout disabled. All sampled slices increased monotonically away from the center at both ±5% and ±10% relative global-L2 perturbation radii. The ratios of symmetric losses at these two finite radii are close to quadratic scaling, but do not estimate Hessian spectra or establish local convexity.

For strict statistical language, avoid implying the 12 matched-loss/seed checkpoints are fully independent across different losses: the loss comparisons share the same seed-specific pretraining/validation examples and experiment design.

**Priority:** Low–medium. Keep current R1 Figure S1, existing numbers and cautiously limited conclusion. Do not add a Hessian experiment or further random directions merely to make a stronger but unjustified theoretical claim.

## 5. Audit implementation

Companion no-GPU standard-library script: `revision_audits/audit_local_landscape_fixed_slices.py`. It verifies source controls, archived hashes, 60 full directional monotonicity sequences, 12 center-value checks, published endpoint summaries and the approximately quadratic two-radius ratio. The raw archived curves were independently analyzed in this turn using live GitHub content; the newly added standalone Python audit has **not yet been executed** from a full local clone. No existing frozen file was changed.
