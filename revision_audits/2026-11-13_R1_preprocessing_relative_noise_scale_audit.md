# 26SP R1: controlled CMS normalization experiment also changes relative impulsive-noise scale

Date: 2026-11-13. Status: post-submission audit and suggested next-round precision edits. **Do not change frozen v70, submitted R1 PDF, or its Table 2 values.**

## 1. Source and archived-data verification

Checked:

- `revision_v70_20261010/code/r5_preprocessing_audit.py`, matching frozen manifest SHA-256 `5e2655170dbbeb54771fc4c473784cd27e6765dd5641ef51cc6c4fcfc6434cf9` as listed in `results/normalization/config.json`;
- `revision_v70_20261010/results/normalization/{config.json,input_diagnostics.json,results.json,summary.json}`;
- assembled unmarked R1 manuscript Section 4.4 and Table 2 (PDF pages 82–83).

The two corrected-CMS arms share **exactly the same** clean pretraining signals and base CMS variables `U,W`; downstream they share clean train/test sets and base noise variables for a given budget and seed, with identical pretrained model initialization/shuffle seeds and matched downstream finetuning seeds. The pretraining/checkpoint is still **one per arm and loss**, repeated downstream with five seeds. This confirms good common-random-number design, without establishing independent SSL replication.

The third, `submitted_legacy_maxabs`, represents the withdrawn incorrect sampler and **is not** an acceptable `S_alpha S` benchmark. The following contrasts exclude it.

## 2. Exact scale derivation: beyond operation order, the RMS arm reduces noise relative to clean signal

Let `x` denote the clean signal *including pre-CMS Gaussian jitter* and `Z` a corrected unit-scale symmetric alpha-stable draw, with the same paired `Z` in both arms. Let `sigma` denote the stable-noise scale, and `S` the RMS of the appropriate clean training pool.

**Corrected noisy-max-abs arm**:

    y_max = (x + sigma Z) / (||x + sigma Z||_infinity + 1e-8).

**Corrected clean-global-RMS arm**:

    y_rms = (x / S) + sigma Z.

Multiplying the latter by `S` (a global, positive constant) yields

    S*y_rms = x + (S*sigma) Z.

Thus, *in raw clean-signal units and modulo a common global scaling of the input*, the RMS arm reduces the S-alpha-S noise amplitude scale from `sigma` to **`S*sigma`**. It is **not** a matched-relative-noise comparison, even though the underlying noise variates and the code's nominal `noise_scale` are identical.

This equality does not assert network-invariance to a global scaling; in fact learned networks and optimization are generally scale-dependent. It isolates one precise confounding factor. The max-abs arm separately applies stochastic, sample-dependent scaling, whose effect is not reduced to changing sigma.

Archived RMS data:

| Stage | Clean RMS S | Nominal sigma | Effective raw-coordinate noise scale S*sigma in RMS arm | Relative ratio versus max-abs pre-normalization noise scale |
| --- | ---: | ---: | ---: | ---: |
| Pretraining | 0.625311 | 0.15 | 0.093797 | **62.5311%** |
| Downstream, 100 or 200 labels, 5 seeds each | 0.718149–0.719107 | 0.30 | 0.215445–0.215732 | **71.8149–71.9107%** |

The downstream S is computed from each jittered clean training split and reused for that split's test data. Neither arm uses noisy test data to compute its RMS.

Do **not** translate these scales into *finite-variance SNR* or dB derived from population noise power: the ideal alpha-stable noise has infinite variance at alpha=1.5. The ratios are stable-law **scale parameters**, not power ratios.

## 3. The input distributions are not of comparable dynamic range

Archived pretraining absolute-value diagnostics:

| Metric | CMS then max-abs | Clean-RMS then CMS |
| --- | ---: | ---: |
| abs q50 | 0.116846 | 0.592096 |
| abs q90 | 0.465255 | 1.782037 |
| abs q99 | 0.924798 | 3.158628 |
| abs q99.9 | 1.0000 | 8.3360 |
| max | 1.0000 | **5028.8594** |

These are realized archived input statistics, not a claim about asymptotic residual moments or convergence. In the max-abs arm, a large impulse forces a large sample-specific denominator for both signal and noise, which can attenuate the usable clean waveform amplitude; in the clean-RMS arm the denominator is not chosen from each corrupted sequence. That mechanism is mathematically plausible but **its quantitative contribution to the accuracy difference has not been isolated**.

## 4. Complete paired Table 2 reconciliation

| Loss | Labels | CMS max-abs accuracy | Clean-global-RMS accuracy | RMS minus max-abs (pp) | Paired seed differences: positive / equal / negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| MSE | 100 | 94.04% | 97.40% | +3.36 | 5 / 0 / 0 |
| MSE | 200 | 96.20% | 98.08% | +1.88 | 5 / 0 / 0 |
| L1 | 100 | 95.64% | 98.40% | +2.76 | 5 / 0 / 0 |
| L1 | 200 | 96.52% | 98.80% | +2.28 | 4 / 1 / 0 |
| Charbonnier | 100 | 95.68% | 98.64% | +2.96 | 5 / 0 / 0 |
| Charbonnier | 200 | 95.44% | 98.84% | +3.40 | 5 / 0 / 0 |
| p=1.2 | 100 | 94.76% | 98.40% | +3.64 | 5 / 0 / 0 |
| p=1.2 | 200 | 96.40% | 99.00% | +2.60 | 5 / 0 / 0 |

The mean of these **eight cell means** is +2.86 percentage points. Across the 8 × 5 matched comparisons, 39 paired differences are positive and one is exactly zero; there are no negative comparisons. This directional consistency is descriptively interesting, but **the 40 cells/seed differences are not independent replicates**: losses, budgets, downstream generated datasets, and arm-specific pretrained checkpoints induce dependence. Do not conduct a naive 40-trial sign/binomial significance calculation or use the contrasts as a population-wide effect estimate.

## 5. Suggested next-round English manuscript language

Add after Table 2's current acknowledgement that operation order and effective noise level differ:

> The clean-global-RMS arm also changes the relative scale of the injected impulsive noise: if S is the RMS of the jittered clean training pool, scaling x by 1/S before adding a fixed-scale stable perturbation yields y = x/S + sigma Z, or equivalently Sy = x + S sigma Z. In the archived runs, S was 0.6253 during pretraining and 0.7181–0.7191 in the downstream splits. The nominal CMS scale was therefore only 62.5% and about 71.8–71.9%, respectively, of its max-abs-arm pre-normalization counterpart in raw clean-signal coordinates. This and the large differences in input dynamic range mean the observed 1.88–3.64 percentage-point accuracy changes cannot be attributed to normalization in isolation.

Optional shorter form if page limits constrain length:

> The clean-RMS arm implicitly reduces the stable-noise scale relative to the unscaled clean signal by its training-pool RMS (0.625 in pretraining; about 0.719 downstream). Hence this is a *joint scaling and effective-corruption* sensitivity test, not a normalization-only ablation.

## 6. Assessment and workflow

**No newly discovered code defect in the paired Table 2 comparison; the code implements the stated three arms and the manuscript's caution is directionally correct.** The new audit upgrades a qualitative warning (effective noise level differs) to an exact algebraic explanation and archived quantitative ratios. This clarification increases scientific value, as normalization can strongly suppress absolute template information under extremely impulsive contamination.

Companion standard-library checker: `revision_audits/audit_preprocessing_effective_noise.py`; independently reaggregated from GitHub archived JSON and source. No GPU retraining. The newly authored Python script has not yet been run within a complete local checkout.
