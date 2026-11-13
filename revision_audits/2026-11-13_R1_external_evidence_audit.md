# 26SP R1: recorded-noise / bearing-vibration evidence audit

Date: 2026-11-13.
Status: post-R1 prospective revision recommendation. **Do not interpret as an alteration of R1, revised figures, or frozen v70 results.**
Paper: SIGPRO-D-26-02622R1, "Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective".

Audited in GitHub archive revision_v70_20261010:
- code/noisex92_ucr_seeded.py
- results/noisex/results.json, results/noisex/summary.json, results/noisex/run_metadata.json
- code/cwru_loss_compare_sourceclosed.py
- results/cwru/metadata.json, results/cwru/results.json

Archived calculations below use the GitHub-connector source data; new independent checker is [audit_noisex_cwru_external_protocols.py](audit_noisex_cwru_external_protocols.py). The checker runs without GPU or data downloads, but the newly added copy has not been execution-tested in a local clone.

## Finding 1: NOISEX samplewise RMS-based 5 dB calibration is valid at the specified mixing stage

For a clean sequence x and selected recorded-noise segment n, the runner scales n by

    a = RMS(x) / (RMS(n)+1e-12) * 10^(-5/20)

and mixes x+a*n. Ignoring harmless numerical stabilizers on non-degenerate segments, the injected noise-to-signal RMS ratio is 10^(-5/20), i.e., nominal power SNR=5 dB. Per-sequence max-abs scaling is applied **after mixing**; this common scalar leaves the decomposition's power ratio unchanged.

This supports describing the experiment as *RMS-calibrated 5 dB injection of excerpts from NOISEX-92 recordings*. The code does **not** read or align noise WAV sampling rates to the distinct physical/sampling semantics of UCR time series, so this is **not** a sample-rate-calibrated physical sensor/noise simulation. Short excerpts come from three recordings (machinegun, factory1, factory2). Do not infer real operating-field SNRs.

At a given master seed, all losses share the same selected/noisy train and test signals, the same augmented pretraining pool, the same per-class few-shot indices, and a common fine-tuning seed. Across the five master seeds, the runner re-mixes noise and re-pretrains separately. Its five-seed SD thus captures a broader pipeline-replication hierarchy than the **one-pretraining-checkpoint-per-loss** ordinary synthetic or ordinary UCR experiments. This distinction should be preserved in the paper.

## Finding 2: win counts and mean accuracy give different conclusions on NOISEX

Reaggregation of archived *ordinary test accuracy* equally over the eight UCR datasets × two shot counts (16 conditions), using each cell's mean over five master seeds:

| Method | Equal-weight mean test accuracy (%) | Number of best-mean cells |
| --- | ---: | ---: |
| scratch | 52.0843 | 1 |
| MSE | 58.3917 | 5 |
| L1 | 58.2973 | 8 |
| FLOM-based p-moment, p=1.2 | 58.1749 | 2 |

These win counts agree with results/noisex/summary.json, but **MSE has the numerically highest equal-weight average** while L1 has the most best-mean wins. Compared with MSE, L1 averages -0.0944 percentage points (eight wins vs MSE), and p=1.2 averages -0.2168 points (seven higher than MSE, one tie). No inference of significance/equivalence follows from such differences.

Per-dataset p=1.2 minus MSE effect averaged across the two shot budgets, in percentage points:
- ECG200 +1.20; ECG5000 -0.50;
- Wafer +3.18; GunPoint -2.73;
- CBF -2.03; FaceAll -2.17;
- SwedishLeaf +0.91; ChlorineConcentration +0.41.

These are not test-set-size-weighted pooled accuracies; 5- and 10-shot cells within each dataset share underlying data and do not constitute 16 independent sampled datasets.

Suggested future manuscript text, for the NOISEX subsection:
> Under 5 dB RMS-calibrated injection of recorded NOISEX-92 excerpts, rankings varied by dataset and labeled budget. L1 achieved the highest mean in 8 of 16 dataset–budget conditions, compared with five for MSE and two for the p-moment loss. Yet the equally weighted mean accuracies across all 16 conditions were 58.39% (MSE), 58.30% (L1), and 58.17% (p=1.2). Hence win counts do not establish a consistent overall advantage for a robust objective. The audio excerpts are mixed at the sample-index level without physical sampling-rate alignment to UCR sensors; this study is a recorded-waveform perturbation check, not a field-calibrated measurement experiment.

Use "recorded noise from NOISEX-92" rather than "realistically calibrated physical noise" or "field-confirmed robustness".

## Finding 3: CWRU is load-held-out, but window observations are strongly clustered

The CWRU runner:
- uses four bearing classes (Normal / IR007 / B007 / OR007@6);
- uses source files at load 0/1/2 hp for train, load 3 hp for test;
- obtains **12 train-load source files and four test-load source files** (16 total, archived hashes);
- segments time series into length-1024 windows with stride 512, **50% overlap for adjacent windows**;
- archives 2850 train windows and 953 test windows;
- regenerates α=1.0 impulsive noise and retrains each loss within each of five seeds.

This is not an overlapping-window *cross-split* leakage: source files from the held-out load are not used for training. However, the 953 test windows are **not 953 independent operating-condition replicates**: adjacent windows share raw data and each held-out class comes from one load-specific recording. A window-wise classification accuracy is legitimate as the reported metric, but inferential/generalization claims must not treat it as independent experimental replication across bearings or acquisition runs.

The actual best means at 10/20/50 shots, respectively, are 27.41% (L1), 28.54% (Charbonnier), and 31.10% (p-moment), compared with nominal four-class chance 25%. This supports treating CWRU as a *difficult, near-chance stress check* rather than persuasive evidence of robust field classification.

## Finding 4: CWRU post-noise z-scoring also bounds residual input coordinates

The CWRU runner first per-window z-scores raw segments, adds synthetic α=1.0 noise, and **again** centers and z-scores the resulting 1024-sample window via

    y_i = (v_i - mean(v)) / (std(v) + 1e-6),

where std is computed over the window with denominator N=1024.

Let s=std(v). Then

    sum_i y_i^2 = 1024 * (s/(s+1e-6))^2 <= 1024,

and in particular |y_i| <= 32 for each finite post-normalized segment (strict inequality when s>0 and finite). At fixed finite network weights, its reconstruction residual is consequently bounded over the normalized input domain. So a non-degenerate exactly α-stable **learned residual law** does not apply to this processed CWRU protocol either; this is a mathematical bound and does not estimate a learned residual tail index.

Do not confuse the unbounded pre-normalization injected α-stable noise with the post-processing residual model.

Suggested future manuscript text, for CWRU limitations:
> The CWRU load-held-out check uses different source recordings at train and test loads and 1024-sample windows with a 512-sample stride. Adjacent windows overlap, and the held-out load contributes only one source recording per class. Accordingly, window-level test counts are not independent machinery runs; the near-chance accuracies should be interpreted as a severe stress diagnostic rather than validated industrial deployment performance. Post-noise per-window standardization also bounds the processed inputs, precluding literal use of the idealized non-degenerate α-stable residual assumption.

## Disposition

- No confirmed numerical error in the two archived result tables.
- One meaningful **presentation discrepancy**: NOISEX winning-condition count versus equally weighted mean accuracy. Report both if revising externally.
- Scope corrections: NOISEX is sample-index recorded-noise mixing without physical sampling-rate alignment; CWRU 953 windows are not 953 independent test acquisitions.
- A small theoretical clarification: post-noise CWRU per-window standardization itself bounds processed coordinates.
- No fresh GPU work is necessary to make these protocol and interpretation statements.
- Frozen v70 release and assembled R1 remain untouched.
