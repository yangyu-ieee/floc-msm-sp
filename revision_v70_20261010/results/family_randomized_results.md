# Randomized-family synthetic diagnostic

All values are test accuracy (%), mean ± population SD across five matched complete seeds. Each loss was pretrained once per seed; each seed/loss/budget used one matched downstream fine-tuning run. Paired comparisons are within this randomized task only. The manuscript calls the proposed objective the $p$-moment loss; `floc_1.2` remains its legacy result key.

| Labels | Scratch | MSE | L1 | Huber | Charbonnier | $p$-moment loss ($p=1.2$) | $p$-moment loss−MSE (pp), exact p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 37.20 ± 3.39 | 38.92 ± 3.82 | 38.08 ± 4.63 | 36.68 ± 3.95 | 39.48 ± 3.67 | 39.28 ± 3.22 | +0.36; p=0.9375 |
| 100 | 29.48 ± 8.25 | 48.16 ± 2.91 | 47.64 ± 3.16 | 44.72 ± 1.83 | 47.92 ± 3.54 | 46.60 ± 3.62 | -1.56; p=0.7500 |
| 200 | 45.88 ± 3.81 | 53.24 ± 1.45 | 57.60 ± 4.34 | 52.24 ± 2.17 | 55.40 ± 4.30 | 56.64 ± 2.48 | +3.40; p=0.0625 |
| 500 | 56.52 ± 3.23 | 62.48 ± 0.97 | 66.96 ± 2.56 | 63.72 ± 1.82 | 67.00 ± 1.08 | 65.16 ± 1.32 | +2.68; p=0.0625 |

## Seed-level paired $p$-moment loss ($p=1.2$) minus MSE

| Labels | Seed differences (pp) | Mean delta (pp) | Exact two-sided sign-flip p |
|---:|---|---:|---:|
| 50 | +0.2, +2.8, -0.8, +2.6, -3.0 | +0.36 | 0.9375 |
| 100 | -1.4, +0.2, +2.0, -9.0, +0.4 | -1.56 | 0.7500 |
| 200 | +4.0, +3.4, +5.2, +2.4, +2.0 | +3.40 | 0.0625 |
| 500 | +2.4, +3.0, +1.8, +2.4, +3.8 | +2.68 | 0.0625 |

This is one fixed configuration. Do not compare its p-values or small differences inferentially against the historical fixed-template table, whose pretraining-seed structure differs.
