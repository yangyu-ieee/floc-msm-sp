# FLOC-MSM: Robust Masked Signal Modeling under Impulsive Noise

Reproducibility package for: *"Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective"* (submitted to Signal Processing, Elsevier).

## Requirements

```bash
conda create -n floc-msm python=3.12
conda activate floc-msm
pip install torch numpy scipy scikit-learn matplotlib aeon
```

GPU recommended (RTX 5060 Ti 16GB used in paper). CPU fallback works but slower.

## Quick Start

```bash
# Main experiment (alpha=1.5, 10 seeds, FLOC vs MSE vs L1 vs Huber)
python code/floc_10seed.py

# Multi-alpha robust loss comparison (alpha=2.0/1.5/1.0)
python code/floc_tsp_loss_compare_fixed.py

# UCR benchmark with NOISEX-92 real impulsive noise
python code/noisex92_ucr.py

# Statistical analysis
python code/stats_analysis.py
```

## Reproducing Key Results

| Figure/Table | Script | Output |
|---|---|---|
| Table 1 (Main alpha=1.5) | `code/floc_10seed.py` | `results/floc_10seed/` |
| Table 2 (Multi-alpha loss map) | `code/floc_tsp_loss_compare_fixed.py` | `results/tsp_loss_compare_fixed/` |
| Table 3 (UCR benchmark) | `code/ucr_all_losses.py` | `results/ucr_all_losses/` |
| Figure 5 (NOISEX-92 check) | `code/noisex92_ucr.py` | `results/noisex92_ucr/` |
| Statistical significance | `code/stats_analysis.py` | stdout |

## Data

- **UCR datasets**: auto-downloaded via `aeon` package
- **CWRU bearing data**: Case Western Reserve University Bearing Data Center (public)
- **NOISEX-92**: public noise corpus (bundled with most signal processing toolboxes)
- **Synthetic data**: generated on-the-fly by the scripts

## License

MIT License — see LICENSE file.

## Citation

If you use this code, please cite:
```
[Paper citation to be added upon publication]
```
