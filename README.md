> [!WARNING]
> **Superseded pre-revision release — do not use this repository to reproduce the revised manuscript.**
>
> The original submission's synthetic Chambers–Mallows–Stuck sampler did not implement the stated symmetric alpha-stable law for the affected alpha=1.5 experiments. The original alpha=1.5 results and associated significance claims were withdrawn during revision. The scripts and results below are retained as a record of the initial submission only.
>
> For the revised manuscript, use the corrected revision-specific code, configurations, and audit archive supplied with the resubmission. This repository is not that release.

# FLOC-MSM: Robust Masked Signal Modeling under Impulsive Noise

Legacy reproducibility package for the original submission of “Robust Masked Signal Modeling under Impulsive Noise: A Fractional Lower-Order Perspective” to Signal Processing (Elsevier). The contents below document the pre-revision implementation only.

## Requirements

```bash
conda create -n floc-msm python=3.12
conda activate floc-msm
pip install torch numpy scipy scikit-learn matplotlib aeon
```

GPU recommended (RTX 5060 Ti 16GB used in paper). CPU fallback works but slower.

## Quick Start

```bash
# Original pre-revision experiment (alpha=1.5, 10 seeds; superseded)
python code/floc_10seed.py

# Original multi-alpha robust loss comparison (superseded)
python code/floc_tsp_loss_compare_fixed.py

# Original UCR benchmark with NOISEX-92 real impulsive noise (superseded)
python code/noisex92_ucr.py

# Original statistical analysis (superseded)
python code/stats_analysis.py
```

## Original Result Mapping (Historical Only)

| Figure/Table | Script | Output |
|---|---|---|
| Original Table 1 (alpha=1.5; superseded) | `code/floc_10seed.py` | `results/floc_10seed/` |
| Original Table 2 (multi-alpha; superseded) | `code/floc_tsp_loss_compare_fixed.py` | `results/tsp_loss_compare_fixed/` |
| Original Table 3 (UCR benchmark; superseded) | `code/ucr_all_losses.py` | `results/ucr_all_losses/` |
| Original Figure 5 (NOISEX-92 check) | `code/noisex92_ucr.py` | `results/noisex92_ucr/` |
| Original statistical analysis | `code/stats_analysis.py` | stdout |

## Data

- **UCR datasets**: auto-downloaded via `aeon` package
- **CWRU bearing data**: Case Western Reserve University Bearing Data Center (public)
- **NOISEX-92**: public noise corpus (bundled with most signal processing toolboxes)
- **Synthetic data**: generated on-the-fly by the legacy scripts

## License

MIT License — see LICENSE file.

## Citation

If you use this historical code for the original, superseded results, cite the manuscript as appropriate. Do not cite this repository as the reproducibility release for the revised results.
