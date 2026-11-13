# 26SP R1: controlled pretraining comparisons and the 2026 STAR-MAE neighbor

Date: 2026-11-13. Status: prospective post-assembly audit. This does not replace or rewrite the frozen v70 results, the assembled Editorial Manager R1, or its citations.

## A. Within the corrected alpha=1.5 core trial, the pretraining controls are particularly well matched

Verified against revision_v70_20261010/code/floc_msm_main_corrected.py and the archived main reconciliation config/results.

- The code draws the 10,000 unlabeled examples **once** before entering the loss loop. All losses see the same unlabeled pretraining signals and the same corrected-CMS corruption draws.
- The main core includes MSE, L1, Huber (delta=1), and FLOM-based p-moment (p=1.2). Before each pretraining run, torch.manual_seed(args.seed+1) is called. Both the encoder and decoder/mask token are constructed after this reset, aligning their initial random draws for the fixed architecture.
- Every loss creates a fresh DataLoader with an independent generator initialized at **the same fixed seed** args.seed+2. This matches shuffled pretraining minibatch order across losses.
- Within those four fixed branches, the same call structure and tensor shapes drive token-mask sampling and network dropout. Because PyTorch randomness is restarted before each run, the random streams are designed to be aligned, although bitwise identity of every stochastic mask/dropout realization was not independently instrumented.
- The Adam learning rate, architecture, number of epochs, and encoder-only gradient clip threshold are fixed. The only objective-specific differentiation in the core pretraining loops is the loss formula and ensuing optimizer dynamics.
- This is strong *within-run, common-random-number control*, but there is still only **one pretrained checkpoint per objective**. Five downstream seeds regenerate synthetic training and testing examples and fine-tuning randomness, but do not estimate the variance over independently reinitialized/retrained SSL encoders.
- The downstream fixed templates differ in waveform family distribution and noise scale from the randomized unlabeled pretraining generator; the unmarked R1 Section 4.1 already correctly describes these as related but nonidentical distributions. Do not call the downstream task perfectly pretrain-distribution-matched.

Suggested revised Section 4.1 paragraph:

> The corrected alpha=1.5 core benchmark uses a common 10,000-sample unlabeled pretraining corpus for all objectives. To reduce nuisance variation across objectives, the four loss runs restart from the same model-initialization seed and independently seeded-but-identical shuffle schedule, with a fixed architecture, mask ratio, optimizer, and training budget. The stochastic masking and dropout procedures use the same seeded random streams under this common architecture. This controlled comparison isolates the effect of the specified reconstruction objective within one pretraining realization; the five matched downstream data-generation and fine-tuning runs do not provide independent pretraining replicates.

Note: 'isolates' here is within the *specified fixed protocol* and does not imply a universal causal improvement on independently selected architectures, tasks, or SSL seeds.

## B. Newly verified close 2026 neighbor: STAR-MAE (Pattern Recognition)

Direct publisher record: https://www.sciencedirect.com/science/article/pii/S0031320326000968
Associated reproducibility repository: https://github.com/Jiayu-Xiong/STAR-MAE

J. Xiong, J. Wang, W. Wang, X. Lyu, J. Kwan, and J. Xue, “Masked autoencoders for spatio-temporal audio representations: Theory and optimization,” *Pattern Recognition*, vol. 175, 113133, July 2026. DOI: 10.1016/j.patcog.2026.113133.

Publisher page explicitly reports structured audio masking and a Distribution-aware Loss Reweighting (DLR) module intended to suppress extreme/noisy reconstruction gradients and improve stable pretraining. It reports 10 audio classification tasks and around 40% lower training cost in the authors' setting. The source exists before the revised 26SP October 2026 document; it was not located by title/author/text search in the R1 assembled manuscript.

Provenance and scope comparison:

| Dimension | STAR-MAE (Xiong et al., 2026) | 26SP R1 |
| --- | --- | --- |
| Research object | Spatio-temporal *spectrogram/audio* representation architecture | One-dimensional raw-signal masked reconstruction with fixed Conv1D–Transformer backbone |
| Algorithmic change | Structured masking plus DLR module | Comparisons among pre-existing pointwise reconstruction losses, without a new architecture or reweighting mechanism |
| Tail/stability claim | DLR attenuates noisy gradient contributions and aids optimization in its audio experiments | Conditional stable-residual moment windows; actual normalized residuals need not be stable; no network-convergence claim |
| Empirical scope | Audio classification suite and pretraining efficiency | Corrected synthetic S-alpha-S injected-noise tasks, UCR, NOISEX injection, and near-chance CWRU diagnostics |
| Appropriate relation | Direct noisy-gradient/masked-learning neighbor | Complementary diagnostic comparison, **not** first use of robust gradient reweighting in masked learning |

Suggested restrained Section 2.1 insert:

> Recent audio-specific masked autoencoding work also treats reconstruction-gradient quality as an optimization concern. Xiong et al. [new citation] combine structured spectrotemporal masking with distribution-aware reconstruction-loss reweighting to down-weight noisy extremes in audio pretraining. Their method introduces an adaptive architectural/training mechanism and evaluates audio representations, whereas we hold the one-dimensional signal backbone fixed to examine the behavior of established pointwise losses under controlled impulsive corruption. Our contribution is a protocol- and assumption-aware assessment of these objectives, not a new reweighting rule or the first demonstration of outlier-sensitive gradients in masked learning.

Suggested future rebuttal sentence, if novelty challenged:

> We have acknowledged distribution-aware loss reweighting in prior masked-audio learning and distinguish it from our fixed-backbone, lower-order residual-moment audit. Our scope is comparative and diagnostic, not a new robust masking algorithm.

## C. Avoid introducing a new unrun STAR-MAE experiment by default

The methods use different modalities and data representations (spectrogram/time-frequency features vs 1-D raw sequences), masking policies, and architectural designs. A numerical cross-paper accuracy comparison would be unsound. Treat STAR-MAE as a **prior-art boundary reference**, not as a directly comparable baseline requiring a potentially expensive architectural port. New benchmarking is optional only if a reviewer explicitly asks for it and a fair common task can be defined.

## D. Priority

1. Future editable R1: add a one-paragraph pretraining matching explanation so readers appreciate the controlled comparison, while clearly marking one-checkpoint inference.
2. Future editable R1: insert the Xiong et al. 2026 citation and the four-sentence scope contrast.
3. Retain the corrected alpha=1.5 test and Cauchy Welch statements, but do not expand the work into a new STAR-MAE-based algorithm.
4. Preserve frozen v70 and already-submitted R1 as historical record.
