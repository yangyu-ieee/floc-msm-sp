# 26SP R1: masked-slot positional identity and raw-sample exposure imbalance

Date: 2026-11-13. Status: prospective post-submission R1 architecture audit. Frozen v70 code, archived results and submitted R1 are unchanged.

## Verified source paths

- `revision_v70_20261010/code/floc_msm_main_corrected.py` (tokenizer, positional encoding, masked-token substitution and mean-pooling decoder).
- `revision_v70_20261010/code/r2_mcc_baseline.py` (same architecture in validation-selected baselines).
- `revision_v70_20261010/code/noisex92_ucr_seeded.py` and `revision_v70_20261010/code/cwru_loss_compare_sourceclosed.py` (same *ordering* of mask replacement after positional embeddings, although these external tests use differing signal lengths and corruption protocols).
- R1 unmarked manuscript Section 3.1 (assembled PDF page 70) already distinguishes its full-signal loss from canonical masked-only MAE; Section 4.1 (PDF page 76) acknowledges overlapping token receptive fields. Section 4.4 (pages 83–84) reports 7 random-token wins and one tie at the same *token* masking ratio.

## 1. Structure-level finding: masked token locations do not have individual explicit positional embeddings

The core pretraining loop contains:

```python
tok=enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1,2))
masked=torch.where(mask,mask_tok.expand(xb.shape[0],tok.shape[1],-1),tok)
recon=dec(enc.trf(masked).mean(dim=1))
```

Because learned mask-token replacement occurs **after** positional encoding, each masked input token is the same learned `mask_tok` vector and **does not retain its own explicit position embedding**. In evaluation mode (dropout off), a Transformer encoder without subsequent position-dependent operations treats these identical masked input slots symmetrically; their output features are identical. The final reconstruction decoder consumes a **mean over all encoded tokens**, rather than a position-specific mask-token decoder.

This does **not** imply that mask locations are completely unknown to the model: visible tokens retain positional embeddings and therefore implicitly identify which positions are missing. The model can also recover time structure through the fixed-index 128-dimensional output head. Training-mode dropout can make masked-slot outputs numerically unequal on a single stochastic forward pass; that is not learned positional encoding.

A CPU-only structural reproduction with 4-layer, d=64, 4-head `nn.TransformerEncoder`, kernel-8/stride-4/pad-2 tokenizer, 16 identical masked tokens and a positional table applied before replacement yielded exact zero difference between masked outputs in `.eval()` mode. This test demonstrates an architectural symmetry, not a new trained-checkpoint evaluation.

## 2. Exact raw-sample coverage: equal token masking does not equal equal raw exposure

At raw length 128, kernel 8, stride 4, padding 2, there are 32 overlapping token windows. Four raw coordinates feed exactly one token; the other 124 feed exactly two. Count a raw coordinate as *directly hidden* when **every tokenizer token containing it is masked**. This is not a claim that correlated signal content is statistically irrecoverable.

For 16-of-32 uniform random tokens:

- A coordinate covered by one token is hidden with probability `16/32=1/2`.
- A coordinate covered by two tokens is hidden with probability `C(30,14)/C(32,16)=16*15/(32*31)=15/62`.
- The expected fully hidden raw coordinates are `4*(1/2)+124*(15/62)=32`, i.e. **25.0000%**.

For a contiguous block of 16 tokens with uniformly selected start from the 17 possible placements:

- Fifteen placements directly hide 60 raw coordinates; two edge placements hide 62.
- Expected hidden count `(15*60+2*62)/17=1024/17=60.235294`, i.e. **47.0588%** of all raw coordinates.
- The between-arm gap is `(8/17-1/4)*100=22.0588 percentage points`; the block hides **1.88235 times as many** raw coordinates under this direct receptive-field definition.

Thus the Section 4.4 accuracy result (random-token higher mean in seven of eight cells, one numerical tie) cannot be interpreted as a pure mask-shape effect: at the same fraction of *tokens*, the geometric extent of raw signal exposure is substantially different.

## 3. Optional geometry-matched diagnostic (not an instruction to rerun GPU experiments)

If an additional control were ever necessary, a *9-of-32 contiguous block* would directly hide `32 + 1/6 = 32.166667` raw coordinates on average, very close to a random 16-token pattern's `32.000000`. This would **match raw coverage approximately**, but **would change masked token fraction from 50% to 28.125%**, so it alone would not identify all causal mechanisms. A proper factorial study of pattern, token ratio and raw exposure would exceed the current loss-focused paper's scope. Do not add new experiments unless reviewer-driven.

## 4. Manuscript/rebuttal wording, for next editable revision only

Suggested Section 4.1 sentence:

> In our global-reconstruction encoder, positional encodings are added to convolutional tokens before learned-mask-token substitution. Thus masked input slots share the same embedding, while visible tokens retain their positions; the decoder reconstructs the full time series from a mean-pooled representation, rather than decoding individual masked positions. This architectural choice is held fixed across the reconstruction-loss comparisons.

Suggested Section 4.4 replacement after the 7/8 count:

> Although the random-token and contiguous-block conditions both mask 16 of 32 overlapping tokens, they do not hide the same number of raw sample coordinates. The expected raw positions with no path into any visible token are 32/128 for uniformly selected random-token masks and 60.24/128 for a uniformly located 16-token block. Hence the observed masking-pattern contrast combines pattern topology with effective raw-coordinate visibility, and cannot isolate the effect of mask geometry alone. Both conditions use a global full-signal reconstruction objective and one pretrained encoder per loss/pattern; the downstream seeds also regenerate synthetic samples.

Suggested scope statement:

> Our conclusions compare pointwise reconstruction objectives within a fixed global masked-signal architecture. They should not be transferred without validation to per-patch decoding masked autoencoders that preserve individual masked positions or optimize masked patches only.

These amendments are more important than another mask-ratio accuracy bar chart: they define what the study actually tests.

## 5. Reproducibility and risk grading

- A new non-training structural check is available at `revision_audits/audit_mask_geometry_and_position_identity.py` (requires CPU-capable PyTorch, and validates both geometry and masked-input/output identity against fixed architecture).
- The independently constructed equivalent CPU Transformer test was executed and passed (`max absolute masked-output difference = 0`); the newly added GitHub script itself has not yet been executed directly against a full local clone.
- Important but **not a correctness bug** in existing accuracy tables. Source hash of historical exact-count random masking arm remains unrecovered as acknowledged by frozen v70 README, so this audit does not certify cross-arm execution equivalence.
- STAR-MAE prior-art citation (Xiong et al., *Pattern Recognition* 175:113133, 2026, doi:10.1016/j.patcog.2026.113133) independently confirmed via publisher: https://www.sciencedirect.com/science/article/pii/S0031320326000968. Its use of structured spectrotemporal masking and distribution-aware loss reweighting supports mentioning adjacent work, not transplanting its algorithm into 26SP.
