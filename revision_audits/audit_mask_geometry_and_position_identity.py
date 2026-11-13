"""CPU-only structural audit: raw-sample coverage and masked-slot identity.

Python 3.10+, PyTorch CPU or CUDA build; no dataset or experiment retraining.
Run: python revision_audits/audit_mask_geometry_and_position_identity.py
This is a structure-level demonstration, not a performance re-evaluation.
"""
from __future__ import annotations

from math import comb, exp, log, sin, cos
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "revision_v70_20261010/code/floc_msm_main_corrected.py"


def coverage():
    length, kernel, stride, padding = 128, 8, 4, 2
    tokens = (length + 2 * padding - kernel) // stride + 1
    assert tokens == 32
    positions = [tuple(i for i in range(tokens)
                       if i*stride-padding <= t < i*stride-padding+kernel)
                 for t in range(length)]
    assert sum(len(pos) == 1 for pos in positions) == 4
    assert sum(len(pos) == 2 for pos in positions) == 124
    masked_count = 16
    p_one = masked_count / tokens
    p_two = comb(tokens-2, masked_count-2) / comb(tokens, masked_count)
    uniform_expected = sum(p_one if len(c) == 1 else p_two for c in positions)
    block_counts = []
    for start in range(tokens-masked_count+1):
        masked = set(range(start,start+masked_count))
        block_counts.append(sum(all(i in masked for i in c) for c in positions))
    block_expected = sum(block_counts)/len(block_counts)
    assert abs(uniform_expected-32) < 1e-12
    assert abs(block_expected-1024/17) < 1e-12
    assert block_counts.count(60) == 15 and block_counts.count(62) == 2
    print(f"16/32 random tokens: E[fully unobserved raw] = {uniform_expected:.6f}/128 ({uniform_expected/128:.2%})")
    print(f"16/32 contiguous:    E[fully unobserved raw] = {block_expected:.6f}/128 ({block_expected/128:.2%})")
    print(f"Difference: {(block_expected-uniform_expected)/128*100:.4f} percentage points of raw coordinates")
    # Optional comparison-design reference; 9-token block approximately
    # matches raw-coordinate hiding, but NOT the same masked-token ratio.
    nine = []
    for start in range(tokens-9+1):
        masked = set(range(start,start+9))
        nine.append(sum(all(i in masked for i in c) for c in positions))
    assert abs(sum(nine)/len(nine) - (32+1/6)) < 1e-12
    print(f"9/32 contiguous:     E[fully unobserved raw] = {sum(nine)/len(nine):.6f}/128")


def masked_identity():
    import torch
    from torch import nn
    torch.set_num_threads(1)
    source = SRC.read_text(encoding="utf-8")
    # Structural guards: PE is introduced before the mask replacement;
    # no second masked-token positional encoding is added after where().
    assert "tok=enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1,2))" in source
    assert "masked=torch.where(mask,mask_tok.expand(xb.shape[0],tok.shape[1],-1),tok)" in source
    assert "recon=dec(enc.trf(masked).mean(dim=1))" in source
    torch.manual_seed(20261113)
    d = 64
    positional = torch.zeros((1,32,d))
    indices = torch.arange(32,dtype=torch.float32).unsqueeze(1)
    omega = torch.exp(torch.arange(0,d,2,dtype=torch.float32)*(-log(10000.0)/d))
    positional[0,:,0::2] = torch.sin(indices*omega)
    positional[0,:,1::2] = torch.cos(indices*omega)
    conv = nn.Conv1d(1,d,8,stride=4,padding=2)
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d,4,256,dropout=0.1,batch_first=True),4)
    model.eval()
    sample = torch.randn((1,1,128))
    token = conv(sample).transpose(1,2)+positional
    mask = torch.zeros((1,32,1),dtype=torch.bool)
    mask[:,5:21,:] = True
    mask_token = torch.randn((1,1,d))*0.02
    masked = torch.where(mask,mask_token.expand(1,32,d),token)
    assert (masked[:,5:21]-masked[:,5:6]).abs().max().item() == 0.0
    with torch.no_grad():
        output = model(masked)
    maxdiff = (output[:,5:21]-output[:,5:6]).abs().max().item()
    print(f"Masked-slot output identity, eval mode: max |difference|={maxdiff:.8g}")
    assert maxdiff < 1e-5, maxdiff
    print("PASS: masked slots lose their explicit PE; identical slots remain identical under eval-mode encoder.")
    print("Visible slots retain position embeddings; missing positions are still inferable indirectly.")
    print("Training-time independent dropout may break per-forward masked-output equality.")


if __name__ == "__main__":
    coverage()
    masked_identity()
