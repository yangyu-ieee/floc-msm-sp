"""One-batch L1 vs FLOC(p=1) parity probe against the packaged runner."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("floc_msm_main_corrected.py")
EXPECTED = "880fbd11139d81d5abe462cd5190a28c973dee1c54e7932d7f629a9c7b541104"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [str(__file__)]
source_bytes = SOURCE.read_bytes()
source_hash = hashlib.sha256(source_bytes).hexdigest()
if source_hash != EXPECTED:
    raise RuntimeError(f"Packaged runner hash mismatch: {source_hash}")
source = source_bytes.decode("utf-8")
prefix = source.split("# ═══════════════════════ Main ═══════════════════════")[0]
ns = {"__file__": str(Path(__file__).resolve())}
exec(compile(prefix, "archived_floc_msm_main_corrected.py", "exec"), ns)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = ns["device"]
SignalEncoder = ns["SignalEncoder"]
DataLoader = ns["DataLoader"]
TensorDataset = ns["TensorDataset"]

np.random.seed(20261008)
xb = ns["gen_unlabeled"](16, 128, 1.5).to(device)

def one_step(kind):
    torch.manual_seed(1701)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1701)
    enc = SignalEncoder().to(device)
    dec = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128)).to(device)
    mask_tok = nn.Parameter(torch.randn(1, 1, 64, device=device) * 0.02)
    params = list(enc.parameters()) + list(dec.parameters()) + [mask_tok]
    initial = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=1e-3)

    tok = enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1, 2))
    n_mask = max(1, int(round(tok.shape[1] * 0.5)))
    random_order = torch.rand(xb.shape[0], tok.shape[1], 1, device=device).argsort(dim=1)
    mask = torch.zeros(xb.shape[0], tok.shape[1], 1, dtype=torch.bool, device=device)
    mask.scatter_(1, random_order[:, :n_mask, :], True)
    masked = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
    recon = dec(enc.trf(masked).mean(dim=1))
    r = recon - xb
    loss = r.abs().mean() if kind == "l1" else r.abs().pow(1.0).mean()
    opt.zero_grad()
    loss.backward()
    grads = [p.grad.detach().clone() for p in params]
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    opt.step()
    updated = [p.detach().clone() for p in params]
    return {
        "loss": float(loss.detach().cpu()),
        "mask": mask.detach().cpu(),
        "initial": initial,
        "gradients": grads,
        "updated": updated,
    }

a, b = one_step("l1"), one_step("floc1")
def maxdiff(xs, ys):
    return max(float((x-y).abs().max().cpu()) for x, y in zip(xs, ys))

# Also compare one full archived pretraining epoch on the same 10k-record corpus.
ns["CONFIG"]["pt_epochs"] = 1
np.random.seed(1701)
x_pt = ns["gen_unlabeled"](10000, 128, 1.5)
def one_epoch(kind):
    torch.manual_seed(2701)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2701)
    loader = DataLoader(
        TensorDataset(x_pt), batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(2702),
    )
    return ns["pretrain_msm"](loader, kind, epochs=1)

c, d = one_epoch("l1"), one_epoch("floc_1.0")
keys = sorted(k for k in c if not k.startswith("_"))
epoch_encoder_maxdiff = maxdiff([c[k] for k in keys], [d[k] for k in keys])

report = {
    "runner_sha256": source_hash,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "input_identical": True,
    "mask_identical": bool(torch.equal(a["mask"], b["mask"])),
    "loss_l1": a["loss"],
    "loss_floc_p1": b["loss"],
    "loss_abs_difference": abs(a["loss"]-b["loss"]),
    "initial_parameter_max_abs_difference": maxdiff(a["initial"], b["initial"]),
    "gradient_max_abs_difference": maxdiff(a["gradients"], b["gradients"]),
    "post_adam_parameter_max_abs_difference": maxdiff(a["updated"], b["updated"]),
    "one_full_epoch_encoder_max_abs_difference": epoch_encoder_maxdiff,
    "one_epoch_trajectory_loss_abs_difference": abs(c["_pretraining_trajectory"][0]["mean_train_loss"]-d["_pretraining_trajectory"][0]["mean_train_loss"]),
    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
}
out = ROOT / "results" / "loss_equivalence_probe.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
