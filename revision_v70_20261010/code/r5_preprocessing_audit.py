"""R5/G0 paired audit: sampler correction versus post-noise normalization.

Arms:
  submitted_legacy_maxabs: submitted transform + per-sequence max-abs scaling;
  cms_maxabs: correct CMS S_alpha S draw + per-sequence max-abs scaling;
  cms_clean_global_rms: correct CMS draw, scale clean data by a training-only
      global RMS, then add noise without per-sequence noisy-data rescaling.

The experiment follows the submitted 1-D CNN/Transformer masked-pretraining
and downstream fine-tuning protocol. All conditions share clean signals and
the same underlying U/W random variates. Outputs are written only to the
revision folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_noise import draw_cms_variates, from_cms_variates, from_submitted_legacy_variates


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "r5_preprocessing_audit"
ALPHA = 1.5
SEQ_LEN = 128
LOSSES = ("mse", "l1", "charbonnier", "floc_1.2")
ARMS = ("submitted_legacy_maxabs", "cms_maxabs", "cms_clean_global_rms")


def seed_torch(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_unlabeled(n: int, seed: int, seq_len: int = SEQ_LEN):
    rng = np.random.RandomState(seed)
    t = np.arange(seq_len)
    x = np.zeros((n, seq_len), dtype=np.float64)
    for i in range(n):
        st = rng.randint(0, 5)
        if st == 0:
            sig = np.sin(2 * np.pi * rng.uniform(0.01, 0.2) * t) + 0.5 * np.sin(
                2 * np.pi * rng.uniform(0.01, 0.15) * t
            )
        elif st == 1:
            f0, f1 = rng.uniform(0.01, 0.1, 2)
            sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / seq_len) * t)
        elif st == 2:
            period = rng.randint(10, 40)
            sig = np.zeros(seq_len)
            for j in range(0, seq_len, period):
                sig[j : j + 2] = rng.uniform(0.5, 1.5)
        elif st == 3:
            tau = rng.uniform(20, 80)
            sig = np.sin(2 * np.pi * rng.uniform(0.03, 0.15) * t) * np.exp(-t / tau)
        else:
            sig = np.cumsum(rng.randn(seq_len) * 0.1)
        sig = sig + 0.02 * rng.randn(seq_len)
        x[i] = sig
    return x.astype(np.float32)


def clean_classification(n: int, seed: int, seq_len: int = SEQ_LEN):
    rng = np.random.RandomState(seed)
    t = np.arange(seq_len)
    x = np.zeros((n, seq_len), dtype=np.float64)
    y = np.arange(n, dtype=np.int64) % 5
    for i, cls in enumerate(y):
        if cls == 0:
            sig = np.sin(2 * np.pi * 0.02 * t) + 0.3 * np.sin(2 * np.pi * 0.05 * t)
        elif cls == 1:
            sig = np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)
        elif cls == 2:
            sig = np.sign(np.sin(2 * np.pi * 0.03 * t))
        elif cls == 3:
            sig = 2.0 * (t % 20) / 20.0 - 1.0
        else:
            sig = np.zeros(seq_len)
            for j in range(0, seq_len, 25):
                sig[j : j + 2] = 1.0
        x[i] = sig + 0.05 * rng.randn(seq_len)
    return x.astype(np.float32), y


def noise_variates(shape, seed: int):
    return draw_cms_variates(shape, np.random.RandomState(seed))


def make_observations(clean, variates, arm: str, noise_scale: float, train_scale=None):
    u, w = variates
    if arm == "submitted_legacy_maxabs":
        noise = from_submitted_legacy_variates(ALPHA, u, w)
        observed = clean.astype(np.float64) + noise_scale * noise
        observed /= np.max(np.abs(observed), axis=1, keepdims=True) + 1e-8
    elif arm == "cms_maxabs":
        noise = from_cms_variates(ALPHA, u, w)
        observed = clean.astype(np.float64) + noise_scale * noise
        observed /= np.max(np.abs(observed), axis=1, keepdims=True) + 1e-8
    elif arm == "cms_clean_global_rms":
        if train_scale is None or not np.isfinite(train_scale) or train_scale <= 0.0:
            raise ValueError("A positive training-only clean RMS is required")
        noise = from_cms_variates(ALPHA, u, w)
        observed = clean.astype(np.float64) / train_scale + noise_scale * noise
    else:
        raise ValueError(f"Unknown arm: {arm}")
    if not np.isfinite(observed).all():
        raise FloatingPointError(f"Non-finite observations in arm {arm}")
    return observed.astype(np.float32)


def quantile_summary(x):
    ax = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    q = np.quantile(ax, [0.5, 0.9, 0.99, 0.999, 0.9999])
    return {
        "abs_q50": float(q[0]),
        "abs_q90": float(q[1]),
        "abs_q99": float(q[2]),
        "abs_q999": float(q[3]),
        "abs_q9999": float(q[4]),
        "abs_max": float(ax.max()),
        "finite_fraction": float(np.isfinite(ax).mean()),
    }


class PosEnc(nn.Module):
    def __init__(self, d, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        dv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * dv)
        pe[:, 1::2] = torch.cos(pos * dv)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SignalEncoder(nn.Module):
    def __init__(self, d=64, heads=4, n_layers=4):
        super().__init__()
        self.tok = nn.Conv1d(1, d, 8, stride=4, padding=2)
        self.pe = PosEnc(d, 300)
        layer = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.trf = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x, cls=None):
        if x.dim() == 2:
            x = self.pe(self.tok(x.unsqueeze(1)).transpose(1, 2))
        if cls is not None:
            x = torch.cat([cls, x], dim=1)
        return self.trf(x)


def make_loss(name, residual):
    if name == "mse":
        return residual.square().mean()
    if name == "l1":
        return residual.abs().mean()
    if name == "charbonnier":
        return torch.sqrt(residual.square() + 0.01**2).mean()
    if name == "floc_1.2":
        return residual.abs().pow(1.2).mean()
    raise ValueError(name)


def pretrain(x_np, loss_name, epochs, device, seed):
    seed_torch(seed)
    x = torch.from_numpy(x_np)
    loader_gen = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(x), batch_size=128, shuffle=True, generator=loader_gen)
    enc = SignalEncoder().to(device)
    dec = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, SEQ_LEN)).to(device)
    mask_tok = nn.Parameter(torch.randn(1, 1, 64, device=device) * 0.02)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [mask_tok], lr=1e-3)
    trajectory = []
    for epoch in range(epochs):
        enc.train()
        dec.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            tok = enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1, 2))
            mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < 0.5
            masked = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
            recon = dec(enc.trf(masked).mean(dim=1))
            loss = make_loss(loss_name, recon - xb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite pretraining loss: arm/loss={loss_name}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        trajectory.append({"epoch": epoch + 1, "train_loss_mean": float(np.mean(losses))})
    state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
    del enc, dec, opt, mask_tok, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state, trajectory


def finetune(state, x_train, y_train, x_test, y_test, epochs, device, seed):
    seed_torch(seed)
    enc = SignalEncoder().to(device)
    enc.load_state_dict(state)
    cls_tok = nn.Parameter(torch.randn(1, 1, 64, device=device) * 0.02)
    head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 5)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + [cls_tok] + list(head.parameters()), lr=1e-3)
    gen = torch.Generator().manual_seed(seed + 11)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=min(32, len(x_train)),
        shuffle=True,
        generator=gen,
    )
    enc.train()
    head.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            c = cls_tok.expand(xb.shape[0], 1, -1)
            logits = head(enc(xb, cls=c)[:, 0, :])
            loss = F.cross_entropy(logits, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite fine-tuning loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
    enc.eval()
    head.eval()
    with torch.no_grad():
        c = cls_tok.expand(len(x_test), 1, -1)
        pred = head(enc(torch.from_numpy(x_test).to(device), cls=c)[:, 0, :]).argmax(1).cpu()
    acc = float((pred == torch.from_numpy(y_test)).float().mean())
    del enc, head, cls_tok, opt, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc


def build_downstream_arm(arm, shot, seed, n_test, clean_train, train_vars, clean_test, test_vars):
    if arm == "cms_clean_global_rms":
        train_scale = float(np.sqrt(np.mean(clean_train.astype(np.float64) ** 2)))
    else:
        train_scale = None
    x_train = make_observations(clean_train, train_vars, arm, 0.30, train_scale)
    x_test = make_observations(clean_test, test_vars, arm, 0.30, train_scale)
    return x_train, x_test, train_scale


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain-samples", type=int, default=10000)
    p.add_argument("--pretrain-epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--shots", type=int, nargs="+", default=[100, 200])
    p.add_argument("--test-size", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="one tiny end-to-end run for pipeline validation")
    p.add_argument("--output-dir", type=Path, default=OUT)
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.pretrain_samples = 64
        args.pretrain_epochs = 1
        args.seeds = 1
        args.shots = [10]
        args.test_size = 50
    device = torch.device(args.device)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    script_path = Path(__file__).resolve()
    config = {
        "experiment": "G0_R5_sampler_and_normalization_audit",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "alpha": ALPHA,
        "sequence_length": SEQ_LEN,
        "arms": list(ARMS),
        "losses": list(LOSSES),
        "pretrain_samples": args.pretrain_samples,
        "pretrain_epochs": args.pretrain_epochs,
        "fine_tune_epochs": 30,
        "seeds": args.seeds,
        "shots": args.shots,
        "test_size": args.test_size,
        "mask_ratio": 0.5,
        "device": str(device),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": sha256(script_path),
        "preprocessing": {
            "submitted_legacy_maxabs": "noise added at submitted scale, then each noisy sequence divided by its own max absolute value",
            "cms_maxabs": "verified CMS S_alpha S noise at submitted scale, then each noisy sequence divided by its own max absolute value",
            "cms_clean_global_rms": "clean signals divided by RMS estimated only from the corresponding clean training pool, then CMS noise added; no noisy-sample rescaling",
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("experiment", "alpha", "arms", "losses", "pretrain_samples", "pretrain_epochs", "seeds", "shots", "device", "script_sha256")}, indent=2), flush=True)

    clean_pt = clean_unlabeled(args.pretrain_samples, seed=13001)
    pt_vars = noise_variates(clean_pt.shape, seed=13002)
    pt_train_scale = float(np.sqrt(np.mean(clean_pt.astype(np.float64) ** 2)))
    pt_data = {}
    input_diagnostics = {"pretraining": {"clean_train_global_rms": pt_train_scale}}
    for arm in ARMS:
        xp = make_observations(clean_pt, pt_vars, arm, 0.15, pt_train_scale if arm == "cms_clean_global_rms" else None)
        pt_data[arm] = xp
        input_diagnostics["pretraining"][arm] = quantile_summary(xp)

    print("Pretraining data generated; fitting encoder per arm/loss...", flush=True)
    states = {}
    trajectories = {}
    for arm_i, arm in enumerate(ARMS):
        states[arm] = {}
        trajectories[arm] = {}
        for loss_i, loss_name in enumerate(LOSSES):
            seed = 5000
            print(f"PRETRAIN arm={arm} loss={loss_name}", flush=True)
            state, traj = pretrain(pt_data[arm], loss_name, args.pretrain_epochs, device, seed)
            states[arm][loss_name] = state
            trajectories[arm][loss_name] = traj

    results = {arm: {loss: {str(n): {"seeds": []} for n in args.shots} for loss in LOSSES} for arm in ARMS}
    for shot in args.shots:
        for seed in range(args.seeds):
            clean_train, y_train = clean_classification(shot, seed=20000 + seed)
            clean_test, y_test = clean_classification(args.test_size, seed=30000 + seed)
            train_vars = noise_variates(clean_train.shape, seed=40000 + seed)
            test_vars = noise_variates(clean_test.shape, seed=50000 + seed)
            prepared = {}
            scales = {}
            for arm in ARMS:
                xtr, xte, scale = build_downstream_arm(
                    arm, shot, seed, args.test_size, clean_train, train_vars, clean_test, test_vars
                )
                prepared[arm] = (xtr, xte)
                scales[arm] = scale
                input_diagnostics.setdefault("downstream", {}).setdefault(str(shot), {}).setdefault(str(seed), {})[arm] = {
                    "clean_train_global_rms": scales[arm],
                    "train_input": quantile_summary(xtr),
                    "test_input": quantile_summary(xte),
                }
            for arm in ARMS:
                xtr, xte = prepared[arm]
                for loss_name in LOSSES:
                    acc = finetune(
                        states[arm][loss_name], xtr, y_train, xte, y_test,
                        epochs=30, device=device, seed=60000 + seed,
                    )
                    results[arm][loss_name][str(shot)]["seeds"].append(acc)
                    print(
                        f"EVAL arm={arm} loss={loss_name} shots={shot} seed={seed} acc={acc:.4f}",
                        flush=True,
                    )

    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for loss_name in LOSSES:
            summary[arm][loss_name] = {}
            for shot in args.shots:
                vals = np.asarray(results[arm][loss_name][str(shot)]["seeds"], dtype=np.float64)
                results[arm][loss_name][str(shot)].update({
                    "mean": float(vals.mean()),
                    "std_ddof0": float(vals.std(ddof=0)),
                    "n_seeds": int(vals.size),
                })
                summary[arm][loss_name][str(shot)] = {
                    "mean": float(vals.mean()),
                    "std_ddof0": float(vals.std(ddof=0)),
                    "seeds": [float(v) for v in vals],
                }
    elapsed = time.time() - started
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "input_diagnostics.json").write_text(json.dumps(input_diagnostics, indent=2), encoding="utf-8")
    (out_dir / "pretraining_trajectories.json").write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(json.dumps({"elapsed_seconds": elapsed}, indent=2), encoding="utf-8")
    print(f"COMPLETE elapsed_seconds={elapsed:.1f} output={out_dir}", flush=True)
    for arm in ARMS:
        for loss_name in LOSSES:
            cells = {str(n): round(summary[arm][loss_name][str(n)]["mean"], 4) for n in args.shots}
            print(f"SUMMARY arm={arm} loss={loss_name} means={cells}", flush=True)


if __name__ == "__main__":
    main()
