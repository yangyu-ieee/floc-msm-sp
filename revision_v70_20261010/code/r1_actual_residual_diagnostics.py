"""R1 mechanism audit on residuals from trained masked-signal models.

Uses the corrected symmetric CMS generator and the manuscript's 1-D CNN plus
4-layer Transformer architecture. It reports held-out reconstruction residual,
score-function, and parameter-gradient diagnostics; it is not a downstream
classification comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_noise import draw_cms_variates, from_cms_variates


ROOT = Path(__file__).resolve().parents[1]
LOSSES = ("mse", "l1", "huber", "charbonnier", "floc_1.2")
SEQ_LEN = 128


def clean_unlabeled(n: int, seed: int):
    rng = np.random.RandomState(seed)
    t = np.arange(SEQ_LEN)
    x = np.zeros((n, SEQ_LEN), dtype=np.float64)
    for i in range(n):
        st = rng.randint(0, 5)
        if st == 0:
            sig = np.sin(2 * np.pi * rng.uniform(0.01, 0.2) * t) + 0.5 * np.sin(
                2 * np.pi * rng.uniform(0.01, 0.15) * t
            )
        elif st == 1:
            f0, f1 = rng.uniform(0.01, 0.1, 2)
            sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / SEQ_LEN) * t)
        elif st == 2:
            period = rng.randint(10, 40)
            sig = np.zeros(SEQ_LEN)
            for j in range(0, SEQ_LEN, period):
                sig[j : j + 2] = rng.uniform(0.5, 1.5)
        elif st == 3:
            tau = rng.uniform(20, 80)
            sig = np.sin(2 * np.pi * rng.uniform(0.03, 0.15) * t) * np.exp(-t / tau)
        else:
            sig = np.cumsum(rng.randn(SEQ_LEN) * 0.1)
        x[i] = sig + 0.02 * rng.randn(SEQ_LEN)
    return x.astype(np.float32)


def corrupted(clean, rng: np.random.RandomState, scale: float):
    u, w = draw_cms_variates(clean.shape, rng)
    noisy = clean.astype(np.float64) + scale * from_cms_variates(1.5, u, w)
    noisy /= np.max(np.abs(noisy), axis=1, keepdims=True) + 1e-8
    return noisy.astype(np.float32)


class PosEnc(nn.Module):
    def __init__(self, d=64, max_len=300):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class SignalEncoder(nn.Module):
    def __init__(self, d=64, heads=4, n_layers=4):
        super().__init__()
        self.tok = nn.Conv1d(1, d, 8, stride=4, padding=2)
        self.pe = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.trf = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x):
        return self.pe(self.tok(x.unsqueeze(1)).transpose(1, 2))


def loss_value(name, residual):
    if name == "mse":
        return residual.square().mean()
    if name == "l1":
        return residual.abs().mean()
    if name == "huber":
        return F.huber_loss(residual, torch.zeros_like(residual), delta=1.0)
    if name == "charbonnier":
        return torch.sqrt(residual.square() + 0.01**2).mean()
    if name == "floc_1.2":
        return residual.abs().pow(1.2).mean()
    raise ValueError(name)


def residual_score(name: str, r: np.ndarray):
    if name == "mse":
        return 2.0 * r
    if name == "l1":
        return np.sign(r)
    if name == "huber":
        return np.clip(r, -1.0, 1.0)
    if name == "charbonnier":
        return r / np.sqrt(r * r + 0.01**2)
    if name == "floc_1.2":
        return 1.2 * np.sign(r) * np.maximum(np.abs(r), 1e-30) ** 0.2
    raise ValueError(name)


def quantiles(x):
    a = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    probs = ((0.5, "abs_q50"), (0.9, "abs_q90"), (0.99, "abs_q99"), (0.999, "abs_q999"), (0.9999, "abs_q9999"))
    return {name: float(np.quantile(a, q)) for q, name in probs} | {
        "abs_max": float(a.max()), "finite_fraction": float(np.isfinite(a).mean())
    }


def score_summary(score):
    a = np.abs(np.asarray(score, dtype=np.float64)).ravel()
    sq = a * a
    k = max(1, int(np.ceil(0.001 * len(sq))))
    return {
        **quantiles(a),
        "sample_second_moment": float(np.mean(sq)),
        "top_0p1pct_squared_score_share": float(np.partition(sq, -k)[-k:].sum() / (sq.sum() + 1e-300)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "r1_actual_residual_diagnostics")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--pretrain-samples", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--validation-samples", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.seeds, args.pretrain_samples, args.epochs, args.validation_samples = 1, 64, 1, 50
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    config = {
        "experiment": "R1_actual_residual_score_parameter_gradient_audit",
        "generator": "symmetric CMS S1, alpha=1.5, CF exp(-|t|^alpha)",
        "normalization": "per-sequence max-abs after corruption, retained to match corrected main benchmark",
        "architecture": "Conv1d(k=8,stride=4,pad=2,d=64) + 4-layer Transformer(d=64,heads=4,ff=256,dropout=0.1); pooled linear decoder",
        "mask_ratio": 0.5, "losses": list(LOSSES), "seeds": args.seeds,
        "pretrain_samples": args.pretrain_samples, "epochs": args.epochs,
        "validation_samples": args.validation_samples, "device": str(device),
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records, trajectories = [], {}
    started = time.time()

    for seed in range(args.seeds):
        clean_train = clean_unlabeled(args.pretrain_samples, 71000 + seed)
        clean_val = clean_unlabeled(args.validation_samples, 72000 + seed)
        x_train = corrupted(clean_train, np.random.RandomState(73000 + seed), 0.15)
        x_val = corrupted(clean_val, np.random.RandomState(74000 + seed), 0.15)
        for loss_name in LOSSES:
            torch.manual_seed(75000 + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(75000 + seed)
            loader = DataLoader(
                TensorDataset(torch.from_numpy(x_train)), batch_size=128, shuffle=True,
                generator=torch.Generator().manual_seed(76000 + seed),
            )
            enc = SignalEncoder().to(device)
            dec = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, SEQ_LEN)).to(device)
            mask_tok = nn.Parameter(torch.randn(1, 1, 64, device=device) * 0.02)
            opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [mask_tok], lr=1e-3)
            epoch_losses = []
            for epoch in range(args.epochs):
                enc.train(); dec.train()
                batch_losses = []
                for (xb,) in loader:
                    xb = xb.to(device)
                    tok = enc(xb)
                    mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < 0.5
                    z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
                    recon = dec(enc.trf(z).mean(dim=1))
                    loss = loss_value(loss_name, recon - xb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss: {loss_name}, seed={seed}")
                    opt.zero_grad(set_to_none=True); loss.backward()
                    enc_grad = torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                    opt.step()
                    batch_losses.append(float(loss.detach().cpu()))
                epoch_losses.append({"epoch": epoch + 1, "mean_loss": float(np.mean(batch_losses))})
            trajectories[f"seed={seed}/loss={loss_name}"] = epoch_losses

            enc.eval(); dec.eval()
            residuals, param_grad_norms = [], []
            val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=128, shuffle=False)
            for batch_i, (xb,) in enumerate(val_loader):
                xb = xb.to(device)
                with torch.no_grad():
                    tok = enc(xb)
                    # A fixed validation masking protocol, reproducibly paired across losses.
                    gen = torch.Generator(device=device).manual_seed(77000 + seed + batch_i)
                    mask = torch.rand((xb.shape[0], tok.shape[1], 1), generator=gen, device=device) < 0.5
                    z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
                    recon = dec(enc.trf(z).mean(dim=1))
                    residuals.append((recon - xb).detach().cpu().numpy().ravel())

                # Parameter-gradient norm on held-out batches; same masked inputs and
                # initialized model state, computed independently of the optimizer.
                enc.zero_grad(set_to_none=True); dec.zero_grad(set_to_none=True)
                mask_tok.grad = None
                tok = enc(xb)
                z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
                recon = dec(enc.trf(z).mean(dim=1))
                val_loss = loss_value(loss_name, recon - xb)
                val_loss.backward()
                g2 = 0.0
                for param in list(enc.parameters()) + list(dec.parameters()) + [mask_tok]:
                    if param.grad is not None:
                        g2 += float(param.grad.detach().float().norm().cpu()) ** 2
                param_grad_norms.append(math.sqrt(g2))

            r = np.concatenate(residuals)
            psi = residual_score(loss_name, r)
            records.append({
                "seed": seed, "loss": loss_name, "n_residuals": int(r.size),
                "residual": quantiles(r), "score": score_summary(psi),
                "heldout_parameter_gradient_norm": {
                    "mean": float(np.mean(param_grad_norms)),
                    "std_ddof0": float(np.std(param_grad_norms)),
                    "per_batch": param_grad_norms,
                },
                "training_loss_initial": epoch_losses[0]["mean_loss"],
                "training_loss_final": epoch_losses[-1]["mean_loss"],
            })
            print(f"DONE seed={seed} loss={loss_name} final_train={epoch_losses[-1]['mean_loss']:.6g} grad_norm={np.mean(param_grad_norms):.6g}", flush=True)
            del enc, dec, mask_tok, opt, loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {}
    for loss_name in LOSSES:
        rows = [row for row in records if row["loss"] == loss_name]
        summary[loss_name] = {}
        for metric, fields in {
            "residual": ["abs_q90", "abs_q99", "abs_q999", "abs_max"],
            "score": ["abs_q90", "abs_q99", "abs_q999", "sample_second_moment", "top_0p1pct_squared_score_share"],
        }.items():
            summary[loss_name][metric] = {
                field: {
                    "mean": float(np.mean([row[metric][field] for row in rows])),
                    "std_ddof0": float(np.std([row[metric][field] for row in rows])),
                }
                for field in fields
            }
        summary[loss_name]["heldout_parameter_gradient_norm"] = {
            "mean": float(np.mean([row["heldout_parameter_gradient_norm"]["mean"] for row in rows])),
            "std_ddof0": float(np.std([row["heldout_parameter_gradient_norm"]["mean"] for row in rows])),
        }

    result = {
        "scope": "Actual held-out reconstruction residuals and loss-score/parameter-gradient diagnostics from trained models; not downstream accuracy or a representation-stability theorem.",
        "config": config, "summary": summary, "records": records,
        "pretraining_trajectories": trajectories,
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps({"elapsed_seconds": time.time() - started}, indent=2), encoding="utf-8"
    )
    print(f"COMPLETE output={args.output_dir} elapsed={time.time()-started:.1f}s", flush=True)
    for loss_name in LOSSES:
        print(loss_name, json.dumps(summary[loss_name], separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
