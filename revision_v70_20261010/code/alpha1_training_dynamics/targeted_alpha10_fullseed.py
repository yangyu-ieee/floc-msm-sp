"""Targeted high-seed alpha=1.0 robust-loss check.

Purpose
-------
The existing corrected multi-alpha experiment is useful as a regime map, but
its tiny FLOC-vs-L1/Charbonnier margins are based on a shared pre-trained
encoder per loss and five downstream seeds. This script targets the main
reviewer risk directly:

  Does the FLOC family remain competitive in the extreme heavy-tail regime
  when every seed re-generates pre-training data and re-trains the MSM encoder?

It intentionally focuses on alpha=1.0 and the strongest relevant robust losses.
Do not use this script to claim universal FLOC dominance.
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
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = PROJECT / "logs" / "targeted_alpha10_fullseed_20260612"
LOSS_SEED_OFFSET = {
    "mse": 11,
    "l1": 23,
    "charbonnier": 37,
    "floc_1.0": 101,
    "floc_1.2": 103,
    "floc_1.5": 107,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PosEnc(nn.Module):
    def __init__(self, d: int, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SignalEncoder(nn.Module):
    def __init__(self, d: int = 64, heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.tok = nn.Conv1d(1, d, 8, stride=4, padding=2)
        self.pe = PosEnc(d, 300)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.trf = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, x: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = self.pe(self.tok(x.unsqueeze(1)).transpose(1, 2))
        if cls is not None:
            x = torch.cat([cls, x], dim=1)
        return self.trf(x)


def alpha_stable_noise(seq_len: int, alpha: float, scale: float) -> np.ndarray:
    u = np.random.uniform(-np.pi / 2, np.pi / 2, seq_len)
    w = np.random.exponential(1, seq_len)
    noise = (
        np.sin(alpha * u)
        * (np.cos(u) / w) ** ((1 - alpha) / alpha)
        / (np.cos(u) ** (1 / alpha))
    )
    return noise.astype(np.float32) * scale


def gen_unlabeled(n: int, seq_len: int = 128, alpha: float = 1.0) -> torch.Tensor:
    t = np.arange(seq_len)
    x = np.zeros((n, seq_len), dtype=np.float32)
    for i in range(n):
        st = np.random.randint(0, 5)
        if st == 0:
            sig = np.sin(2 * np.pi * np.random.uniform(0.01, 0.2) * t) + 0.5 * np.sin(
                2 * np.pi * np.random.uniform(0.01, 0.15) * t
            )
        elif st == 1:
            f0, f1 = np.random.uniform(0.01, 0.1, 2)
            sig = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / seq_len) * t)
        elif st == 2:
            period = np.random.randint(10, 40)
            sig = np.zeros(seq_len)
            for j in range(0, seq_len, period):
                sig[j : j + 2] = np.random.uniform(0.5, 1.5)
        elif st == 3:
            tau = np.random.uniform(20, 80)
            sig = np.sin(2 * np.pi * np.random.uniform(0.03, 0.15) * t) * np.exp(-t / tau)
        else:
            sig = np.cumsum(np.random.randn(seq_len) * 0.1)
        sig += 0.02 * np.random.randn(seq_len)
        sig += alpha_stable_noise(seq_len, alpha, 0.15)
        x[i] = sig.astype(np.float32)
    x /= np.abs(x).max(axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(x)


def gen_cls(n: int, seq_len: int = 128, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    t = np.arange(seq_len)
    x = np.zeros((n, seq_len), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = i % 5
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
        sig += 0.05 * np.random.randn(seq_len)
        sig += alpha_stable_noise(seq_len, alpha, 0.3)
        x[i] = sig.astype(np.float32)
        y[i] = cls
    x /= np.abs(x).max(axis=1, keepdims=True) + 1e-8
    return torch.from_numpy(x), torch.from_numpy(y)


def recon_loss(residual: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return residual.pow(2).mean()
    if loss_name == "l1":
        return residual.abs().mean()
    if loss_name == "charbonnier":
        return torch.sqrt(residual.pow(2) + 0.01**2).mean()
    if loss_name.startswith("floc_"):
        p = float(loss_name.split("_")[1])
        return residual.abs().pow(p).mean()
    raise ValueError(loss_name)


def pretrain(loss_name: str, args: argparse.Namespace, device: str) -> dict[str, torch.Tensor]:
    x_pt = gen_unlabeled(args.pt_samples, args.seq_len, args.alpha)
    loader = DataLoader(TensorDataset(x_pt), batch_size=args.batch_size, shuffle=True)
    enc = SignalEncoder(args.d_model, args.heads, args.layers).to(device)
    dec = nn.Sequential(nn.Linear(args.d_model, 128), nn.ReLU(), nn.Linear(128, args.seq_len)).to(device)
    mask_tok = nn.Parameter(torch.randn(1, 1, args.d_model, device=device) * 0.02)
    params = list(enc.parameters()) + list(dec.parameters()) + [mask_tok]
    opt = torch.optim.Adam(params, lr=args.lr)

    for _ in range(args.pt_epochs):
        enc.train()
        dec.train()
        for (xb,) in loader:
            xb = xb.to(device)
            tok = enc.pe(enc.tok(xb.unsqueeze(1)).transpose(1, 2))
            mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < args.mask_ratio
            masked = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
            recon = dec(enc.trf(masked).mean(dim=1))
            loss = recon_loss(recon - xb, loss_name)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def eval_downstream(
    weights: dict[str, torch.Tensor] | None,
    n_train: int,
    args: argparse.Namespace,
    device: str,
    seed_offset: int,
) -> float:
    set_seed(seed_offset)
    enc = SignalEncoder(args.d_model, args.heads, args.layers).to(device)
    if weights is not None:
        enc.load_state_dict({k: v.to(device) for k, v in weights.items()})
    cls_tok = nn.Parameter(torch.randn(1, 1, args.d_model, device=device) * 0.02)
    head = nn.Sequential(nn.Linear(args.d_model, 32), nn.ReLU(), nn.Linear(32, 5)).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + [cls_tok] + list(head.parameters()), lr=args.lr)

    xtr, ytr = gen_cls(n_train, args.seq_len, args.alpha)
    xte, yte = gen_cls(args.n_test, args.seq_len, args.alpha)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=min(args.ft_batch_size, n_train), shuffle=True)
    if weights is None:
        epochs = args.scratch_epochs
    else:
        epochs = args.ft_epochs if n_train <= 200 else args.ft_epochs_large
    for _ in range(epochs):
        enc.train()
        head.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            cls = cls_tok.expand(xb.shape[0], 1, -1)
            logits = head(enc(xb, cls=cls)[:, 0, :])
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()

    enc.eval()
    head.eval()
    with torch.no_grad():
        cls = cls_tok.expand(args.n_test, 1, -1)
        pred = head(enc(xte.to(device), cls=cls)[:, 0, :]).cpu().argmax(1)
    return float((pred == yte).float().mean().item())


def paired(a: list[float], b: list[float]) -> dict:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    diff = aa - bb
    t, p = stats.ttest_rel(aa, bb)
    return {
        "diff": float(diff.mean()),
        "p_value": float(p),
        "wins": int((diff > 0).sum()),
        "n": int(len(diff)),
    }


def summarize(results: dict, losses: list[str], n_labeled: list[int]) -> dict:
    out = {"by_condition": {}, "comparisons": {}, "win_count": {loss: 0 for loss in losses}}
    floc_losses = [loss for loss in losses if loss.startswith("floc_")]
    for n in n_labeled:
        means = {}
        for loss in losses:
            vals = results[f"loss={loss}/n={n}"]["seeds"]
            means[loss] = float(np.mean(vals))
            results[f"loss={loss}/n={n}"]["mean"] = means[loss]
            results[f"loss={loss}/n={n}"]["std"] = float(np.std(vals, ddof=0))
        best = max(means, key=means.get)
        out["win_count"][best] += 1
        best_floc = max(floc_losses, key=lambda loss: means[loss])
        out["by_condition"][str(n)] = {
            "best_loss": best,
            "best_mean": means[best],
            "best_floc": best_floc,
            "best_floc_mean": means[best_floc],
        }
        out["comparisons"][str(n)] = {}
        for baseline in ["mse", "l1", "charbonnier"]:
            if baseline in losses:
                out["comparisons"][str(n)][f"best_floc_vs_{baseline}"] = paired(
                    results[f"loss={best_floc}/n={n}"]["seeds"],
                    results[f"loss={baseline}/n={n}"]["seeds"],
                )
    return out


def save_state(log_dir: Path, config: dict, results: dict, summary: dict, seed_details: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (log_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (log_dir / "seed_details.json").write_text(json.dumps(seed_details, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-labeled", type=str, default="50,100,200,500")
    parser.add_argument("--losses", type=str, default="mse,l1,charbonnier,floc_1.0,floc_1.2,floc_1.5")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--pt-samples", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--pt-epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=30)
    parser.add_argument("--ft-epochs-large", type=int, default=50)
    parser.add_argument("--scratch-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ft-batch-size", type=int, default=32)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    n_labeled = [int(x.strip()) for x in args.n_labeled.split(",") if x.strip()]
    log_dir = Path(args.log_dir)
    t0 = time.time()

    config = vars(args).copy()
    config["device"] = device
    config["losses"] = losses
    config["n_labeled"] = n_labeled
    config["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["python"] = sys.version
    config["platform"] = platform.platform()
    config["torch"] = torch.__version__
    config["numpy"] = np.__version__
    config["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    results = {f"loss={loss}/n={n}": {"seeds": []} for loss in losses for n in n_labeled}
    seed_details: list[dict] = []

    print(f"Device={device} | alpha={args.alpha} | seeds={args.seeds}", flush=True)
    print(f"Losses={losses} | n_labeled={n_labeled}", flush=True)
    for seed in range(args.seeds):
        print(f"\nSeed {seed}", flush=True)
        weights = {}
        for loss in losses:
            print(f"  pretrain {loss}", flush=True)
            set_seed(seed * 1000 + LOSS_SEED_OFFSET.get(loss, 199))
            weights[loss] = pretrain(loss, args, device)

        for n in n_labeled:
            for loss in losses:
                acc = eval_downstream(weights[loss], n, args, device, seed * 10000 + n * 37 + LOSS_SEED_OFFSET.get(loss, 199))
                results[f"loss={loss}/n={n}"]["seeds"].append(acc)
                seed_details.append({"seed": seed, "loss": loss, "n_labeled": n, "accuracy": acc})
                print(f"  n={n:<3} {loss:<12} acc={acc:.4f}", flush=True)

        summary = summarize(results, losses, n_labeled)
        save_state(log_dir, config, results, summary, seed_details)
        print(f"  partial saved after seed {seed}: {log_dir}", flush=True)

    summary = summarize(results, losses, n_labeled)
    summary["elapsed_sec"] = time.time() - t0
    save_state(log_dir, config, results, summary, seed_details)

    print("\nSummary:", flush=True)
    for n in n_labeled:
        line = f"  n={n}:"
        for loss in losses:
            item = results[f"loss={loss}/n={n}"]
            line += f" {loss}={item['mean']:.3f}+-{item['std']:.3f}"
        print(line, flush=True)
    print(json.dumps(summary["by_condition"], indent=2), flush=True)
    print(f"Done | {log_dir} | elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
