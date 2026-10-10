"""Finite-run training-dynamics diagnostic for the alpha=1 assumption mismatch.

This is a separate, source-closed diagnostic; it does not alter the prior
ten-seed downstream result or the canonical manuscript. It records fixed
held-out masked-reconstruction loss and pre-clipping batch gradient norms
over the 20-epoch pretraining schedule used in the targeted alpha=1 study.
The output supports descriptive finite-run statements only, not convergence
guarantees or population-law claims.
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
from torch.utils.data import DataLoader, TensorDataset

import targeted_alpha10_fullseed as base


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "logs" / "r3_2_alpha1_training_dynamics_2026-10-08"
LOSSES = ["mse", "l1", "charbonnier", "floc_1.2"]
LOSS_SEED_OFFSET = {"mse": 11, "l1": 23, "charbonnier": 37, "floc_1.2": 103}


def heldout_loss(encoder, decoder, mask_token, x_eval, masks, loss_name, device, batch_size):
    encoder.eval()
    decoder.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x_eval), batch_size):
            xb = x_eval[start : start + batch_size].to(device)
            mask = masks[start : start + batch_size].to(device)
            tok = encoder.pe(encoder.tok(xb.unsqueeze(1)).transpose(1, 2))
            masked = torch.where(mask, mask_token.expand(xb.shape[0], tok.shape[1], -1), tok)
            recon = decoder(encoder.trf(masked).mean(dim=1))
            values.append(float(base.recon_loss(recon - xb, loss_name).cpu()))
    return float(np.mean(values))


def evaluate_training_dynamics(loss_name, x_train, x_eval, masks, args, device, run_seed):
    base.set_seed(run_seed)
    train_loader = DataLoader(
        TensorDataset(x_train), batch_size=args.batch_size, shuffle=True
    )
    encoder = base.SignalEncoder(args.d_model, args.heads, args.layers).to(device)
    decoder = nn.Sequential(
        nn.Linear(args.d_model, 128), nn.ReLU(), nn.Linear(128, args.seq_len)
    ).to(device)
    mask_token = nn.Parameter(torch.randn(1, 1, args.d_model, device=device) * 0.02)
    params = list(encoder.parameters()) + list(decoder.parameters()) + [mask_token]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    rows = [{
        "epoch": 0,
        "mean_train_loss": None,
        "heldout_fixed_mask_loss": heldout_loss(
            encoder, decoder, mask_token, x_eval, masks, loss_name, device, args.batch_size
        ),
        "grad_norm_median": None,
        "grad_norm_q90": None,
        "grad_norm_max": None,
        "nonfinite_steps": 0,
    }]

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        decoder.train()
        epoch_losses, grad_norms = [], []
        nonfinite_steps = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            tok = encoder.pe(encoder.tok(xb.unsqueeze(1)).transpose(1, 2))
            mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < args.mask_ratio
            masked = torch.where(mask, mask_token.expand(xb.shape[0], tok.shape[1], -1), tok)
            recon = decoder(encoder.trf(masked).mean(dim=1))
            loss = base.recon_loss(recon - xb, loss_name)
            if not torch.isfinite(loss):
                nonfinite_steps += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            norm_value = float(grad_norm.detach().cpu())
            if not math.isfinite(norm_value):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            grad_norms.append(norm_value)

        if not epoch_losses:
            raise RuntimeError(f"all updates were non-finite: {loss_name=} {run_seed=} {epoch=}")
        rows.append({
            "epoch": epoch,
            "mean_train_loss": float(np.mean(epoch_losses)),
            "heldout_fixed_mask_loss": heldout_loss(
                encoder, decoder, mask_token, x_eval, masks, loss_name, device, args.batch_size
            ),
            "grad_norm_median": float(np.median(grad_norms)),
            "grad_norm_q90": float(np.quantile(grad_norms, 0.9)),
            "grad_norm_max": float(np.max(grad_norms)),
            "nonfinite_steps": nonfinite_steps,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--pt-samples", type=int, default=10000)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--losses", default=",".join(LOSSES))
    args = parser.parse_args()
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(base.__file__).resolve()
    config = vars(args) | {
        "losses": losses,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_runner": str(base_path),
        "base_runner_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "interpretation": "finite-run descriptive diagnostics; not a convergence guarantee",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = {}
    start_time = time.time()

    for seed in range(args.seeds):
        # Share each seed's generated train/evaluation signals across losses;
        # reseed Torch separately for each model initialization and mask stream.
        np.random.seed(50000 + seed)
        x_train = base.gen_unlabeled(args.pt_samples, args.seq_len, args.alpha)
        x_eval = base.gen_unlabeled(args.eval_samples, args.seq_len, args.alpha)
        mask_gen = torch.Generator(device="cpu").manual_seed(80000 + seed)
        mask_count = max(1, round((args.seq_len // 4) * args.mask_ratio))
        mask_order = torch.rand(args.eval_samples, args.seq_len // 4, generator=mask_gen).argsort(dim=1)
        masks = torch.zeros(args.eval_samples, args.seq_len // 4, 1, dtype=torch.bool)
        masks.scatter_(1, mask_order[:, :mask_count].unsqueeze(-1), True)

        for loss_name in losses:
            print(f"seed={seed} loss={loss_name}", flush=True)
            seed_rows = evaluate_training_dynamics(
                loss_name, x_train, x_eval, masks, args, device,
                seed * 1000 + LOSS_SEED_OFFSET.get(loss_name, 199),
            )
            results[f"seed={seed}/loss={loss_name}"] = seed_rows
            (out / "trajectories.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    config["elapsed_seconds"] = time.time() - start_time
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved {len(results)} trajectories to {out}", flush=True)


if __name__ == "__main__":
    main()
