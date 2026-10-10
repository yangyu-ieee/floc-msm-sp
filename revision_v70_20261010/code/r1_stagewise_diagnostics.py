"""Stage-wise held-out residual and loss-score audit for Reviewer 1/3."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import r1_actual_residual_diagnostics as base


ROOT = Path(__file__).resolve().parents[1]
STAGES = (1, 5, 10, 20)


def evaluate(enc, dec, mask_tok, x_val, seed, loss_name, device):
    enc.eval()
    dec.eval()
    residuals, param_grad_norms = [], []
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val)), batch_size=128, shuffle=False)
    for batch_i, (xb,) in enumerate(val_loader):
        xb = xb.to(device)
        with torch.no_grad():
            tok = enc(xb)
            gen = torch.Generator(device=device).manual_seed(77000 + seed + batch_i)
            mask = torch.rand((xb.shape[0], tok.shape[1], 1), generator=gen, device=device) < 0.5
            z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
            recon = dec(enc.trf(z).mean(dim=1))
            residuals.append((recon - xb).cpu().numpy().ravel())
        enc.zero_grad(set_to_none=True)
        dec.zero_grad(set_to_none=True)
        mask_tok.grad = None
        tok = enc(xb)
        z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
        recon = dec(enc.trf(z).mean(dim=1))
        loss = base.loss_value(loss_name, recon - xb)
        loss.backward()
        grad_sq = sum(float(p.grad.detach().float().norm().cpu()) ** 2
                      for p in list(enc.parameters()) + list(dec.parameters()) + [mask_tok]
                      if p.grad is not None)
        param_grad_norms.append(float(np.sqrt(grad_sq)))
    residual = np.concatenate(residuals)
    grad = np.asarray(param_grad_norms, dtype=np.float64)
    score_abs = np.abs(base.residual_score(loss_name, residual)).astype(np.float32)
    return {
        "n_residuals": int(residual.size),
        "residual": base.quantiles(residual),
        "score": base.score_summary(score_abs),
        "score_abs_samples": score_abs,
        "parameter_gradient_norm": {
            "n_batches": len(param_grad_norms),
            "batch_values": param_grad_norms,
            "q50": float(np.quantile(grad, 0.50)),
            "q90": float(np.quantile(grad, 0.90)),
            "q99": float(np.quantile(grad, 0.99)),
            "max": float(np.max(grad)),
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / "r1_stagewise_diagnostics")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--pretrain-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.smoke:
        args.seeds, args.pretrain_samples, args.validation_samples, args.epochs = 1, 64, 50, 1
    stages = tuple(s for s in STAGES if s <= args.epochs)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "R1_R3_stagewise_actual_residual_score_diagnostics",
        "generator": "symmetric CMS S1 alpha=1.5, CF exp(-|t|^alpha)",
        "normalization": "per-sequence noisy max-abs; identical to corrected benchmark",
        "architecture": "same Conv1d + 4-layer Transformer and pooled linear decoder as prior R1 residual audit",
        "mask_ratio": 0.5, "losses": list(base.LOSSES), "seeds": args.seeds,
        "pretrain_samples": args.pretrain_samples, "validation_samples": args.validation_samples,
        "epochs": args.epochs, "stages": stages, "device": str(device),
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Descriptive held-out residual and scalar loss-score snapshots; not total network-gradient stability.",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records, trajectories, epoch20_scores = [], {}, {loss: [] for loss in base.LOSSES}
    started = time.time()
    for seed in range(args.seeds):
        x_train = base.corrupted(base.clean_unlabeled(args.pretrain_samples, 71000 + seed), np.random.RandomState(73000 + seed), 0.15)
        x_val = base.corrupted(base.clean_unlabeled(args.validation_samples, 72000 + seed), np.random.RandomState(74000 + seed), 0.15)
        for loss_name in base.LOSSES:
            torch.manual_seed(75000 + seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(75000 + seed)
            loader = DataLoader(TensorDataset(torch.from_numpy(x_train)), batch_size=128, shuffle=True,
                                generator=torch.Generator().manual_seed(76000 + seed))
            enc = base.SignalEncoder().to(device)
            dec = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, base.SEQ_LEN)).to(device)
            mask_tok = nn.Parameter(torch.randn(1, 1, 64, device=device) * 0.02)
            opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()) + [mask_tok], lr=1e-3)
            epoch_losses, snapshots = [], []
            for epoch in range(1, args.epochs + 1):
                enc.train(); dec.train()
                batch_losses = []
                for (xb,) in loader:
                    xb = xb.to(device)
                    tok = enc(xb)
                    mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < 0.5
                    z = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
                    recon = dec(enc.trf(z).mean(dim=1))
                    loss = base.loss_value(loss_name, recon - xb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss {loss_name}, seed={seed}, epoch={epoch}")
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
                    opt.step()
                    batch_losses.append(float(loss.detach().cpu()))
                epoch_losses.append({"epoch": epoch, "mean_loss": float(np.mean(batch_losses))})
                if epoch in stages:
                    snap = evaluate(enc, dec, mask_tok, x_val, seed, loss_name, device)
                    snap.update({"seed": seed, "loss": loss_name, "epoch": epoch})
                    if epoch == 20:
                        epoch20_scores[loss_name].append(snap.pop("score_abs_samples"))
                    else:
                        snap.pop("score_abs_samples")
                    records.append(snap); snapshots.append(epoch)
                    print(f"DONE seed={seed} loss={loss_name} epoch={epoch} q99={snap['residual']['abs_q99']:.5g}", flush=True)
            trajectories[f"seed={seed}/loss={loss_name}"] = epoch_losses
            del enc, dec, mask_tok, opt, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = {}
    for loss_name in base.LOSSES:
        summary[loss_name] = {}
        for epoch in stages:
            rows = [r for r in records if r["loss"] == loss_name and r["epoch"] == epoch]
            summary[loss_name][str(epoch)] = {}
            for metric, keys in (("residual", ("abs_q90", "abs_q99", "abs_q999", "abs_max")),
                                 ("score", ("abs_q90", "abs_q99", "sample_second_moment", "top_0p1pct_squared_score_share")),
                                 ("parameter_gradient_norm", ("q50", "q90", "q99", "max"))):
                summary[loss_name][str(epoch)][metric] = {
                    k: {"mean": float(np.mean([r[metric][k] for r in rows])),
                        "std_ddof0": float(np.std([r[metric][k] for r in rows]))} for k in keys
                }
    result = {"config": config, "summary": summary, "records": records, "pretraining_trajectories": trajectories}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / "epoch20_score_samples.npz",
                        **{loss.replace(".", "p"): (np.concatenate(parts) if parts else np.array([], dtype=np.float32))
                           for loss, parts in epoch20_scores.items()})
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"elapsed_seconds": time.time()-started}, indent=2), encoding="utf-8")
    print(f"COMPLETE output={args.output_dir} elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
