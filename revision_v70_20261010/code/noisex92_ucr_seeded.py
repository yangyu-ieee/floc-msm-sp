"""Seed-controlled NOISEX-92/UCR rerun for reviewer R1.5.

Each of five replicate seeds controls real-noise mixing, augmentation,
pretraining initialization/shuffle/masks, few-shot sampling, and fine-tuning.
Within a replicate, all losses share the same noisy train/test realization,
pretraining corpus, few-shot indices, and downstream initialization seed.
The earlier exploratory runner and outputs are left untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import argparse
import time
import wave
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "noisex92_seeded_rerun_2026-10-07"
UCR_ROOT = OUT / "data" / "ucr"
NOISE_ROOT = Path(os.environ.get("NOISEX_ROOT", "data/NOISEX-92"))
DATASETS = (
    "ECG200", "ECG5000", "Wafer", "GunPoint", "CBF", "FaceAll",
    "SwedishLeaf", "ChlorineConcentration",
)
NOISE_FILES = ("machinegun.wav", "factory1.wav", "factory2.wav")
LOSS_TYPES = ("mse", "l1", "floc")
MASTER_SEEDS = (0, 1, 2, 3, 4)
SHOTS = (5, 10)
SNR_DB = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(master: int, dataset: str, stage: str, substage: str = "shared") -> int:
    payload = f"noisex92-ucr-v1|{master}|{dataset}|{stage}|{substage}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_ucr_tsv(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = UCR_ROOT / dataset / f"{dataset}_{split}.tsv"
    arr = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected UCR shape in {path}: {arr.shape}")
    y = arr[:, 0]
    x = arr[:, 1:].astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Non-finite values in {path}")
    return x, y


def load_noise_bank() -> np.ndarray:
    signals = []
    for filename in NOISE_FILES:
        path = NOISE_ROOT / filename
        with wave.open(str(path), "rb") as wf:
            channels, width, nframes = wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            raw = wf.readframes(nframes)
            dtypes = {1: np.int8, 2: np.int16, 4: np.int32}
            if width not in dtypes:
                raise ValueError(f"Unsupported WAV sample width {width}: {path}")
            signal = np.frombuffer(raw, dtype=dtypes[width]).astype(np.float32)
            if channels > 1:
                signal = signal.reshape(-1, channels).mean(axis=1)
            signal /= np.abs(signal).max() + 1e-8
            signals.append(signal)
            print(f"{filename}: {channels}ch/{width}B, {nframes} frames")
    n = min(map(len, signals))
    return np.stack([s[:n] for s in signals])


def add_real_impulsive_noise(
    x: np.ndarray, noise_bank: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    noisy = x.copy()
    n, length = noisy.shape
    if length >= noise_bank.shape[1]:
        raise ValueError("NOISEX source is shorter than an input time series")
    for i in range(n):
        source = int(rng.integers(0, len(noise_bank)))
        start = int(rng.integers(0, noise_bank.shape[1] - length + 1))
        segment = noise_bank[source, start : start + length]
        signal_rms = np.sqrt(np.mean(x[i] ** 2) + 1e-12)
        noise_rms = np.sqrt(np.mean(segment ** 2) + 1e-12)
        scale = signal_rms * 10 ** (-SNR_DB / 20) / (noise_rms + 1e-12)
        noisy[i] += segment * scale
    noisy /= np.abs(noisy).max(axis=1, keepdims=True) + 1e-8
    return noisy.astype(np.float32)


class PosEnc(nn.Module):
    def __init__(self, d: int = 64, max_len: int = 400):
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
        self.pe = PosEnc(d, 400)
        layer = nn.TransformerEncoderLayer(
            d, heads, 256, dropout=0.1, batch_first=True
        )
        self.trf = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = self.pe(self.tok(x.unsqueeze(1)).transpose(1, 2))
        if cls is not None:
            x = torch.cat([cls, x], dim=1)
        return self.trf(x)


def make_pretraining_set(x_train_noisy: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    copies = max(1, 3000 // len(x_train_noisy))
    x_pt = np.tile(x_train_noisy, (copies, 1))
    x_pt += rng.standard_normal(x_pt.shape, dtype=np.float32) * 0.02
    x_pt /= np.abs(x_pt).max(axis=1, keepdims=True) + 1e-8
    return x_pt.astype(np.float32)


def pretrain(x_unlabeled: np.ndarray, loss_type: str, epochs: int = 20):
    x_tensor = torch.from_numpy(x_unlabeled)
    loader = DataLoader(
        TensorDataset(x_tensor), batch_size=min(128, len(x_tensor)), shuffle=True
    )
    encoder = SignalEncoder().to(DEVICE)
    decoder = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, x_unlabeled.shape[1])
    ).to(DEVICE)
    mask_token = nn.Parameter(torch.randn(1, 1, 64, device=DEVICE) * 0.02)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [mask_token], lr=1e-3
    )
    for _ in range(epochs):
        encoder.train()
        decoder.train()
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            tokens = encoder.pe(encoder.tok(xb.unsqueeze(1)).transpose(1, 2))
            mask = torch.rand(xb.shape[0], tokens.shape[1], 1, device=DEVICE) < 0.5
            masked = torch.where(mask, mask_token.expand(xb.shape[0], tokens.shape[1], -1), tokens)
            reconstruction = decoder(encoder.trf(masked).mean(dim=1))
            residual = reconstruction - xb
            if loss_type == "mse":
                loss = residual.pow(2).mean()
            elif loss_type == "l1":
                loss = residual.abs().mean()
            elif loss_type == "floc":
                loss = residual.abs().pow(1.2).mean()
            else:
                raise ValueError(loss_type)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
    return {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}


def choose_fewshot_indices(y_train: np.ndarray, shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(y_train):
        candidates = np.flatnonzero(y_train == cls)
        chosen.append(rng.choice(candidates, min(shots, len(candidates)), replace=False))
    indices = np.concatenate(chosen)
    rng.shuffle(indices)
    return indices


def evaluate(
    encoder_weights,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    indices: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
) -> float:
    seed_all(seed)
    x_labeled = torch.from_numpy(x_train[indices])
    y_labeled = torch.from_numpy(y_train[indices]).long()
    x_eval = torch.from_numpy(x_test)
    y_eval = torch.from_numpy(y_test).long()
    loader = DataLoader(
        TensorDataset(x_labeled, y_labeled), batch_size=min(32, len(x_labeled)), shuffle=True
    )
    encoder = SignalEncoder().to(DEVICE)
    if encoder_weights is not None:
        encoder.load_state_dict(encoder_weights)
    class_token = nn.Parameter(torch.randn(1, 1, 64, device=DEVICE) * 0.02)
    head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes)).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + [class_token] + list(head.parameters()), lr=1e-3
    )
    for _ in range(epochs):
        encoder.train()
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            cls = class_token.expand(xb.shape[0], 1, -1)
            logits = head(encoder(xb, cls=cls)[:, 0, :])
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
    encoder.eval()
    head.eval()
    with torch.no_grad():
        cls = class_token.expand(len(x_eval), 1, -1)
        prediction = head(encoder(x_eval.to(DEVICE), cls=cls)[:, 0, :]).cpu().argmax(1)
    return float((prediction == y_eval).float().mean().item())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    global NOISE_ROOT, UCR_ROOT, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end smoke test in an isolated output directory")
    parser.add_argument("--noise-root", type=Path, default=NOISE_ROOT, help="Directory containing machinegun.wav, factory1.wav, and factory2.wav")
    parser.add_argument("--ucr-root", type=Path, default=UCR_ROOT, help="Directory containing UCR dataset subdirectories with TRAIN/TEST TSV files")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Output directory for results and downloaded UCR archive")
    args = parser.parse_args()
    NOISE_ROOT, UCR_ROOT, OUT = args.noise_root, args.ucr_root, args.output_dir
    started = time.time()
    out = OUT.with_name(OUT.name + "_smoke") if args.smoke else OUT
    datasets = DATASETS[:1] if args.smoke else DATASETS
    master_seeds = MASTER_SEEDS[:1] if args.smoke else MASTER_SEEDS
    shots_values = SHOTS[:1] if args.smoke else SHOTS
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This planned rerun expects CUDA to be available")
    torch.use_deterministic_algorithms(True)
    noise_bank = load_noise_bank()
    noise_hashes = {name: sha256(NOISE_ROOT / name) for name in NOISE_FILES}
    dataset_hashes = {}
    all_seed_results = {}

    for dataset in datasets:
        x_train_raw, y_train_raw = load_ucr_tsv(dataset, "TRAIN")
        x_test_raw, y_test_raw = load_ucr_tsv(dataset, "TEST")
        all_labels = np.unique(np.concatenate([y_train_raw, y_test_raw]))
        mapping = {label: idx for idx, label in enumerate(all_labels)}
        y_train = np.asarray([mapping[v] for v in y_train_raw], dtype=np.int64)
        y_test = np.asarray([mapping[v] for v in y_test_raw], dtype=np.int64)
        n_classes = len(all_labels)
        dataset_hashes[dataset] = {
            "train_tsv_sha256": sha256(UCR_ROOT / dataset / f"{dataset}_TRAIN.tsv"),
            "test_tsv_sha256": sha256(UCR_ROOT / dataset / f"{dataset}_TEST.tsv"),
            "train_shape": list(x_train_raw.shape),
            "test_shape": list(x_test_raw.shape),
            "class_labels": [float(v) for v in all_labels],
        }
        print(f"\n{dataset}: train={x_train_raw.shape}, test={x_test_raw.shape}, classes={n_classes}", flush=True)

        for master_seed in master_seeds:
            rng_noise = np.random.default_rng(derive_seed(master_seed, dataset, "noise"))
            x_train_noisy = add_real_impulsive_noise(x_train_raw, noise_bank, rng_noise)
            x_test_noisy = add_real_impulsive_noise(x_test_raw, noise_bank, rng_noise)
            x_pt = make_pretraining_set(
                x_train_noisy, derive_seed(master_seed, dataset, "pretraining-corpus")
            )

            weights = {}
            for loss_name in LOSS_TYPES:
                seed_all(derive_seed(master_seed, dataset, "pretrain-model"))
                weights[loss_name] = pretrain(x_pt, loss_name, epochs=1 if args.smoke else 20)

            for shots in shots_values:
                split_seed = derive_seed(master_seed, dataset, "fewshot", str(shots))
                indices = choose_fewshot_indices(y_train, shots, split_seed)
                for method in ("scratch", "mse", "l1", "floc"):
                    fit_seed = derive_seed(master_seed, dataset, "finetune", str(shots))
                    state = None if method == "scratch" else weights[method]
                    epochs = 1 if args.smoke else (80 if method == "scratch" else 50)
                    accuracy = evaluate(
                        state, x_train_noisy, y_train, x_test_noisy, y_test,
                        indices, n_classes, fit_seed, epochs,
                    )
                    key = f"{dataset}/shots={shots}/loss={method}"
                    all_seed_results.setdefault(key, {})[str(master_seed)] = accuracy
                    print(
                        f"seed={master_seed} {key} accuracy={accuracy:.4f}", flush=True
                    )
            save_json(out / "seed_level_results_partial.json", all_seed_results)

    results = {}
    for key, by_seed in all_seed_results.items():
        ordered = [by_seed[str(seed)] for seed in master_seeds]
        results[key] = {
            "mean": float(np.mean(ordered)),
            "std": float(np.std(ordered, ddof=0)),
            "seeds": ordered,
            "seed_ids": list(master_seeds),
            "std_scope": "complete pipeline replicate (noise mixing, pretraining, few-shot sampling, fine-tuning)",
        }
    win_count = {name: 0 for name in ("scratch", "mse", "l1", "floc")}
    for dataset in datasets:
        for shots in shots_values:
            means = {loss: results[f"{dataset}/shots={shots}/loss={loss}"]["mean"]
                     for loss in win_count}
            win_count[max(means, key=means.get)] += 1
    save_json(out / "results.json", results)
    save_json(out / "summary.json", {
        "win_count": win_count,
        "n_conditions": len(datasets) * len(shots_values),
        "master_seeds": list(master_seeds),
        "snr_db": SNR_DB,
        "noise_sources": list(NOISE_FILES),
        "std_scope": "complete pipeline replicate; each seed re-mixes noise and re-pretrains",
    })
    save_json(out / "run_metadata.json", {
        "experiment": "NOISEX-92_UCR_full_pipeline_seeded_rerun",
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "master_seeds": list(master_seeds),
        "datasets": dataset_hashes,
        "noise_file_sha256": noise_hashes,
        "ucr_source": "Official UCR Time Series Classification Archive 2018",
        "ucr_source_archive_sha256": sha256(OUT / "download" / "UCRArchive_2018.zip"),
        "script_sha256": sha256(Path(__file__)),
        "protocol": {
            "noise_sources": list(NOISE_FILES), "snr_db": SNR_DB,
            "max_abs_normalization_after_noise": True,
            "pretraining_corpus_size_approx": 3000, "pretraining_epochs": 1 if args.smoke else 20,
            "downstream_epochs": {"scratch": 80, "pretrained": 50},
            "fewshot_per_class": list(shots_values), "losses": list(LOSS_TYPES),
            "replication_unit": "full pipeline seed, including noise realization and pretraining",
            "matching": "same replicate noisy data, pretrained data, sampled few-shot indices, and finetuning RNG seed across loss methods",
        },
    })
    print(f"\nDONE in {time.time() - started:.1f}s; outputs: {out}", flush=True)
    print(f"Win counts: {win_count}", flush=True)


if __name__ == "__main__":
    main()
