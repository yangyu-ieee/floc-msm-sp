"""CWRU bearing-vibration robust-loss comparison.

This script adds a real-signal/semi-real robustness check for the SP paper:
masked signal modeling on Case Western Reserve University bearing vibration
data, evaluated with a load-held-out split.

Default task:
  - classes: Normal, IR007, B007, OR007@6
  - train loads: 0/1/2 hp
  - test load: 3 hp
  - optional alpha-stable impulsive interference on real vibration segments

The goal is not to replace the synthetic alpha-stable benchmark. It checks
whether the robust-loss conclusions remain plausible on real machinery signals.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import requests
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path("D:/deepl/paper2_floc_msm")
DATA_ROOT = PROJECT_ROOT / "data" / "cwru"
CWRU_NPZ_ROOT = PROJECT_ROOT / "data" / "cwru_npz_repo" / "Data"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "cwru_loss_compare"

CWRU_MAT_IDS = {
    "Normal": {
        0: "97",
        1: "98",
        2: "99",
        3: "100",
    },
    "IR007": {
        0: "105",
        1: "106",
        2: "107",
        3: "108",
    },
    "B007": {
        0: "118",
        1: "119",
        2: "120",
        3: "121",
    },
    "OR007@6": {
        0: "130",
        1: "131",
        2: "132",
        3: "133",
    },
}

CWRU_RPM_BY_LOAD = {
    0: "1797",
    1: "1772",
    2: "1750",
    3: "1730",
}

CWRU_NPZ_TEMPLATES = {
    "Normal": "{rpm}_Normal.npz",
    "IR007": "{rpm}_IR_7_DE12.npz",
    "B007": "{rpm}_B_7_DE12.npz",
    "OR007@6": "{rpm}_OR@6_7_DE12.npz",
}

LOSSES = ["scratch", "mse", "l1", "huber", "charbonnier", "floc"]


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_text(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def download_cwru_file(file_id, out_path, timeout):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return

    url = f"https://engineering.case.edu/sites/default/files/{file_id}.mat"
    headers = {"User-Agent": "FLOC-MSM-CWRU-benchmark/1.0"}
    tmp_path = out_path.with_suffix(".mat.tmp")
    last_err = None
    for attempt in range(1, 4):
        try:
            print(f"  download {url} (attempt {attempt})", flush=True)
            with requests.get(url, timeout=(10, timeout), stream=True, headers=headers) as r:
                r.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(out_path)
            return
        except Exception as exc:
            last_err = exc
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"    download failed: {exc}", flush=True)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def load_drive_end_signal(mat_path):
    mat = sio.loadmat(mat_path)
    de_keys = [k for k in mat.keys() if k.endswith("_DE_time")]
    if not de_keys:
        de_keys = [k for k in mat.keys() if "DE" in k and "time" in k]
    if not de_keys:
        raise KeyError(f"No drive-end time key found in {mat_path}; keys={list(mat.keys())}")
    sig = np.asarray(mat[de_keys[0]]).reshape(-1).astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


def load_npz_signal(npz_path, channel):
    z = np.load(npz_path)
    if channel not in z.files:
        raise KeyError(f"Channel {channel} not found in {npz_path}; keys={z.files}")
    sig = np.asarray(z[channel]).reshape(-1).astype(np.float32)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


def segment_signal(signal, seq_len, stride, max_segments, rng):
    starts = np.arange(0, max(0, len(signal) - seq_len + 1), stride)
    if len(starts) == 0:
        return np.empty((0, seq_len), dtype=np.float32)
    if max_segments > 0 and len(starts) > max_segments:
        starts = rng.choice(starts, size=max_segments, replace=False)
        starts.sort()

    xs = []
    for s in starts:
        seg = signal[s : s + seq_len].astype(np.float32).copy()
        seg -= seg.mean()
        seg /= seg.std() + 1e-6
        xs.append(seg)
    return np.stack(xs).astype(np.float32)


def stable_noise(shape, alpha, scale, rng):
    if scale <= 0:
        return np.zeros(shape, dtype=np.float32)
    if alpha >= 1.999:
        z = rng.normal(0.0, scale, size=shape).astype(np.float32)
        return z

    u = rng.uniform(-np.pi / 2, np.pi / 2, size=shape)
    w = rng.exponential(1.0, size=shape)
    z = (
        np.sin(alpha * u)
        / (np.cos(u) ** (1.0 / alpha))
        * (np.cos((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha)
    )
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scale * z


def add_impulsive_noise(x, alpha, scale, seed):
    if scale <= 0:
        return x.astype(np.float32)
    rng = np.random.default_rng(seed)
    xn = x + stable_noise(x.shape, alpha, scale, rng)
    xn -= xn.mean(axis=1, keepdims=True)
    xn /= xn.std(axis=1, keepdims=True) + 1e-6
    return xn.astype(np.float32)


def prepare_cwru_dataset(args):
    classes = parse_csv_text(args.classes)
    train_loads = parse_csv_ints(args.train_loads)
    test_loads = parse_csv_ints(args.test_loads)
    rng = np.random.default_rng(args.data_seed)

    label_map = {name: i for i, name in enumerate(classes)}
    split_x = {"train": [], "test": []}
    split_y = {"train": [], "test": []}
    split_meta = {"train": [], "test": []}

    for cls in classes:
        if args.source == "npz" and cls not in CWRU_NPZ_TEMPLATES:
            raise ValueError(f"Unknown class {cls}. Valid: {sorted(CWRU_NPZ_TEMPLATES)}")
        if args.source == "mat" and cls not in CWRU_MAT_IDS:
            raise ValueError(f"Unknown class {cls}. Valid: {sorted(CWRU_MAT_IDS)}")

        available_loads = CWRU_RPM_BY_LOAD if args.source == "npz" else CWRU_MAT_IDS[cls]
        for load in available_loads:
            if load not in train_loads and load not in test_loads:
                continue
            if args.source == "npz":
                rpm = CWRU_RPM_BY_LOAD[load]
                fname = CWRU_NPZ_TEMPLATES[cls].format(rpm=rpm)
                signal_path = Path(args.npz_root) / f"{rpm} RPM" / fname
                if not signal_path.exists():
                    raise FileNotFoundError(
                        f"Missing {signal_path}. Clone https://github.com/srigas/CWRU_Bearing_NumPy "
                        f"under {CWRU_NPZ_ROOT.parent} or pass --npz-root."
                    )
                sig = load_npz_signal(signal_path, args.channel)
                file_id = fname
            else:
                file_id = CWRU_MAT_IDS[cls][load]
                mat_path = DATA_ROOT / f"{file_id}.mat"
                download_cwru_file(file_id, mat_path, args.download_timeout)
                sig = load_drive_end_signal(mat_path)
            x = segment_signal(sig, args.seq_len, args.stride, args.max_segments_per_file, rng)
            if len(x) == 0:
                continue
            split = "train" if load in train_loads else "test"
            y = np.full(len(x), label_map[cls], dtype=np.int64)
            split_x[split].append(x)
            split_y[split].append(y)
            split_meta[split].extend([{"class": cls, "load": load, "file_id": file_id}] * len(x))

    if not split_x["train"] or not split_x["test"]:
        raise RuntimeError("Empty train or test split after loading CWRU.")

    x_train = np.concatenate(split_x["train"], axis=0)
    y_train = np.concatenate(split_y["train"], axis=0)
    x_test = np.concatenate(split_x["test"], axis=0)
    y_test = np.concatenate(split_y["test"], axis=0)
    return x_train, y_train, x_test, y_test, label_map, split_meta


class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Token length {x.size(1)} exceeds positional table {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1), :]


class SignalEncoder(nn.Module):
    def __init__(self, d_model=64, heads=4, n_layers=4, max_len=1024):
        super().__init__()
        self.tok = nn.Conv1d(1, d_model, kernel_size=8, stride=4, padding=2)
        self.pe = PosEnc(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=False,
        )
        self.trf = nn.TransformerEncoder(enc_layer, n_layers)
        self.d_model = d_model

    def tokenize(self, x):
        return self.pe(self.tok(x.unsqueeze(1)).transpose(1, 2))

    def forward(self, x, cls=None):
        z = self.tokenize(x) if x.dim() == 2 else x
        if cls is not None:
            z = torch.cat([cls, z], dim=1)
        return self.trf(z)


def reconstruction_loss(recon, target, loss_type):
    r = recon - target
    if loss_type == "mse":
        return r.pow(2).mean()
    if loss_type == "l1":
        return r.abs().mean()
    if loss_type == "huber":
        return F.huber_loss(recon, target, delta=1.0)
    if loss_type == "charbonnier":
        return torch.sqrt(r.pow(2) + 0.01**2).mean()
    if loss_type == "floc":
        return r.abs().pow(1.2).mean()
    raise ValueError(loss_type)


def pretrain_msm(x_unlabeled, loss_type, args, device):
    xt = torch.from_numpy(x_unlabeled)
    dl = DataLoader(
        TensorDataset(xt),
        batch_size=min(args.batch_size, len(xt)),
        shuffle=True,
        drop_last=False,
    )
    enc = SignalEncoder(args.d_model, args.heads, args.layers, args.max_tokens).to(device)
    dec = nn.Sequential(
        nn.Linear(args.d_model, 128),
        nn.ReLU(),
        nn.Linear(128, x_unlabeled.shape[1]),
    ).to(device)
    mask_tok = nn.Parameter(torch.randn(1, 1, args.d_model, device=device) * 0.02)
    params = list(enc.parameters()) + list(dec.parameters()) + [mask_tok]
    opt = torch.optim.Adam(params, lr=args.lr)

    for _ in range(args.pt_epochs):
        enc.train()
        dec.train()
        for (xb,) in dl:
            xb = xb.to(device)
            tok = enc.tokenize(xb)
            mask = torch.rand(xb.shape[0], tok.shape[1], 1, device=device) < args.mask_ratio
            masked = torch.where(mask, mask_tok.expand(xb.shape[0], tok.shape[1], -1), tok)
            h = enc(masked).mean(dim=1)
            recon = dec(h)
            loss = reconstruction_loss(recon, xb, loss_type)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def sample_fewshot_indices(y_train, shots, rng):
    idx = []
    for c in sorted(np.unique(y_train)):
        c_idx = np.where(y_train == c)[0]
        if len(c_idx) < shots:
            pick = rng.choice(c_idx, size=shots, replace=True)
        else:
            pick = rng.choice(c_idx, size=shots, replace=False)
        idx.append(pick)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def eval_classifier(weights, x_train, y_train, x_test, y_test, shots, args, device, seed):
    rng = np.random.default_rng(seed + 10007 * shots)
    idx = sample_fewshot_indices(y_train, shots, rng)
    xl = torch.from_numpy(x_train[idx])
    yl = torch.from_numpy(y_train[idx])
    xte = torch.from_numpy(x_test)
    yte = torch.from_numpy(y_test)

    train_dl = DataLoader(
        TensorDataset(xl, yl),
        batch_size=min(args.ft_batch_size, len(xl)),
        shuffle=True,
        drop_last=False,
    )
    test_dl = DataLoader(TensorDataset(xte, yte), batch_size=args.eval_batch_size, shuffle=False)

    enc = SignalEncoder(args.d_model, args.heads, args.layers, args.max_tokens).to(device)
    if weights is not None:
        enc.load_state_dict({k: v.to(device) for k, v in weights.items()})
    cls_tok = nn.Parameter(torch.randn(1, 1, args.d_model, device=device) * 0.02)
    head = nn.Sequential(
        nn.Linear(args.d_model, 32),
        nn.ReLU(),
        nn.Linear(32, int(y_train.max()) + 1),
    ).to(device)
    params = list(enc.parameters()) + [cls_tok] + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    for _ in range(args.ft_epochs if weights is not None else args.scratch_epochs):
        enc.train()
        head.train()
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            cls = cls_tok.expand(xb.shape[0], 1, -1)
            logits = head(enc(xb, cls=cls)[:, 0, :])
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

    enc.eval()
    head.eval()
    correct = 0
    total = 0
    conf = np.zeros((int(y_train.max()) + 1, int(y_train.max()) + 1), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            cls = cls_tok.expand(xb.shape[0], 1, -1)
            pred = head(enc(xb, cls=cls)[:, 0, :]).argmax(dim=1).cpu()
            correct += int((pred == yb).sum())
            total += len(yb)
            for t, p in zip(yb.numpy(), pred.numpy()):
                conf[int(t), int(p)] += 1
    return correct / max(1, total), conf


def subsample_rows(x, y, max_rows, seed):
    if max_rows <= 0 or len(x) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_rows, replace=False)
    return x[idx], y[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="npz", choices=["npz", "mat"])
    parser.add_argument("--npz-root", type=str, default=str(CWRU_NPZ_ROOT))
    parser.add_argument("--channel", type=str, default="DE")
    parser.add_argument("--classes", type=str, default="Normal,IR007,B007,OR007@6")
    parser.add_argument("--train-loads", type=str, default="0,1,2")
    parser.add_argument("--test-loads", type=str, default="3")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-segments-per-file", type=int, default=240)
    parser.add_argument("--max-pretrain", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--download-timeout", type=int, default=45)
    parser.add_argument("--shots", type=str, default="10,20,50")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--losses", type=str, default=",".join(LOSSES))
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--noise-scale", type=float, default=0.30)
    parser.add_argument("--pt-epochs", type=int, default=12)
    parser.add_argument("--ft-epochs", type=int, default=25)
    parser.add_argument("--scratch-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ft-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--data-seed", type=int, default=123)
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG))
    args = parser.parse_args()

    try:
        import sys

        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses = parse_csv_text(args.losses)
    shots_list = parse_csv_ints(args.shots)

    print(f"Device={device}", flush=True)
    print(f"Losses={losses} | shots={shots_list} | seeds={args.seeds}", flush=True)
    print(f"CWRU source={args.source} | channel={args.channel}", flush=True)
    print(f"CWRU classes={args.classes} | train_loads={args.train_loads} | test_loads={args.test_loads}", flush=True)
    print(f"alpha={args.alpha} | noise_scale={args.noise_scale}", flush=True)

    x_train_raw, y_train, x_test_raw, y_test, label_map, split_meta = prepare_cwru_dataset(args)
    print(f"Raw split: train={x_train_raw.shape}, test={x_test_raw.shape}, labels={label_map}", flush=True)

    results = {}
    seed_details = []
    for seed in range(args.seeds):
        print(f"\nSeed {seed}", flush=True)
        set_seed(seed)
        x_train = add_impulsive_noise(x_train_raw, args.alpha, args.noise_scale, seed + 17)
        x_test = add_impulsive_noise(x_test_raw, args.alpha, args.noise_scale, seed + 31)
        x_pt, y_pt = subsample_rows(x_train, y_train, args.max_pretrain, seed + 43)
        if args.max_test > 0:
            x_test_eval, y_test_eval = subsample_rows(x_test, y_test, args.max_test, seed + 53)
        else:
            x_test_eval, y_test_eval = x_test, y_test

        weights = {}
        for lt in losses:
            if lt == "scratch":
                continue
            print(f"  PT {lt}...", flush=True)
            weights[lt] = pretrain_msm(x_pt, lt, args, device)

        for shots in shots_list:
            for lt in losses:
                print(f"  eval {lt} {shots}shot...", flush=True)
                if lt == "scratch":
                    acc, conf = eval_classifier(
                        None, x_train, y_train, x_test_eval, y_test_eval, shots, args, device, seed
                    )
                else:
                    acc, conf = eval_classifier(
                        weights[lt], x_train, y_train, x_test_eval, y_test_eval, shots, args, device, seed
                    )
                key = f"shots={shots}/loss={lt}"
                results.setdefault(key, {"seeds": [], "confusions": []})
                results[key]["seeds"].append(float(acc))
                results[key]["confusions"].append(conf.tolist())
                seed_details.append({"seed": seed, "shots": shots, "loss": lt, "accuracy": float(acc)})
                print(f"    acc={acc:.4f}", flush=True)

    for key, item in results.items():
        vals = np.asarray(item["seeds"], dtype=np.float64)
        item["mean"] = float(vals.mean())
        item["std"] = float(vals.std(ddof=0))

    win_count = {lt: 0 for lt in losses}
    for shots in shots_list:
        best_loss = None
        best_acc = -1.0
        for lt in losses:
            key = f"shots={shots}/loss={lt}"
            if key in results and results[key]["mean"] > best_acc:
                best_acc = results[key]["mean"]
                best_loss = lt
        if best_loss is not None:
            win_count[best_loss] += 1

    print("\nSummary:", flush=True)
    for shots in shots_list:
        line = f"  {shots}shot:"
        for lt in losses:
            key = f"shots={shots}/loss={lt}"
            if key in results:
                line += f" {lt}={results[key]['mean']:.3f}±{results[key]['std']:.3f}"
        print(line, flush=True)
    print(f"Win count: {win_count}", flush=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "CWRU Bearing Data Center",
        "source": args.source,
        "npz_root": args.npz_root if args.source == "npz" else None,
        "channel": args.channel,
        "classes": parse_csv_text(args.classes),
        "label_map": label_map,
        "train_loads": parse_csv_ints(args.train_loads),
        "test_loads": parse_csv_ints(args.test_loads),
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_segments_per_file": args.max_segments_per_file,
        "alpha": args.alpha,
        "noise_scale": args.noise_scale,
        "shots": shots_list,
        "seeds": args.seeds,
        "losses": losses,
        "pt_epochs": args.pt_epochs,
        "ft_epochs": args.ft_epochs,
        "scratch_epochs": args.scratch_epochs,
        "device": device,
        "train_shape": list(x_train_raw.shape),
        "test_shape": list(x_test_raw.shape),
        "win_count": win_count,
        "elapsed_sec": time.time() - t0,
    }
    (log_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (log_dir / "seed_details.json").write_text(json.dumps(seed_details, indent=2), encoding="utf-8")
    print(f"\nDone | log_dir={log_dir} | elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
