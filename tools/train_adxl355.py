#!/usr/bin/env python3
"""Train the 3-class bearing CNN on the ADXL355 triaxial induction-motor dataset.

Dataset: *Triaxial Bearing Vibration Dataset of Induction Motor under Varying Load
Conditions*, Mendeley Data DOI 10.17632/fm6xzxnf36.2. An **ADXL355** MEMS triaxial
accelerometer (same Analog Devices family as the firmware's ADXL345), recorded at 10 kHz,
shipped as CSV with columns `Time Stamp, X-axis, Y-axis, Z-axis` (acceleration in g).
Download the CSVs into `--data` (gitignored, like every other fixture). Filenames encode
the class:

    Healthy-without-pulley.csv / Healthy-with-pulley.csv   -> normal
    <size>inner-<load> watt.csv   e.g. "0.7inner-100 watt"  -> inner_race
    <size>outer-<load> watt.csv   e.g. "1.5outer-300 watt"  -> outer_race

THE COMPROMISE (read this). Each recording is only 1000 samples = 0.1 s, far shorter than
the firmware's 512-sample window. The only ADXL345-achievable rate at which the data yields
a 512-sample window is the chip's max ODR, 3200 Hz, so we **decimate 10 kHz -> 3200 Hz**
(the firmware's `BW_RATE` is set to match) and window 512 samples (0.16 s) with overlap.
The `normal` class has only two recordings (~0.2 s total) so it is data-starved and heavily
overlapped; its held-out metric rests on a single validation recording and is weak by
necessity. This is the documented limitation of using this (otherwise ideal sensor- and
taxonomy-matched) short dataset.

Deployment consistency. The firmware feeds the model RAW int16 counts and the RAW feature
vector from `pmcore::features::extract`. To keep that Rust forward pass untouched while
still training in a well-conditioned space, we train on normalized inputs/features and then
**fold** the normalization back into the weights before export: the window scale folds into
`conv1.weight`, and the per-feature mean/std fold into `fc.weight`/`fc.bias`. The exported
checkpoint therefore operates directly on raw counts + raw features.

Usage:
    python train_adxl355.py --data models/adxl355 --out models/bearing.pt
    # then: python export_model.py --checkpoint models/bearing.pt --out models/bearing_cnn.bin --quantize
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_model import BearingCNN, DEFAULT_CFG, N_AXES, FEATURE_LEN, N_CLASSES
from verify_features import features as compute_features

SRC_RATE = 10_000          # ADXL355 acquisition rate (Hz)
TARGET_RATE = 3_200        # ADXL345 max ODR; the rate the firmware is configured to match
RESAMPLE_UP, RESAMPLE_DOWN = 8, 25     # 10000 * 8 / 25 = 3200
WINDOW_LEN = 512
ADXL345_LSB_PER_G = 256.0  # ADXL345 sensitivity at the +-2 g range (3.9 mg/LSB)
I16_MAX = 32767

# Class index order must match pmcore::model::Class.
CLASS_NAMES = ["normal", "inner_race", "outer_race"]


def label_of(path):
    """Map a dataset filename to a class index, or None to skip."""
    name = os.path.basename(path).lower()
    if "healthy" in name or "normal" in name:
        return 0
    if "inner" in name:
        return 1
    if "outer" in name:
        return 2
    return None


def load_recording_g(path):
    """Load a CSV recording as an (n, N_AXES) float array of g-values (X, Y, Z)."""
    arr = np.genfromtxt(path, delimiter=",", skip_header=1)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"{path}: expected >=4 columns (Time, X, Y, Z)")
    return arr[:, 1:4].astype(np.float64)


def resample_poly_np(x, up, down):
    """Anti-aliased rational resample (scipy.signal.resample_poly equivalent, numpy only).

    Upsample by `up` (zero-stuff), low-pass with a Hann-windowed sinc at the lower of the
    two Nyquists, then keep every `down`-th sample.
    """
    n = len(x)
    upx = np.zeros(n * up)
    upx[::up] = x * up
    half = 10 * max(up, down)
    t = np.arange(-half, half + 1, dtype=np.float64)
    cutoff = 1.0 / max(up, down)                       # normalized to the upsampled Nyquist
    h = cutoff * np.sinc(cutoff * t) * np.hanning(len(t))
    h /= h.sum()
    return np.convolve(upx, h, mode="same")[::down]


def g_to_counts(sig_g):
    """g -> int16 ADC counts in the ADXL345 domain the firmware feeds to `extract`."""
    counts = np.rint(sig_g * ADXL345_LSB_PER_G)
    return np.clip(counts, -I16_MAX, I16_MAX).astype(np.int16)


def windows_from_recordings(recs, target_windows):
    """Decimate, concatenate, and overlap-window a class's recordings into 512-sample
    int16 windows. Stride is chosen so the class yields about `target_windows` windows
    (heavy overlap for the data-starved classes); returns an (m, WINDOW_LEN, N_AXES) array."""
    if not recs:
        return np.empty((0, WINDOW_LEN, N_AXES), dtype=np.int16)
    cols = [np.stack([resample_poly_np(r[:, a], RESAMPLE_UP, RESAMPLE_DOWN)
                      for a in range(N_AXES)], axis=1) for r in recs]
    sig = g_to_counts(np.concatenate(cols, axis=0))
    n = len(sig)
    if n < WINDOW_LEN:
        return np.empty((0, WINDOW_LEN, N_AXES), dtype=np.int16)
    span = n - WINDOW_LEN
    stride = max(1, span // max(1, target_windows - 1)) if target_windows > 1 else span + 1
    starts = list(range(0, span + 1, stride))
    return np.stack([sig[s:s + WINDOW_LEN] for s in starts])


def featurize(windows):
    """Feature matrix (m, FEATURE_LEN) via the shared numpy reference (== Rust `extract`)."""
    return np.stack([np.asarray(compute_features(w), dtype=np.float64) for w in windows])


# A single recording is only ~320 samples after decimation (< WINDOW_LEN), so a split is
# only viable when several recordings are concatenated. Classes with at least this many
# recordings get a leakage-free recording-level split; smaller ones (the 2-recording normal
# class) fall back to a window-level split, which leaks across the overlap and is flagged.
MIN_RECS_FOR_RECSPLIT = 4


def build_split(data_dir, val_frac, target_windows, seed):
    """Group recordings by class and produce train/val windows. Returns
    (Xtr, Ftr, ytr, Xva, Fva, yva) as numpy arrays."""
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in range(N_CLASSES)}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        c = label_of(path)
        if c is not None:
            by_class[c].append(load_recording_g(path))

    tr, va = {"X": [], "F": [], "y": []}, {"X": [], "F": [], "y": []}
    for c, recs in by_class.items():
        if not recs:
            print(f"  WARNING: no recordings for class '{CLASS_NAMES[c]}'")
            continue
        if len(recs) >= MIN_RECS_FOR_RECSPLIT:
            idx = rng.permutation(len(recs))
            n_val = max(1, int(round(len(recs) * val_frac)))
            w_tr = windows_from_recordings([recs[i] for i in idx[n_val:]], target_windows)
            w_va = windows_from_recordings([recs[i] for i in idx[:n_val]], target_windows)
            note = f"{len(recs) - n_val} train / {n_val} val recordings (recording-level)"
        else:
            w_all = windows_from_recordings(recs, target_windows)
            k = int(round(len(w_all) * (1.0 - val_frac)))
            w_tr, w_va = w_all[:k], w_all[k:]
            note = f"{len(recs)} recordings, window-level split (LEAKY — too few recordings)"
        for split, w in ((tr, w_tr), (va, w_va)):
            if len(w):
                split["X"].append(w)
                split["F"].append(featurize(w))
                split["y"].append(np.full(len(w), c))
        print(f"  {CLASS_NAMES[c]:<12}: {note} -> {len(w_tr)} train / {len(w_va)} val windows")

    def cat(d):
        if not d["X"]:
            return (np.empty((0, WINDOW_LEN, N_AXES)), np.empty((0, FEATURE_LEN)), np.empty(0))
        return np.concatenate(d["X"]), np.concatenate(d["F"]), np.concatenate(d["y"])

    return (*cat(tr), *cat(va))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir of ADXL355 CSV recordings")
    ap.add_argument("--out", required=True, help="output checkpoint .pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--target-windows", type=int, default=300,
                    help="approx windows per class (overlap balances the tiny normal class)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = DEFAULT_CFG
    print(f"loading + windowing {args.data} (10kHz -> {TARGET_RATE}Hz, {WINDOW_LEN}-sample windows)")
    Xtr, Ftr, ytr, Xva, Fva, yva = build_split(args.data, args.val_frac,
                                                args.target_windows, args.seed)
    if len(Xtr) == 0:
        ap.error("no training windows — check --data path and filename conventions")
    print(f"train windows: {len(Xtr)}   val windows: {len(Xva)}")

    # Normalize for well-conditioned training; the scales are folded back into the weights
    # at the end so the exported model consumes raw counts + raw features.
    win_scale = float(Xtr.std()) or 1.0
    f_mu = Ftr.mean(axis=0)
    f_sd = Ftr.std(axis=0)
    f_sd[f_sd == 0] = 1.0

    def to_tensors(X, Ffeat, y):
        xn = torch.tensor((X.transpose(0, 2, 1) / win_scale), dtype=torch.float32)   # (m,axes,len)
        fn = torch.tensor(((Ffeat - f_mu) / f_sd), dtype=torch.float32)
        return xn, fn, torch.tensor(y, dtype=torch.long)

    xtr, ftr, ttr = to_tensors(Xtr, Ftr, ytr)
    model = BearingCNN(cfg).train()
    # Class-balanced loss: the normal class is heavily under-represented.
    counts = np.bincount(ytr.astype(int), minlength=N_CLASSES).astype(np.float64)
    weight = torch.tensor((counts.sum() / np.maximum(counts, 1)) / N_CLASSES, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(xtr, ftr), ttr, weight=weight)
        loss.backward()
        opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}  loss {loss.item():.4f}")

    # Fold the input/feature normalization into the weights (see module docstring).
    model.eval()
    with torch.no_grad():
        model.conv1.weight.div_(win_scale)
        c2 = cfg["c2"]
        fw = model.fc.weight                          # (N_CLASSES, c2 + FEATURE_LEN)
        feat_w = fw[:, c2:]
        sd = torch.tensor(f_sd, dtype=torch.float32)
        mu = torch.tensor(f_mu, dtype=torch.float32)
        model.fc.bias.sub_((feat_w * (mu / sd)).sum(dim=1))
        fw[:, c2:] = feat_w / sd

    # Report held-out accuracy + confusion on RAW inputs (verifies the fold is correct).
    if len(Xva):
        with torch.no_grad():
            xva = torch.tensor(Xva.transpose(0, 2, 1).astype("float32"))
            fva = torch.tensor(Fva.astype("float32"))
            pred = model(xva, fva).argmax(dim=1).numpy()
        acc = float((pred == yva).mean())
        conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(yva.astype(int), pred):
            conf[t, p] += 1
        print(f"\nheld-out accuracy: {acc:.3f}  ({len(yva)} windows)")
        print("confusion (rows=true, cols=pred):")
        print("            " + " ".join(f"{n:>11}" for n in CLASS_NAMES))
        for i, n in enumerate(CLASS_NAMES):
            print(f"  {n:<10}" + " ".join(f"{conf[i, j]:>11d}" for j in range(N_CLASSES)))
        if (np.bincount(yva.astype(int), minlength=N_CLASSES) <= 1).any():
            print("  NOTE: a class has <=1 validation recording — its metric is not robust "
                  "(the documented normal-class data-starvation).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nwrote {args.out}  (feed to export_model.py --checkpoint)")


if __name__ == "__main__":
    main()
