#!/usr/bin/env python3
"""Reference feature computation, the ground truth for `pmcore::features`.

Computes, with numpy in float64, the per-axis time-domain stats (RMS, crest factor,
Pearson kurtosis) plus the per-axis log-spaced FFT magnitude bands for a window of
accelerometer samples. The definitions mirror the Rust in pmcore/src/features.rs exactly
-- same formulas, same Hann window and band edges, same layout (all-axes stats block, then
all-axes spectral-band block) -- so `host-sim` (which runs the real Rust `extract`) and
this script agree to f32 tolerance on identical input. That agreement is the Milestone B
gate.

Usage:
    python verify_features.py <window.csv>          # print the 24 features
    python verify_features.py --gen <window.csv>    # write a deterministic test window
"""

import argparse
import numpy as np

N_AXES = 3
WINDOW_LEN = 512

# Spectral band features (must match pmcore::features). Log-spaced bin-index edges over the
# 256-bin magnitude spectrum of a 512-point real FFT; band b is bins BAND_EDGES[b:b+1], DC
# (bin 0) skipped.
FFT_BANDS_PER_AXIS = 5
BAND_EDGES = [1, 3, 9, 28, 84, 256]


def axis_features(x):
    """[rms, crest, kurtosis] for one axis, matching pmcore::features."""
    x = x.astype(np.float64)
    rms = np.sqrt(np.mean(x * x))
    peak = np.max(np.abs(x))
    crest = 0.0 if rms == 0.0 else peak / rms

    d = x - x.mean()
    m2 = np.mean(d * d)            # biased (population) variance
    m4 = np.mean(d**4)
    kurt = 0.0 if m2 == 0.0 else m4 / (m2 * m2)

    return [rms, crest, kurt]


def fft_bands(x):
    """[FFT_BANDS_PER_AXIS] log1p band magnitudes for one axis, matching pmcore::fft_bands.

    Periodic Hann window (denominator N), real FFT magnitude, summed per log-spaced band,
    log1p-compressed. The positive-frequency bins 1..255 of numpy's rfft line up with the
    Rust FFT primitive's bins; DC and Nyquist are excluded by the edge table.
    """
    x = x.astype(np.float64)
    n = len(x)
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)
    mag = np.abs(np.fft.rfft(x * win))
    return [float(np.log1p(np.sum(mag[BAND_EDGES[b]:BAND_EDGES[b + 1]])))
            for b in range(FFT_BANDS_PER_AXIS)]


def features(window):
    """Flat 24-vector: the per-axis [rms, crest, kurt] stats block, then the per-axis
    FFT-band block (axis 0's bands, then axis 1, then axis 2)."""
    stats, bands = [], []
    for a in range(N_AXES):
        col = window[:, a]
        stats.extend(axis_features(col))
        bands.extend(fft_bands(col))
    return stats + bands


def load_csv(path):
    w = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return w.reshape(-1, N_AXES)


def gen(path, seed, n):
    """A deterministic window with non-trivial RMS/crest/kurtosis on every axis."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 1500 * np.sin(2 * np.pi * t / 37) + rng.normal(0, 300, n)   # tone + noise
    y = rng.normal(0, 800, n)                                       # broadband
    z = rng.normal(0, 200, n)
    z[rng.integers(0, n, 8)] += 6000                               # impulses -> high kurtosis
    w = np.stack([x, y, z], axis=1)
    w = np.clip(np.round(w), -32768, 32767).astype(np.int64)
    np.savetxt(path, w, fmt="%d", delimiter=",")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("window", nargs="?", help="window CSV (x,y,z per row)")
    ap.add_argument("--gen", metavar="PATH", help="write a deterministic test window and exit")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=WINDOW_LEN, help="samples in the window")
    args = ap.parse_args()

    if args.gen:
        gen(args.gen, args.seed, args.n)
        return
    if not args.window:
        ap.error("provide a window CSV, or --gen PATH to create one")

    feats = features(load_csv(args.window))
    print(" ".join(f"{v:.6f}" for v in feats))


if __name__ == "__main__":
    main()
