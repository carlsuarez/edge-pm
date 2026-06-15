#!/usr/bin/env python3
"""Generate a multi-window sample stream + model + reference for the Milestone D gate.

`host-sim replay` chops a long `x,y,z` stream into 512-sample windows and drives the whole
pipeline: windowing -> features -> CNN forward pass -> the `NORMAL <-> ALERT` state machine.
This script produces a deterministic stream, a model, and an independent reference:

  * `<out>.bin`     -- a demo model. Its conv stack is random, but its dense head is wired by
    hand so the **z-axis kurtosis** feature drives the `outer_race` class: quiet (Gaussian)
    windows stay `normal`, impulsive (defect-like) windows cross 0.80 into `outer_race`. A
    purely random head classifies every window the same way and never alerts, which exercises
    nothing -- this makes the alert path actually fire on the data that should trip it.
  * `<out>.csv`     -- the stream: quiet/impulsive windows arranged to latch an alert and then
    clear it via the 3-consecutive-normal hysteresis rule.
  * `<out>.ref.bin` -- per-window softmax probs PyTorch produced on this model, plus the
    state-machine trajectory from `fsm_run`, the pure-Python mirror of `pmcore::alert`.

The Rust gate (`host-sim/tests/replay.rs`) must reproduce the probs bit-for-bit and the FSM
trajectory step-for-step, and must observe a real latch + clear.

Usage:
    python make_stream.py --out models/bearing_stream
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_model import (
    BearingCNN, write_model, DEFAULT_CFG,
    N_AXES, WINDOW_LEN, FEATURE_LEN, N_CLASSES,
)
from verify_features import features as compute_features

# Must match pmcore::alert.
ALERT_CONFIDENCE = 0.80
NORMAL_WINDOWS_TO_CLEAR = 3

# Window plan: quiet (Normal) and impulsive (fault) windows arranged so the alert latches at
# window 2, sustains through 3, then clears after three consecutive normals (windows 4-6).
IMPULSE_PLAN = [False, False, True, True, False, False, False, False]

OUTER_RACE = 2          # class index the z-kurtosis feature is wired to drive
Z_KURT_FEATURE = 8      # feature index of axis-2 kurtosis in the 9-vector
KURT_WEIGHT = 0.3       # dense weight on z-kurtosis -> outer_race logit
NORMAL_BIAS = 2.5       # dense bias on normal: quiet kurtosis (~3) keeps normal on top


def build_demo_model(seed):
    cfg = DEFAULT_CFG
    torch.manual_seed(seed)
    model = BearingCNN(cfg).eval()
    with torch.no_grad():
        # Hand-wire the dense head: only z-kurtosis (-> outer_race) and a normal bias matter;
        # zero everything else so the decision is interpretable and reproducible.
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        model.fc.weight[OUTER_RACE, cfg["c2"] + Z_KURT_FEATURE] = KURT_WEIGHT
        model.fc.bias[0] = NORMAL_BIAS
    return model, cfg


def gen_stream(seed):
    """Deterministic int16 stream following IMPULSE_PLAN: broadband noise everywhere, with
    strong periodic impulses on the z-axis of the windows flagged impulsive."""
    rng = np.random.default_rng(seed)
    n_windows = len(IMPULSE_PLAN)
    sig = rng.normal(0.0, 300.0, size=(n_windows * WINDOW_LEN, N_AXES))
    for w, impulsive in enumerate(IMPULSE_PLAN):
        if impulsive:
            base = w * WINDOW_LEN
            sig[base:base + WINDOW_LEN:24, 2] += 8000.0
    return np.clip(np.round(sig), -32768, 32767).astype(np.int16), n_windows


def model_probs(model, stream, n_windows):
    probs = []
    for w in range(n_windows):
        window = stream[w * WINDOW_LEN:(w + 1) * WINDOW_LEN]
        feats = compute_features(window)
        x = torch.tensor(window.T[None].astype("float32"))           # (1, N_AXES, WINDOW_LEN)
        f = torch.tensor(np.asarray(feats, dtype="float32")[None])   # (1, FEATURE_LEN)
        with torch.no_grad():
            p = torch.softmax(model(x, f), dim=-1)[0].numpy()
        probs.append(p.astype("<f4"))
    return np.asarray(probs)


def fsm_run(probs_seq, thr):
    """Pure-Python mirror of pmcore::alert::AlertMachine. Returns (state, latched) per
    window: state 0=Normal/1=Alert, latched=class index while alerting else -1."""
    state, latched, streak = 0, -1, 0
    out = []
    for p in probs_seq:
        idx = int(np.argmax(p))           # first-max on ties, like engine::math::argmax
        conf = float(p[idx])
        crosses = idx != 0 and conf > thr
        if state == 0:
            if crosses:
                state, latched, streak = 1, idx, 0
        else:
            if idx == 0:
                streak += 1
                if streak >= NORMAL_WINDOWS_TO_CLEAR:
                    state, latched, streak = 0, -1, 0
            else:
                streak = 0
                if crosses:
                    latched = idx
        out.append((state, latched))
    return out


def write_ref(path, probs, thr, track):
    n = len(probs)
    with open(path, "wb") as f:
        f.write(int(n).to_bytes(4, "little", signed=True))
        f.write(int(N_CLASSES).to_bytes(4, "little", signed=True))
        f.write(np.asarray([thr], dtype="<f4").tobytes())
        f.write(np.asarray(probs, dtype="<f4").tobytes())
        for state, latched in track:
            f.write(int(state).to_bytes(4, "little", signed=True))
            f.write(int(latched).to_bytes(4, "little", signed=True))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output stem (writes <out>.bin/.csv/.ref.bin)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = build_demo_model(args.seed)
    stream, n_windows = gen_stream(args.seed)
    probs = model_probs(model, stream, n_windows)
    track = fsm_run(probs, ALERT_CONFIDENCE)

    bin_path = args.out + ".bin"
    csv_path = args.out + ".csv"
    ref_path = args.out + ".ref.bin"
    write_model(bin_path, model, cfg)
    np.savetxt(csv_path, stream, fmt="%d", delimiter=",")
    write_ref(ref_path, probs, ALERT_CONFIDENCE, track)

    names = ["normal", "inner_race", "outer_race", "rolling_element"]
    print(f"wrote {bin_path}")
    print(f"wrote {csv_path}  ({len(stream)} samples, {n_windows} windows)")
    print(f"wrote {ref_path}")
    print(f"per-window decision (threshold {ALERT_CONFIDENCE:.2f}):")
    for w, (p, (state, latched)) in enumerate(zip(probs, track)):
        idx = int(np.argmax(p))
        st = "NORMAL" if state == 0 else f"ALERT:{names[latched]}"
        print(f"  win {w}: top={names[idx]:<14} conf={p[idx]:.4f}  -> {st}")


if __name__ == "__main__":
    main()
