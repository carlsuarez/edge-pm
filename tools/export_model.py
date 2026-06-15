#!/usr/bin/env python3
"""Export the bearing-health 1-D CNN to edge-pm's flat binary weight format.

The model is the hybrid CNN that `pmcore::model` runs: two `Conv1d -> ReLU` blocks over
the raw window (N_AXES channels x WINDOW_LEN samples), a global average pool over time,
the result concatenated with the 9 hand-crafted features, then a dense layer to N_CLASSES
logits. This script serializes its weights as a 64-byte header (magic + dims) followed by
the raw little-endian f32 tensors in PyTorch order -- the load-from-flash convention
`pmcore::model::Model::load` expects.

With no --checkpoint it builds a deterministic randomly-initialized model (enough to
validate the no_std forward pass numerically). It also writes a deterministic input window
and the reference features + softmax probs, so `host-sim infer` can be checked against
PyTorch on identical input -- the Milestone C gate.

Usage:
    python export_model.py --out models/bearing_cnn.bin [--checkpoint model.pt] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_features import features as compute_features, gen as gen_window, load_csv

# Fixed pipeline dimensions (must match pmcore::features / pmcore::model).
N_AXES = 3
WINDOW_LEN = 512
FEATURE_LEN = 9
N_CLASSES = 4
MAGIC = b"epm1"
HEADER_BYTES = 64

# Default convolutional layer dims (overridable below); l1=253, l2=125 over WINDOW_LEN.
DEFAULT_CFG = dict(c1=16, k1=7, s1=2, c2=32, k2=5, s2=2)


class BearingCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(N_AXES, cfg["c1"], cfg["k1"], stride=cfg["s1"])
        self.conv2 = nn.Conv1d(cfg["c1"], cfg["c2"], cfg["k2"], stride=cfg["s2"])
        self.fc = nn.Linear(cfg["c2"] + FEATURE_LEN, N_CLASSES)

    def forward(self, x, feats):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=-1)                      # global average pool over time
        x = torch.cat([x, feats], dim=-1)       # fuse the hand-crafted features
        return self.fc(x)                       # logits


def write_model(path, model, cfg):
    header = bytearray(HEADER_BYTES)
    header[:4] = MAGIC
    fields = [1, N_AXES, WINDOW_LEN, cfg["c1"], cfg["k1"], cfg["s1"],
              cfg["c2"], cfg["k2"], cfg["s2"], FEATURE_LEN, N_CLASSES]
    for i, v in enumerate(fields):
        header[4 + i * 4: 8 + i * 4] = int(v).to_bytes(4, "little", signed=True)

    tensors = [model.conv1.weight, model.conv1.bias,
               model.conv2.weight, model.conv2.bias,
               model.fc.weight, model.fc.bias]
    with open(path, "wb") as f:
        f.write(header)
        for t in tensors:
            f.write(t.detach().cpu().numpy().astype("<f4").ravel().tobytes())


def write_ref(path, feats, probs):
    with open(path, "wb") as f:
        f.write(int(FEATURE_LEN).to_bytes(4, "little"))
        f.write(int(N_CLASSES).to_bytes(4, "little"))
        f.write(np.asarray(feats, dtype="<f4").tobytes())
        f.write(np.asarray(probs, dtype="<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output model .bin")
    ap.add_argument("--checkpoint", help="trained state_dict (.pt); else random init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    cfg = DEFAULT_CFG
    torch.manual_seed(args.seed)
    model = BearingCNN(cfg).eval()
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # Deterministic reference window + features (numpy reference == pmcore to ~1e-6).
    stem = args.out[:-4] if args.out.endswith(".bin") else args.out
    win_csv = stem + ".window.csv"
    ref_bin = stem + ".ref.bin"
    gen_window(win_csv, seed=args.seed, n=WINDOW_LEN)
    window = load_csv(win_csv)
    feats = compute_features(window)

    x = torch.tensor(window.T[None].astype("float32"))          # (1, N_AXES, WINDOW_LEN)
    f = torch.tensor(np.asarray(feats, dtype="float32")[None])  # (1, FEATURE_LEN)

    with torch.no_grad():
        # Normalize the dense layer so a randomly-initialized model gives a non-degenerate
        # (non-saturated) softmax on the test window -- makes the gate meaningful.
        s = model(x, f).std().item()
        if s > 1e-6:
            model.fc.weight.div_(s)
            model.fc.bias.div_(s)
        probs = torch.softmax(model(x, f), dim=-1)[0].numpy()

    write_model(args.out, model, cfg)
    write_ref(ref_bin, feats, probs)

    print(f"wrote {args.out}  (cfg {cfg})")
    print(f"wrote {win_csv}  ({WINDOW_LEN} samples)")
    print(f"wrote {ref_bin}")
    print("reference probs: " + " ".join(f"{p:.6f}" for p in probs))


if __name__ == "__main__":
    main()
