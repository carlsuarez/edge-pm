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
import math
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


def quantize_rows(w2d):
    """Per-output-row symmetric int8 quantization, matching engine::quant::quantize.

    `w2d` is a `[d_out, d_in]` float array; each row (one output channel) gets a single
    scale `max(|row|)/127` so its largest magnitude maps to +-127. Returns the int8 values
    `[d_out, d_in]` and the per-row f32 scales `[d_out]`. An all-zero row gets scale 0 and
    quantizes to all zeros (no divide). Rounding is half-away-from-zero, matching libm roundf.
    """
    d_out = w2d.shape[0]
    q = np.zeros_like(w2d, dtype=np.int8)
    scales = np.zeros(d_out, dtype="<f4")
    for o in range(d_out):
        row = w2d[o].astype(np.float64)
        scale = float(np.max(np.abs(row))) / 127.0
        scales[o] = np.float32(scale)
        if scale != 0.0:
            qr = np.sign(row) * np.floor(np.abs(row) / scale + 0.5)
            q[o] = np.clip(qr, -127, 127).astype(np.int8)
    return q, scales


def quantize_multiplier(m):
    """Split a real multiplier m>0 into a Q31 mantissa in [2^30, 2^31) and a signed shift,
    matching gemmlowp/TFLite `QuantizeMultiplier` and engine's `requantize`. m == 0 -> (0, 0)."""
    if m == 0.0:
        return 0, 0
    frac, shift = math.frexp(m)            # m = frac * 2^shift, frac in [0.5, 1)
    q = int(round(frac * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        shift += 1
    return q, shift


def calibrate(model, windows, feats_list):
    """Per-tensor activation ranges over a calibration set, for static quantization.

    Returns max|.| of: the raw window (conv1 input), conv1 output (post-ReLU), conv2 output
    (post-ReLU), and the dense input (pooled conv2 concatenated with the 9 features). These
    fix the activation scales the integer-only forward pass uses at runtime.
    """
    a_win = a_c1 = a_c2 = a_fc = 0.0
    with torch.no_grad():
        for win, feats in zip(windows, feats_list):
            x = torch.tensor(win.T[None].astype("float32"))          # (1, N_AXES, WINDOW_LEN)
            f = torch.tensor(np.asarray(feats, dtype="float32")[None])
            c1 = F.relu(model.conv1(x))
            c2 = F.relu(model.conv2(c1))
            fcin = torch.cat([c2.mean(dim=-1), f], dim=-1)
            a_win = max(a_win, float(np.abs(win).max()))
            a_c1 = max(a_c1, float(c1.abs().max()))
            a_c2 = max(a_c2, float(c2.abs().max()))
            a_fc = max(a_fc, float(fcin.abs().max()))
    # A dead (all-zero) tensor gets scale 1.0 so its multipliers stay finite (its quantized
    # values are zero regardless).
    scale = lambda a: (a / 127.0) if a > 0 else 1.0
    return scale(a_win), scale(a_c1), scale(a_c2), scale(a_fc)


def write_quantized_model(path, model, cfg, cal_windows, cal_feats):
    """Serialize the model in the v2 **integer-only** (static int8) format that
    `pmcore::model::QuantizedWeights` loads — the CMSIS-NN / TFLite scheme.

    Weights are int8 (per-output-channel scale); activations are int8 at calibration-fixed
    per-tensor scales; biases are i32 (pre-scaled to each layer's accumulator domain); and the
    per-channel requantizers `(mult, shift)` rescale each i32 accumulator to the next layer's
    int8 domain with no float at inference. Layout after the 64-byte v2 header:
      f32 block: s_in0, s_fc_in, fc_out_scale[N_CLASSES], fc_bias[N_CLASSES]
      i32 block: conv1 (bias, mult, shift)[c1], conv2 (bias, mult, shift)[c2],
                 pool_mult, pool_shift
      i8  block: conv1_w, conv2_w, fc_w

    The dense bias stays f32 (added at the final dequant) so it survives an all-zero weight
    row, whose s_w = 0 would otherwise collapse an i32-bias accumulator domain.
    """
    c1, c2 = cfg["c1"], cfg["c2"]
    l2 = ((((WINDOW_LEN - cfg["k1"]) // cfg["s1"] + 1) - cfg["k2"]) // cfg["s2"] + 1)

    # Calibrated per-tensor activation scales.
    s_in0, s_c1, s_c2, s_fc = calibrate(model, cal_windows, cal_feats)

    # Per-output-channel weight quantization.
    q1, sw1 = quantize_rows(model.conv1.weight.detach().cpu().numpy().reshape(c1, -1))
    q2, sw2 = quantize_rows(model.conv2.weight.detach().cpu().numpy().reshape(c2, -1))
    q3, swf = quantize_rows(model.fc.weight.detach().cpu().numpy().reshape(N_CLASSES, -1))
    b1 = model.conv1.bias.detach().cpu().numpy()
    b2 = model.conv2.bias.detach().cpu().numpy()
    bf = model.fc.bias.detach().cpu().numpy()

    # Conv i32 biases (pre-scaled to the accumulator domain) + per-channel requant multipliers.
    conv1_bias = np.round(b1 / (s_in0 * sw1)).astype("<i4")
    conv2_bias = np.round(b2 / (s_c1 * sw2)).astype("<i4")
    conv1_qm = [quantize_multiplier(s_in0 * sw1[o] / s_c1) for o in range(c1)]
    conv2_qm = [quantize_multiplier(s_c1 * sw2[o] / s_c2) for o in range(c2)]
    pool_qm = quantize_multiplier(s_c2 / (l2 * s_fc))
    fc_out_scale = (s_fc * swf).astype("<f4")
    # The dense bias stays f32 (added after the dequant).
    fc_bias = bf.astype("<f4")

    def i32a(vals):
        return np.asarray(vals, dtype="<i4").tobytes()

    header = bytearray(HEADER_BYTES)
    header[:4] = MAGIC
    fields = [2, N_AXES, WINDOW_LEN, cfg["c1"], cfg["k1"], cfg["s1"],
              cfg["c2"], cfg["k2"], cfg["s2"], FEATURE_LEN, N_CLASSES]
    for i, v in enumerate(fields):
        header[4 + i * 4: 8 + i * 4] = int(v).to_bytes(4, "little", signed=True)

    with open(path, "wb") as f:
        f.write(header)
        # f32 block.
        f.write(np.asarray([s_in0, s_fc], dtype="<f4").tobytes())
        f.write(fc_out_scale.tobytes())
        f.write(fc_bias.tobytes())
        # i32 block.
        f.write(conv1_bias.tobytes())
        f.write(i32a([m for m, _ in conv1_qm]))
        f.write(i32a([s for _, s in conv1_qm]))
        f.write(conv2_bias.tobytes())
        f.write(i32a([m for m, _ in conv2_qm]))
        f.write(i32a([s for _, s in conv2_qm]))
        f.write(i32a([pool_qm[0], pool_qm[1]]))
        # i8 block.
        for q in (q1, q2, q3):
            f.write(q.ravel().astype(np.int8).tobytes())


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
    ap.add_argument("--quantize", action="store_true",
                    help="also emit an int8 (W8A8) v2 model alongside, <stem>_q8.bin")
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

    if args.quantize:
        q8_out = stem + "_q8.bin"
        # Calibrate the activation scales on the single deterministic window.
        write_quantized_model(q8_out, model, cfg, [window], [feats])
        print(f"wrote {q8_out}  (int8 integer-only, v2)")


if __name__ == "__main__":
    main()
