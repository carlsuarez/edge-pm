#!/usr/bin/env python3
"""Export a trained 1-D CNN to edge-pm's flat binary weight format.

Trained offline in PyTorch on the CWRU Bearing Dataset, the model is serialized as a
header of layer dimensions followed by raw little-endian f32 weight arrays in row-major
order — the same loading convention as tiny-infer's Llama checkpoint, so the no_std
loader in `pmcore::model` can mmap/read it directly. The Conv1d weight layout matches
`engine::nn::conv1d`'s expectation: [c_out, c_in, k].

Milestone C — not yet implemented.

Usage (planned):
    python export_model.py --checkpoint model.pt --out ../models/bearing_cnn.bin
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="trained PyTorch checkpoint (.pt)")
    ap.add_argument("--out", required=True, help="output flat binary (.bin)")
    ap.add_argument("--quantize", action="store_true", help="export int8 weights (stretch)")
    ap.parse_args()
    raise SystemExit("export_model.py: not yet implemented (Milestone C)")


if __name__ == "__main__":
    main()
