#!/usr/bin/env python3
"""Convert the Milestone-D sample stream into a flat blob the firmware bakes into flash.

`make_stream.py` writes `<stem>.csv` — the `x,y,z` int16 accelerometer stream that
`host-sim replay` and the PyTorch reference consume. The `sim` build of the firmware
(`cargo build --features sim`, used for Renode/CI bring-up) `include_bytes!`s the same data
so it can drive the on-device pipeline with no SPI/sensor model. This emits that data as
raw little-endian `i16`, x,y,z interleaved — exactly the layout `sim_source.rs` reads.

Usage:
    python tools/export_sim_stream.py            # models/bearing_stream.csv -> .samples.bin
    python tools/export_sim_stream.py --in <stem>.csv --out <stem>.samples.bin
"""

import argparse
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(ROOT, "models", "bearing_stream.csv"))
    ap.add_argument("--out", dest="out",
                    default=os.path.join(ROOT, "models", "bearing_stream.samples.bin"))
    args = ap.parse_args()

    n = 0
    with open(args.inp) as src, open(args.out, "wb") as dst:
        for lineno, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                raise SystemExit(f"{args.inp}:{lineno}: expected 3 columns, got {len(parts)}")
            x, y, z = (int(p) for p in parts)
            dst.write(struct.pack("<3h", x, y, z))  # little-endian i16 x,y,z
            n += 1

    print(f"wrote {args.out}  ({n} samples, {n * 6} bytes)")


if __name__ == "__main__":
    main()
