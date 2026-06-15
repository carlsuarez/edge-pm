"""Replay CWRU vibration data for edge-pm.

Two modes:
  * pack  — convert a CWRU recording into the flat window-stream binary that `host-sim`
            replays through pmcore on the host (Milestones B-D).
  * uart  — stream the same data over a serial port to the board, for end-to-end
            validation against the firmware (Milestone E / 6; needs pyserial).

Not yet implemented.

Usage (planned):
    python replay.py pack --input cwru.mat --out ../data/stream.bin
    python replay.py uart --input cwru.mat --port /dev/ttyACM0 --baud 115200
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    pack = sub.add_parser("pack", help="write a host-sim window-stream binary")
    pack.add_argument("--input", required=True)
    pack.add_argument("--out", required=True)

    uart = sub.add_parser("uart", help="stream samples to the board over serial")
    uart.add_argument("--input", required=True)
    uart.add_argument("--port", required=True)
    uart.add_argument("--baud", type=int, default=115200)

    ap.parse_args()
    raise SystemExit("replay.py: not yet implemented")


if __name__ == "__main__":
    main()
