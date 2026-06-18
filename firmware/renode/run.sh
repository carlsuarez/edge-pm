#!/usr/bin/env bash
#
# Build and run the edge-pm firmware in Renode emulation.
#
#   ./run.sh                  boot the sim firmware, run the baked stream, print the UART log, exit
#   ./run.sh --q8             boot the quantized (q8) sim firmware, run the stream, print UART log
#   ./run.sh --test           run the headless robot test
#   ./run.sh --test --q8      run the headless robot test for the quantized (q8) build
#
# Assumes `renode` (and, for --test, `renode-test`) is on PATH. `cargo run` is NOT the
# emulation path — that runner flashes real hardware via probe-rs; Renode loads the ELF
# directly, which is what this script does.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # firmware/renode
FIRMWARE_DIR="$(dirname "$HERE")"                    # firmware
cd "$FIRMWARE_DIR"

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: '$1' not found on PATH" >&2
    exit 1
  }
}

# --- Argument Parsing --------------------------------------------------------
TEST_MODE=false
Q8_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
  --test)
    TEST_MODE=true
    shift
    ;;
  --q8 | --quantized)
    Q8_MODE=true
    shift
    ;;
  *)
    echo "error: unknown argument '$1'" >&2
    echo "usage: $0 [--test] [--q8|--quantized]" >&2
    exit 1
    ;;
  esac
done

require cargo

# Determine cargo features and robot target based on flags
FEATURES="sim"
ROBOT_FILE="renode/edge-pm.robot"

if [[ "$Q8_MODE" == true ]]; then
  FEATURES="sim,q8"
  ROBOT_FILE="renode/edge-pm-q8.robot"
fi

echo ">>> building firmware (--features $FEATURES)"
cargo build --features "$FEATURES"

# --- robot-test mode ---------------------------------------------------------
if [[ "$TEST_MODE" == true ]]; then
  require renode-test
  echo ">>> renode-test $ROBOT_FILE"
  exec renode-test "$ROBOT_FILE"
fi

# --- one-shot boot mode ------------------------------------------------------
require renode
echo ">>> booting in Renode (the baked stream is finite; runs to completion)"

log="$(mktemp -t edge-pm-renode.XXXXXX.log)"
# Include the script, run a few host-seconds (the stream finishes in ~0.6s virtual), then
# quit. Capture the FULL output to a log (so failures are visible), `--plain` to drop colour
# codes, stdin from /dev/null and a timeout so it can never hang waiting for the monitor.
timeout 120 renode --disable-xwt --console --plain \
  -e "include @renode/edge-pm.resc; start; sleep 4; quit" \
  </dev/null >"$log" 2>&1 || true

# Pull out just the USART2 trajectory.
uart="$(grep -oE 'edge-pm: (boot|sim source[^[:cntrl:]]*|sim stream complete)|ALERT [a-z_]+ conf=[0-9.]+|CLEAR' "$log" || true)"
if [[ -n "$uart" ]]; then
  printf '%s\n' "$uart"
  rm -f "$log"
else
  echo "!! No UART output captured — Renode did not run the firmware as expected." >&2
  echo "   Renode errors:" >&2
  grep -iE 'error|exception|does not exist|^E[0-9]' "$log" | head -10 | sed 's/^/     /' >&2 ||
    echo "     (none found in the log)" >&2
  echo "   Full Renode log: $log" >&2
  echo "   Tip: 'firmware/renode/run.sh --test' runs the same thing via the robot harness." >&2
  exit 1
fi
