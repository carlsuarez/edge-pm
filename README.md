# edge-pm — on-device predictive-maintenance sensor node

A self-contained Rust firmware application for the **STM32F411 Nucleo** (Cortex-M4F) that
reads high-frequency vibration data from an **ADXL345** accelerometer over SPI + DMA,
extracts signal features on-chip, and runs a small **1-D CNN** to classify bearing health in
real time — no cloud, no WiFi, no OS, no heap in the hot path.

Bearing-health classes: `0 normal · 1 inner_race · 2 outer_race · 3 rolling_element`.

The same forward pass runs in **two numeric representations** from one source: fp32, and an
**integer-only int8** build (int8 weights + activations, `i32` accumulation, fixed-point
requantization — no float between layers) that cuts static RAM by **2.7×** on the
Cortex-M4F. Both are validated bit-for-bit against PyTorch on the host and on-target in
emulation.

The number-crunching (conv / relu / pooling / matmul kernels, fp32 and int8) comes from the
[`tiny-infer`](../tiny-infer) `engine` crate — a no_std, allocation-free kernel library
checked out beside this repo. edge-pm adds the feature extraction, the bearing model, the
real-time pipeline, and the firmware.

---

## How it fits together

One signal-processing pipeline, written once in a portable no_std library and run in three
places (laptop, emulator, real board):

```
        ┌──────────────────────────  pmcore (no_std core)  ──────────────────────────┐
 sample │  windowing       features::extract     model::forward        alert::Machine │
 stream │  512×[i16;3]  →   9 features        →   1-D CNN → softmax  →   NORMAL ⇄ ALERT│
        └────────────────────────────────────────────────────────────────────────────┘
              ▲                                                              │
   ADXL345 (firmware) · CSV file (host-sim) · baked blob (firmware sim)      ▼ LED + UART log
```

**Host-first development.** Everything that isn't hardware is built and validated *on the
laptop* against recorded/synthetic data before any board enters the picture. The portable
logic lives in **`pmcore`**, which the host harness and the firmware share **unchanged** — so
only the SPI/DMA bring-up is hardware-specific. The Python tools generate the models, the
data, and the independent references each stage is checked against. The emulator for the
hardware milestone is **Renode** (it models STM32F4 + SPI + DMA faithfully).

**How a window arrives is the caller's job.** `pmcore` exposes `process_window()` — the
`extract → forward → decide` step — but does not own the windowing: the firmware's `sampler`
task drains the ADXL345 FIFO on each watermark interrupt and hands full `[Sample; 512]`
buffers across an `embassy_sync` channel, while `host-sim` simply chops a recorded stream
into 512-sample slices. Both feed the *same* `process_window()`.

---

## Int8 quantization — the integer-only build

The forward pass is generic over its element type (`RunState<T>` / `Arena<T>`), so one code
path runs in fp32 (`RunState<f32>`) or int8 (`RunState<i8>`). The int8 build is **static,
integer-only** quantization: weights are int8 (per-output-channel scale), activations are
int8 at calibration-fixed per-tensor scales, accumulation is `i32`, and each layer rescales
with a fixed-point multiplier (`mult`·2^`shift`) — **no floating point between layers**.
Float appears only at the boundaries: quantizing the input window and the 9 features going
in, and dequantizing the four class logits for the closing softmax. (The dense layer's bias
stays fp32 and is added *after* the final dequant, which keeps a zero-weight class exact.)

Because the representation is fixed at build time, the firmware is `cfg`-gated on a `q8`
feature: the fp32 build carves a `RunState<f32>`, the q8 build carves only a `RunState<i8>` —
which is what actually realizes the RAM win on-device.

Measured footprint (release builds, `thumbv7em-none-eabihf`, via `size`):

| | fp32 | int8 (`--features q8`) | |
| --- | --- | --- | --- |
| flash — code + rodata + model | **80.0 KB** | **74.9 KB** | of 512 KB |
| static RAM — `bss` | **44.5 KB** | **16.4 KB** | of 128 KB |
| └ forward-pass arena | 38.5 KB | 9.4 KB | (the int8 win) |
| model weights (in flash) | 12,512 B | 3,748 B | **3.3× smaller** |

The big RAM drop is the arena: an int8 working set is a quarter the size of the fp32 one. On
the host the integer-only path tracks fp32 to ~3×10⁻³ absolute probability, and in Renode it
reproduces the host decision exactly (`conf=0.995`, see below).

---

## Repository map

```
edge-pm/
├── pmcore/                  no_std library — THE portable core (shared by host-sim + firmware)
│   └── src/
│       ├── features.rs        RMS / crest factor / kurtosis over a 512-sample window → [f32; 9]
│       ├── model.rs           1-D CNN: ModelConfig, zero-copy Weights/QuantizedWeights, `forward()` + `forward_q8()`
│       ├── state.rs           RunState<T> — forward-pass activation buffers, carved from an Arena<T> (T = f32 or i8)
│       ├── pipeline.rs        `process_window()` / `process_window_q8()` — the shared loop body
│       ├── alert.rs           AlertMachine — NORMAL ⇄ ALERT decision FSM with hysteresis
│       └── lib.rs             crate root + re-exports (Arena, RunState)
│
├── host-sim/                std binary — runs pmcore on the laptop against CSV/fixture data
│   ├── src/main.rs            `features` / `infer` / `replay` subcommands (auto-dispatch fp32 vs int8 by model version)
│   └── tests/
│       ├── infer.rs           gate: pmcore fp32 forward == PyTorch (Milestone C)
│       ├── infer_q8.rs        gate: pmcore integer-only int8 forward == PyTorch (Milestone F)
│       └── replay.rs          gate: pmcore probs == PyTorch AND alert FSM == Python (Milestone D)
│
├── firmware/                no_std Cortex-M4F binary (embassy-stm32) — EXCLUDED from the workspace
│   ├── src/
│   │   ├── main.rs            embassy executor, peripheral init, the acquisition + inference loop
│   │   ├── sampler.rs         hardware acquisition: ADXL345 FIFO + watermark interrupt → window channel
│   │   ├── adxl345.rs         ADXL345 SPI driver (FIFO stream config + register/burst reads)
│   │   └── sim_source.rs      flash-baked sample source (`--features sim`) — no sensor needed
│   ├── renode/               emulation: see firmware/README.md
│   │   ├── run.sh              build + run the sim firmware in Renode (`--q8` for the int8 build)
│   │   ├── edge-pm.resc        Renode script: load the ELF, mirror UART to the console
│   │   ├── edge-pm.robot       headless CI test (fp32 sim): assert the alert latch/clear trajectory
│   │   ├── edge-pm-q8.robot    headless CI test (int8 sim): same trajectory, integer-only on-target
│   │   └── stm32f411.repl      STM32F411 platform description
│   ├── memory.x / build.rs / .cargo/config.toml   linker layout, target, flash runner
│   └── README.md             firmware + emulation deep-dive (pin map, embassy version notes)
│
├── tools/                   Python — generate models, data, and reference outputs (see below)
├── models/                  generated fixtures (GITIGNORED — recreate with the tools)
└── README.md               (this file)
```

> **Why `pmcore` exists.** The original spec put everything under `firmware/`. Pulling the
> testable logic into a no_std library means feature extraction and inference can be unit-
> tested and replayed on the host, instead of being trapped behind a hardware binary. The
> firmware and host-sim then run *identical* code.

---

## The Python tools (`tools/`)

These run **offline on the host** to produce the models, the data streams, and the
independent reference outputs that the Rust gates check against. Everything they write lands
in `models/` (gitignored), so a fresh checkout regenerates it. They need a venv:

```sh
python -m venv venv && source venv/bin/activate
pip install -r tools/requirements.txt      # numpy, scipy, torch, pyserial
```

| Script | What it does | Writes (in `models/`) | Consumed by |
|--------|--------------|------------------------|-------------|
| **export_model.py** | Builds/serializes the bearing 1-D CNN to the flat `epm1` weight format, plus one deterministic input window and the PyTorch reference (features + softmax). `--quantize` also writes the int8 v2 model. | `bearing_cnn.bin`, `bearing_cnn.window.csv`, `bearing_cnn.ref.bin` (+ `bearing_cnn_q8.bin`) | `host-sim infer`, `tests/infer.rs` / `tests/infer_q8.rs`, **firmware default/q8 builds** |
| **make_stream.py** | Builds a *demo* model whose dense head is hand-wired so z-axis kurtosis drives `outer_race`, plus a multi-window sample stream that latches then clears an alert, plus the PyTorch probs + Python FSM trajectory. `--quantize` also writes the int8 v2 demo model. | `bearing_stream.bin` *(model!)*, `bearing_stream.csv` *(stream)*, `bearing_stream.ref.bin` (+ `bearing_stream_q8.bin`) | `host-sim replay`, `tests/replay.rs`, **firmware sim / sim+q8 builds** |
| **export_sim_stream.py** | Converts `bearing_stream.csv` → a raw little-endian `i16` blob the firmware `include_bytes!`s. Pure stdlib (no numpy). | `bearing_stream.samples.bin` | **firmware `--features sim`** |
| **verify_features.py** | numpy float64 reference for `pmcore::features` (RMS/crest/kurtosis). `--gen` also writes a deterministic test window. | (prints; `--gen` writes a CSV) | the Milestone B feature gate |
| **replay.py** | *Planned stub* — CWRU `.mat` ingestion (pack to a stream / stream over UART). Prints "not yet implemented". | — | — (future) |

> **Naming gotcha:** `make_stream.py` writes `bearing_stream.bin` — that is the **model**,
> not the data. The data stream is `bearing_stream.csv`. (`export_model.py`'s
> `bearing_cnn.bin` is a *different*, randomly-initialized model used only for the
> numeric-parity gate; it does not alert.)

**Regenerate every fixture** (run from the repo root, venv active):

```sh
python tools/export_model.py  --out models/bearing_cnn.bin --quantize     # infer gates + default/q8 firmware
python tools/make_stream.py   --out models/bearing_stream  --quantize     # replay gate + sim/sim+q8 firmware
python tools/export_sim_stream.py                                         # blob for the sim firmware
```

---

## Commands

> Prerequisites: Rust **stable** (`rust-toolchain.toml` pins it) and
> [`tiny-infer`](../tiny-infer) checked out **beside** this repo (`pmcore` depends on
> `../../tiny-infer/engine`). For the embedded targets:
> `rustup target add thumbv7em-none-eabi thumbv7em-none-eabihf`.

### Host simulator — `host-sim` (run pmcore on the laptop)

```sh
cargo run -p host-sim                                          # print usage
cargo run -p host-sim -- features models/bearing_cnn.window.csv          # Stage 2: the 9 features
cargo run -p host-sim -- infer    models/bearing_cnn.bin models/bearing_cnn.window.csv   # +Stage 3: class probs
cargo run -p host-sim -- infer    models/bearing_cnn_q8.bin models/bearing_cnn.window.csv # same, integer-only int8
cargo run -p host-sim -- replay   models/bearing_stream.bin models/bearing_stream.csv    # Stages 1–4: windowing + alert FSM
#   `infer` / `replay` auto-detect fp32 (v1) vs int8 (v2) from the model header.
#   add: --alert-confidence <f>   to override the 0.80 threshold in `replay`
```

### Firmware in emulation — `--features sim` + Renode

The sim build swaps the ADXL345 for the baked `bearing_stream` blob, so the whole pipeline +
UART log runs with no sensor model. `run.sh` builds and launches Renode for you (it expects
`renode` and `renode-test` on `PATH`):

```sh
firmware/renode/run.sh                  # fp32 sim: build + boot + print the UART log, then exit
firmware/renode/run.sh --q8             # int8 sim: the integer-only build, same trajectory
firmware/renode/run.sh --test           # headless robot test (fp32, asserts latch/clear)
firmware/renode/run.sh --test --q8      # headless robot test (int8)
```

Expected UART: `boot → sim source → ALERT outer_race conf=<C> → CLEAR → sim stream complete`,
where `C` is `1.000` for the fp32 build and `0.995` for the integer-only int8 build — the
same softmax the host produces, reproduced on the Cortex-M4F. See
[`firmware/README.md`](firmware/README.md) for the manual Renode invocation and details.

> `cargo run` is **not** the emulation path — the firmware's runner is `probe-rs`, which
> flashes a real board. Build the hardware firmware (real ADXL345 over SPI) with:
> `cd firmware && cargo build` (fp32) or `cargo build --features q8` (int8).

### Tests & checks

```sh
cargo test                                              # host: pmcore unit tests + host-sim gates
cargo clippy --all-targets -- -D warnings               # host lints
cargo build -p pmcore --target thumbv7em-none-eabi      # prove pmcore stays no_std

cd firmware                                             # all four configs lint clean:
cargo clippy -- -D warnings                             #   hardware fp32
cargo clippy --features q8 -- -D warnings               #   hardware int8
cargo clippy --features sim -- -D warnings              #   sim fp32
cargo clippy --features sim,q8 -- -D warnings           #   sim int8
```

> The `host-sim` gates (`tests/infer.rs`, `tests/infer_q8.rs`, `tests/replay.rs`) read
> fixtures from `models/` and **skip cleanly if they're absent** — generate them with the
> tools above to run them. The firmware robot tests (`run.sh --test [--q8]`) need
> `robotframework` etc. — install them from your Renode install's `tests/requirements.txt`
> (`pip install --user -r <renode>/tests/requirements.txt`).

---

## Roadmap (host-first order)

| #   | Milestone                                           | Where                      | Status |
| --- | --------------------------------------------------- | -------------------------- | ------ |
| A   | CNN ops (`conv1d`/`relu`/`global_avg_pool`)         | `tiny-infer` `engine::nn`  | ✅ done |
| B   | Feature extraction (RMS, crest, kurtosis)           | `pmcore::features`         | ✅ done (vs `verify_features.py`) |
| C   | Model format + loader + forward pass                | `pmcore::model`            | ✅ done (bit-identical to PyTorch) |
| D   | Pipeline + decision state machine                   | `pmcore::{pipeline,alert}` | ✅ done (replay gate) |
| E   | Firmware: embassy, SPI/DMA bring-up, real-time loop | `firmware/`                | ✅ done — **Renode-validated** |
| F   | Int8 integer-only quantization (W8A8)               | `engine::nn`/`quant` + `pmcore` | ✅ done — host + on-target (Renode) |

All planned milestones are complete. The fp32 and int8 paths are unified behind a single
generic `RunState<T>` / `Arena<T>`, so there is one forward pass templated on the element
type rather than two parallel implementations.
