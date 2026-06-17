# edge-pm — on-device predictive-maintenance sensor node

A self-contained Rust firmware application for the **STM32F411 Nucleo** (Cortex-M4F) that
reads high-frequency vibration data from an **ADXL345** accelerometer over SPI + DMA,
extracts signal features on-chip, and runs a small **1-D CNN** to classify bearing health in
real time — no cloud, no WiFi, no OS, no heap in the hot path.

Bearing-health classes: `0 normal · 1 inner_race · 2 outer_race · 3 rolling_element`.

The number-crunching (conv / relu / pooling / matmul kernels) comes from the
[`tiny-infer`](../tiny-infer) `engine` crate — a no_std, allocation-free kernel library
checked out beside this repo. edge-pm adds the feature extraction, the bearing model, the
real-time pipeline, and the firmware.

---

## How it fits together

One signal-processing pipeline, written once in a portable no_std library and run in three
places (laptop, emulator, real board):

```
        ┌──────────────────────────  pmcore (no_std core)  ──────────────────────────┐
 sample │  Windower        features::extract     model::forward        alert::Machine │
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

---

## Repository map

```
edge-pm/
├── pmcore/                  no_std library — THE portable core (shared by host-sim + firmware)
│   └── src/
│       ├── features.rs        RMS / crest factor / kurtosis over a 512-sample window → [f32; 9]
│       ├── model.rs           1-D CNN: ModelConfig, zero-copy Weights, free `forward()` + softmax
│       ├── state.rs           RunState — all forward-pass activation buffers, carved from an arena
│       ├── pipeline.rs        Windower (ring-buffer handoff) + `process_window()` loop body
│       ├── alert.rs           AlertMachine — NORMAL ⇄ ALERT decision FSM with hysteresis
│       └── lib.rs             crate root + re-exports (Arena, RunState)
│
├── host-sim/                std binary — runs pmcore on the laptop against CSV/fixture data
│   ├── src/main.rs            `features` / `infer` / `replay` subcommands (see Commands)
│   └── tests/
│       ├── infer.rs           gate: pmcore forward pass == PyTorch (Milestone C)
│       └── replay.rs          gate: pmcore probs == PyTorch AND alert FSM == Python (Milestone D)
│
├── firmware/                no_std Cortex-M4F binary (embassy-stm32) — EXCLUDED from the workspace
│   ├── src/
│   │   ├── main.rs            embassy executor, peripheral init, the acquisition + inference loop
│   │   ├── adxl345.rs         ADXL345 SPI driver (default build) — register config + burst read
│   │   └── sim_source.rs      flash-baked sample source (`--features sim`) — no sensor needed
│   ├── renode/               emulation: see firmware/README.md
│   │   ├── run.sh              build + run the sim firmware in Renode (start here)
│   │   ├── edge-pm.resc        Renode script: load the ELF, mirror UART to the console
│   │   ├── edge-pm.robot       headless CI test: assert the alert latch/clear trajectory
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
| **export_model.py** | Builds/serializes the bearing 1-D CNN to the flat `epm1` weight format, plus one deterministic input window and the PyTorch reference (features + softmax). | `bearing_cnn.bin`, `bearing_cnn.window.csv`, `bearing_cnn.ref.bin` | `host-sim infer`, `tests/infer.rs`, **firmware default build** |
| **make_stream.py** | Builds a *demo* model whose dense head is hand-wired so z-axis kurtosis drives `outer_race`, plus a multi-window sample stream that latches then clears an alert, plus the PyTorch probs + Python FSM trajectory. | `bearing_stream.bin` *(model!)*, `bearing_stream.csv` *(stream)*, `bearing_stream.ref.bin` | `host-sim replay`, `tests/replay.rs`, **firmware sim build** |
| **export_sim_stream.py** | Converts `bearing_stream.csv` → a raw little-endian `i16` blob the firmware `include_bytes!`s. Pure stdlib (no numpy). | `bearing_stream.samples.bin` | **firmware `--features sim`** |
| **verify_features.py** | numpy float64 reference for `pmcore::features` (RMS/crest/kurtosis). `--gen` also writes a deterministic test window. | (prints; `--gen` writes a CSV) | the Milestone B feature gate |
| **replay.py** | *Planned stub* — CWRU `.mat` ingestion (pack to a stream / stream over UART). Prints "not yet implemented". | — | — (future) |

> **Naming gotcha:** `make_stream.py` writes `bearing_stream.bin` — that is the **model**,
> not the data. The data stream is `bearing_stream.csv`. (`export_model.py`'s
> `bearing_cnn.bin` is a *different*, randomly-initialized model used only for the
> numeric-parity gate; it does not alert.)

**Regenerate every fixture** (run from the repo root, venv active):

```sh
python tools/export_model.py      --out models/bearing_cnn.bin     # infer gate + default firmware
python tools/make_stream.py       --out models/bearing_stream      # replay gate + sim firmware
python tools/export_sim_stream.py                                  # blob for the sim firmware
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
cargo run -p host-sim -- replay   models/bearing_stream.bin models/bearing_stream.csv    # Stages 1–4: windowing + alert FSM
#   add: --alert-confidence <f>   to override the 0.80 threshold in `replay`
```

### Firmware in emulation — `--features sim` + Renode

The sim build swaps the ADXL345 for the baked `bearing_stream` blob, so the whole pipeline +
UART log runs with no sensor model. `run.sh` builds and launches Renode for you (it expects
`renode` and `renode-test` on `PATH`):

```sh
firmware/renode/run.sh            # build (--features sim) + boot + print the UART log, then exit
firmware/renode/run.sh --test     # build + run the headless robot test (asserts latch/clear)
```

Expected UART: `boot → sim source → ALERT outer_race conf=0.966 → CLEAR → sim stream complete`.
See [`firmware/README.md`](firmware/README.md) for the manual Renode invocation and details.

> `cargo run` is **not** the emulation path — the firmware's runner is `probe-rs`, which
> flashes a real board. Build the hardware firmware (real ADXL345 over SPI) with:
> `cd firmware && cargo build` (or `cargo build --release`).

### Tests & checks

```sh
cargo test                                              # host: pmcore unit tests + host-sim gates
cargo clippy --all-targets -- -D warnings               # host lints
cargo build -p pmcore --target thumbv7em-none-eabi      # prove pmcore stays no_std

cd firmware
cargo clippy -- -D warnings                             # firmware lints (hardware build)
cargo clippy --features sim -- -D warnings              # firmware lints (sim build)
```

> The `host-sim` gates (`tests/infer.rs`, `tests/replay.rs`) read fixtures from `models/` and
> **skip cleanly if they're absent** — generate them with the tools above to run them. The
> firmware robot test (`run.sh --test`) needs `robotframework` etc. — install them from your
> Renode install's `tests/requirements.txt` (`pip install --user -r <renode>/tests/requirements.txt`).

---

## Roadmap (host-first order)

| #   | Milestone                                           | Where                      | Status |
| --- | --------------------------------------------------- | -------------------------- | ------ |
| A   | CNN ops (`conv1d`/`relu`/`global_avg_pool`)         | `tiny-infer` `engine::nn`  | ✅ done |
| B   | Feature extraction (RMS, crest, kurtosis)           | `pmcore::features`         | ✅ done (vs `verify_features.py`) |
| C   | Model format + loader + forward pass                | `pmcore::model`            | ✅ done (bit-identical to PyTorch) |
| D   | Pipeline + decision state machine                   | `pmcore::{pipeline,alert}` | ✅ done (replay gate) |
| E   | Firmware: embassy, SPI/DMA bring-up, real-time loop | `firmware/`                | ✅ done — **Renode-validated** |
| F   | Int8 weight quantization (stretch)                  | reuse `engine::quant`      | ⬜ planned |
