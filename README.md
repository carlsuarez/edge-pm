# edge-pm — on-device predictive-maintenance sensor node

A self-contained Rust firmware application for the **STM32F411CE "Black Pill"** (Cortex-M4F)
that reads high-frequency vibration data from an **ADXL345** accelerometer over SPI + DMA,
extracts signal features on-chip, and runs a small **1-D CNN** to classify bearing health in
real time — no cloud, no WiFi, no OS, no heap in the hot path.

Bearing-health classes: `0 normal · 1 inner_race · 2 outer_race`.

The same forward pass runs in **two numeric representations** from one source: fp32, and an
**integer-only int8** build (int8 weights + activations, `i32` accumulation, fixed-point
requantization — no float between layers) that cuts static RAM by **2.7×** on the
Cortex-M4F. Both are validated bit-for-bit against PyTorch on the host.

The number-crunching (conv / relu / pooling / matmul kernels, fp32 and int8) comes from the
[`tiny-infer`](../tiny-infer) `engine` crate — a no_std, allocation-free kernel library
checked out beside this repo. edge-pm adds the feature extraction, the bearing model, the
real-time pipeline, and the firmware.

---

## How it fits together

One signal-processing pipeline, written once in a portable no_std library:

```
        ┌──────────────────────────  pmcore (no_std core)  ──────────────────────────┐
 sample │  windowing       features::extract     model::forward        alert::Machine │
 stream │  512×[i16;3]  →  24 features        →   1-D CNN → softmax  →   NORMAL ⇄ ALERT│
        └────────────────────────────────────────────────────────────────────────────┘
              ▲                                                              │
   ADXL345 over SPI + DMA (firmware)                                         ▼ LED + UART log
```

**Portable core.** The signal-processing and inference logic lives in **`pmcore`**, a no_std
library with no hardware dependencies, so it can be unit-tested on a laptop before it runs on
the board — only the SPI/DMA bring-up is hardware-specific. The Python tools generate the
model and the independent references each stage is checked against.

**How a window arrives is the caller's job.** `pmcore` exposes `process_window()` — the
`extract → forward → decide` step — but does not own the windowing: the firmware's `sampler`
task drains the ADXL345 FIFO on each watermark interrupt and hands full `[Sample; 512]`
buffers across an `embassy_sync` channel into `process_window()`.

---

## Int8 quantization — the integer-only build

The forward pass is generic over its element type (`RunState<T>` / `Arena<T>`), so one code
path runs in fp32 (`RunState<f32>`) or int8 (`RunState<i8>`). The int8 build is **static,
integer-only** quantization: weights are int8 (per-output-channel scale), activations are
int8 at calibration-fixed per-tensor scales, accumulation is `i32`, and each layer rescales
with a fixed-point multiplier (`mult`·2^`shift`) — **no floating point between layers**.
Float appears only at the boundaries: quantizing the input window and the 24 features going
in, and dequantizing the three class logits for the closing softmax. (The dense layer's bias
stays fp32 and is added *after* the final dequant, which keeps a zero-weight class exact.)

Because the representation is fixed at build time, the firmware is `cfg`-gated on a `q8`
feature: the fp32 build carves a `RunState<f32>`, the q8 build carves only a `RunState<i8>` —
which is what actually realizes the RAM win on-device.

Measured footprint (release builds, `thumbv7em-none-eabihf`, via `size`):

| | fp32 | int8 (`--features q8`) | |
| --- | --- | --- | --- |
| flash — code + rodata + model | **92.4 KB** | **87.0 KB** | of 512 KB |
| static RAM — `bss` | **44.6 KB** | **16.4 KB** | of 128 KB |
| └ forward-pass arena | 37.7 KB | 9.4 KB | (the int8 win) |
| model weights (in flash) | 12,524 B | 3,744 B | **3.3× smaller** |

The big RAM drop is the arena: an int8 working set is a quarter the size of the fp32 one. On
the host the integer-only path tracks fp32 to ~3×10⁻³ absolute probability. (The spectral
features added ~13 KB of flash over the time-domain-only build — the FFT code plus the wider
dense layer.)

---

## Repository map

```
edge-pm/
├── pmcore/                  no_std library — THE portable core (used by the firmware)
│   └── src/
│       ├── features.rs        per-axis RMS/crest/kurtosis + log-spaced FFT bands over a 512-sample window → [f32; 24]
│       ├── model.rs           1-D CNN: ModelConfig, zero-copy Weights/QuantizedWeights, `forward()` + `forward_q8()`
│       ├── state.rs           RunState<T> — forward-pass activation buffers, carved from an Arena<T> (T = f32 or i8)
│       ├── pipeline.rs        `process_window()` / `process_window_q8()` — the shared loop body
│       ├── alert.rs           AlertMachine — NORMAL ⇄ ALERT decision FSM with hysteresis
│       └── lib.rs             crate root + re-exports (Arena, RunState)
│
├── firmware/                no_std Cortex-M4F binary (embassy-stm32) — EXCLUDED from the workspace
│   ├── src/
│   │   ├── main.rs            embassy executor, peripheral init, the acquisition + inference loop
│   │   └── sampler.rs         hardware acquisition: drains the FIFO via the `adxl345-async`
│   │                          driver on each watermark interrupt → window channel
│   ├── memory.x / build.rs / .cargo/config.toml   linker layout, target, flash runner
│   └── README.md             firmware deep-dive (pin map, embassy version notes)
│
├── tools/                   Python — generate the model and reference outputs (see below)
├── models/                  generated fixtures (GITIGNORED — recreate with the tools)
└── README.md               (this file)
```

> **Why `pmcore` exists.** The original spec put everything under `firmware/`. Pulling the
> testable logic into a no_std library means feature extraction and inference can be
> unit-tested on the host, instead of being trapped behind a hardware binary.

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
| **export_model.py** | Builds/serializes the bearing 1-D CNN to the flat `epm1` weight format, plus one deterministic input window and the PyTorch reference (features + softmax). `--quantize` also writes the int8 v2 model. | `bearing_cnn.bin`, `bearing_cnn.window.csv`, `bearing_cnn.ref.bin` (+ `bearing_cnn_q8.bin`) | **firmware default/q8 builds** |
| **verify_features.py** | numpy float64 reference for `pmcore::features` (RMS/crest/kurtosis). `--gen` also writes a deterministic test window. | (prints; `--gen` writes a CSV) | the feature-extraction reference |
| **replay.py** | *Planned stub* — CWRU `.mat` ingestion (pack to a stream / stream over UART). Prints "not yet implemented". | — | — (future) |

**Regenerate the model fixtures** (run from the repo root, venv active):

```sh
python tools/export_model.py  --out models/bearing_cnn.bin --quantize     # default/q8 firmware
```

---

## Commands

> Prerequisites: Rust **stable** (`rust-toolchain.toml` pins it) and two sibling crates
> checked out **beside** this repo: [`tiny-infer`](../tiny-infer) (`pmcore` depends on
> `../../tiny-infer/engine`) and [`adxl345-async`](../adxl345-async) (the firmware's ADXL345
> driver, at `../../adxl345-async`). For the embedded targets:
> `rustup target add thumbv7em-none-eabi thumbv7em-none-eabihf`.

### Firmware — build & flash

The firmware's runner is `probe-rs`, which flashes a real board over a debug probe. Build the
firmware (real ADXL345 over SPI) with:

```sh
cd firmware && cargo build              # fp32
cargo build --features q8               # integer-only int8
cargo run                               # flash + run on a connected board (probe-rs)
```

See [`firmware/README.md`](firmware/README.md) for the pin map and embassy version notes.

### Training on real data — `tools/train_adxl355.py`

The deployed model is trained on the **ADXL355 triaxial induction-motor dataset** (Mendeley
DOI `10.17632/fm6xzxnf36.2`) — a MEMS triaxial accelerometer in the same family as the
firmware's ADXL345, with `normal` / `inner_race` / `outer_race` recordings. Download the CSVs
into `models/adxl355/` (gitignored), then:

```sh
python tools/train_adxl355.py --data models/adxl355 --out models/bearing.pt   # prints held-out acc + confusion
python tools/export_model.py --checkpoint models/bearing.pt --out models/bearing_cnn.bin --quantize
```

> **Sampling-rate note.** The dataset's recordings are only 0.1 s, shorter than the 512-sample
> window, so the loader decimates 10 kHz → **3200 Hz** (the ADXL345's max ODR, which the
> firmware's `BW_RATE` matches) and overlap-windows per class. The `normal` class has only two
> recordings, so it is data-starved and its held-out metric is weak — a documented limitation
> of this (otherwise ideal sensor- and taxonomy-matched) short dataset. Training normalizes
> inputs/features and folds the normalization back into the weights at export, so the deployed
> model consumes raw counts + raw features and the Rust forward pass is unchanged.

Without a checkpoint, `export_model.py` emits a deterministic random demo model — enough to
exercise the no_std forward pass on a clean checkout.

### Tests & checks

```sh
cargo test                                              # host: pmcore unit tests
cargo clippy --all-targets -- -D warnings               # host lints
cargo build -p pmcore --target thumbv7em-none-eabi      # prove pmcore stays no_std

cd firmware                                             # both firmware configs lint clean:
cargo clippy -- -D warnings                             #   fp32
cargo clippy --features q8 -- -D warnings               #   int8
```

---

## Architecture notes

The fp32 and int8 paths are unified behind a single generic `RunState<T>` / `Arena<T>`, so
there is one forward pass templated on the element type rather than two parallel
implementations. Feature extraction runs per-axis log-spaced **FFT band features** (a
Hann-windowed real FFT from `tiny-infer`'s `engine::dsp`, behind an opt-in `fft` feature)
alongside the time-domain RMS/crest/kurtosis stats. The taxonomy is the three classes the
real triaxial dataset provides (`normal` / `inner_race` / `outer_race`), and
`tools/train_adxl355.py` is the training pipeline that produces the deployed model.
