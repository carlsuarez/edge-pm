# edge-pm — on-device predictive-maintenance sensor node

A self-contained Rust firmware application for the **STM32F411 Nucleo** (Cortex-M4F)
that reads high-frequency vibration data from an **ADXL345** accelerometer over SPI +
DMA, extracts signal features on-chip, and runs a small **1-D CNN** to classify bearing
health in real time — no cloud, no WiFi, no OS, no heap in the hot path.

The inference runs on the [`tiny-infer`](../tiny-infer) `engine` crate (a no_std,
allocation-free transformer/CNN kernel library); edge-pm adds the feature extraction,
the bearing-health model, and the real-time pipeline around it.

Bearing-health classes: `0 normal · 1 inner-race · 2 outer-race · 3 rolling-element`.

## Development approach: host-first

Everything that isn't hardware is developed and tested **on the host** against recorded
data before any MCU emulation enters the picture. The portable logic lives in a no_std
library (`pmcore`) that the host harness and the firmware share unchanged, so feature
extraction and inference are validated on the laptop and only the SPI/DMA bring-up needs
the (emulated, then real) board. The emulator for the hardware milestones is **Renode**
— the one that faithfully models STM32F4 + SPI + DMA + a sensor fed from a sample file.

## Layout

```
edge-pm/
  pmcore/        no_std library — the portable core, shared by host-sim and firmware
    features.rs    RMS / crest factor / kurtosis / (optional) FFT over a window
    model.rs       1-D CNN: load weights, forward pass via engine::nn, softmax
    pipeline.rs    windowing / ring-buffer handoff (platform-agnostic parts)
    alert.rs       decision state machine (NORMAL ⇄ ALERT), thresholds
  host-sim/      std binary — replays recorded/synthetic windows through pmcore
  firmware/      no_std Cortex-M4F binary (embassy-stm32) — Milestone E; excluded
                 from the workspace (own target + linker script)
    adxl345.rs     SPI driver + register config
    main.rs        embassy executor, DMA wiring, real-time loop
    memory.x       STM32F411 linker layout (512K flash / 128K RAM)
  tools/         Python helpers (model export, feature reference, data replay)
```

This reshapes the original "everything under `firmware/`" sketch: the testable logic is
pulled into `pmcore` so it isn't trapped behind a no_std/hardware binary. The engine ops
(`conv1d`, `relu`, `global_avg_pool`) live upstream in `tiny-infer`'s `engine::nn`; the
bearing-specific _model_ lives here in `pmcore::model`.

## Dependency on tiny-infer

`pmcore` pulls the engine as a **path dependency**, expecting the two repos checked out
side by side:

```
<parent>/
  tiny-infer/    (engine crate)
  edge-pm/       (this repo)   ->  pmcore/Cargo.toml: engine = { path = "../../tiny-infer/engine" }
```

Both pin the same nightly toolchain (the engine's SIMD kernels need `core::simd`).

## Build & test

```sh
cargo test                                        # host: pmcore + host-sim
cargo clippy --all-targets
cargo build -p pmcore --target thumbv7em-none-eabi  # confirm pmcore stays no_std
cargo run -p host-sim                             # the replay harness
```

## Roadmap (host-first order)

| #   | Milestone                                           | Where                      | Emulation                            |
| --- | --------------------------------------------------- | -------------------------- | ------------------------------------ |
| A   | CNN ops (`conv1d`/`relu`/`global_avg_pool`)         | `tiny-infer` `engine::nn`  | none — **done**                      |
| B   | Feature extraction (RMS, crest, kurtosis, FFT)      | `pmcore::features`         | none (vs `tools/verify_features.py`) |
| C   | Model format + loader + forward pass                | `pmcore::model`            | none (vs laptop inference)           |
| D   | Pipeline + decision state machine                   | `pmcore::{pipeline,alert}` | none (replay CWRU windows)           |
| E   | Firmware: embassy, SPI/DMA bring-up, real-time loop | `firmware/`                | **Renode**                           |
| F   | Int8 weight quantization (stretch)                  | reuse `engine::quant`      | none                                 |
