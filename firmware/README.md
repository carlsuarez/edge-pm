# firmware — STM32F411 Cortex-M4F application

The hardware application: an `embassy-stm32` async executor brings up the ADXL345 over
SPI2 + DMA, and the real-time loop drives [`pmcore`](../pmcore)'s `features → model → alert`
pipeline, the on-board alert LED, and a UART2 debug log. It runs the **same** `pmcore` code
the host simulator proved out — only the sample source differs (a real accelerometer here, a
CSV file in `host-sim`).

**Status: Renode-validated, fp32 + int8.** Builds for `thumbv7em-none-eabihf` (dev +
release), all four configs clippy clean, fits the chip with room to spare, and the
`--features sim` builds boot in **Renode** emulation and reproduce the host gate's alert
trajectory on the real Cortex-M4F ISA — in both fp32 and integer-only int8 (see
[Emulation](#emulation-renode)).

Measured release footprint (`size`):

| region | fp32 | int8 (`--features q8`) | of | |
|--------|------|------------------------|----|--|
| flash (code + rodata + embedded model) | 92.4 KB | 87.0 KB | 512 KB | 17–18% |
| RAM (`bss`) | 44.6 KB | 16.4 KB | 128 KB | 13–35% |
| └ forward-pass arena | 37.7 KB | 9.4 KB | | |
| embedded model weights | 12,524 B | 3,744 B | | 3.3× |

The forward-pass arena and run state are carved **once** at boot from a `static` buffer, so
the acquisition loop allocates nothing — the same contract `pmcore` enforces. The
representation is fixed at build time (`q8` cargo feature), so the int8 build carves only the
smaller `i8` arena: that is where the RAM win comes from. (The flash figures are ~13 KB above
the earlier time-domain-only build: `pmcore::features` now also runs a Hann-windowed real FFT
per axis — from `tiny-infer`'s `engine::dsp` — for the log-spaced spectral band features, and
the dense layer widened with them. The FFT scratch is on the stack, so `bss` is unchanged.)

## Acquisition (hardware build)

The ADXL345 runs in **FIFO stream mode** (32-deep) at its **3200 Hz** max ODR (`BW_RATE`
rate code `0x0F`; the rate the training data is decimated to) and raises **INT1** every 16
samples (watermark). The `sampler` task (`src/sampler.rs`) sleeps on that interrupt via async EXTI,
and only wakes to drain the buffered burst out over SPI — so the CPU is idle between
watermarks instead of polling. Completed `[Sample; 512]` windows are handed to the inference
loop in `main` over two `embassy_sync` channels (a free-buffer queue and a full-buffer
queue), which double-buffers acquisition against inference with no shared mutable state.

## Pin map (Nucleo-F411RE)

- **LD2 alert LED** — PA5
- **USART2 debug log** — PA2 (TX) / PA3 (RX), DMA1 streams 6/5
- **ADXL345 over SPI2** — PB13 (SCK) / PB15 (MOSI) / PB14 (MISO), CS PB12, DMA1 streams 4/3
- **ADXL345 INT1** — PB1 (EXTI1), active-high FIFO watermark

SPI2 rather than SPI1 because SPI1's SCK is PA5 — the on-board LED — on this board.

## Build

```sh
rustup target add thumbv7em-none-eabihf      # Cortex-M4F hard-float (once)
python ../tools/export_model.py --out ../models/bearing_cnn.bin --quantize   # the embedded model(s)
cd firmware && cargo build                    # fp32 (uses .cargo/config.toml + memory.x)
cargo build --features q8                     # integer-only int8 build
```

The model is compiled into flash from `../models/bearing_cnn.bin` (gitignored, like the host
test fixtures) — or `bearing_cnn_q8.bin` for the `q8` build — so generate it first
(`--quantize` writes both). Its layer dimensions must match the `CFG` constant in
`src/main.rs`; the boot code asserts this against the loaded header.

## Emulation (Renode)

Renode has no SPI ADXL345 model (its `Sensors.ADXL345` is I2C-only), so the emulation build
swaps the sensor for a flash-baked sample stream instead of faking the SPI wire protocol:

```sh
python ../tools/make_stream.py --out ../models/bearing_stream --quantize   # demo model(s) + stream + ref
python ../tools/export_sim_stream.py                                       # stream -> .samples.bin blob

renode/run.sh          # fp32 sim: build (--features sim) + boot in Renode + print the UART log
renode/run.sh --q8     # int8 sim: the integer-only build
renode/run.sh --test   # headless robot test (fp32; --q8 for int8)
```

`run.sh` assumes `renode` and `renode-test` are on `PATH`. **Note:** `cargo run` is *not* the
emulation path — the `.cargo/config.toml` runner is `probe-rs`, which flashes a real board;
Renode loads the ELF directly, which is what `run.sh` does. To drive Renode by hand instead:

```sh
cargo build --features sim                                 # (or --features sim,q8)
renode-test renode/edge-pm.robot                          # headless CI test (edge-pm-q8.robot for int8)
renode --console -e 'include @renode/edge-pm.resc; start' # interactive (in a TTY)
```

> **Gotcha:** the robots `LoadELF` the **debug** ELF (`target/thumbv7em-none-eabihf/debug/`),
> not release. After changing the firmware, rebuild the matching debug ELF
> (`cargo build --features sim[,q8]`, no `--release`) before `renode-test`, or it silently
> runs a stale binary. `run.sh` builds the right one for you.

The `sim` feature (see `src/sim_source.rs`) reads `bearing_stream` from flash and drives the
**same** `windowing → features → CNN → alert` pipeline and UART log; it bakes the
`bearing_stream` demo model (whose hand-wired head latches `outer_race`) rather than the
deployed `bearing_cnn.bin`. The run reproduces the host gate exactly — UART shows
`ALERT outer_race conf=<C>` (the latch) then `CLEAR` (the 3-normal-window hysteresis) then
`sim stream complete`, where `C` matches `bearing_stream.ref.bin`'s PyTorch softmax: **1.000**
for the fp32 build (`edge-pm.robot`) and **0.996** for the integer-only int8 build
(`edge-pm-q8.robot`). The int8 figure is computed entirely in integer arithmetic on the
Cortex-M4F — float appears only at the final logit dequantize + softmax.

This validates the on-target compute + control flow + UART; the ADXL345 SPI/DMA driver and
the FIFO/watermark acquisition path are build-checked and exercised on real hardware (Renode
has no ADXL345 SPI model and cannot toggle INT1; the boot `device_id()` DMA read does work
against a `Sensors.GenericSPISensor` stub, which is how the SPI path was smoke-tested).

## Files

- `src/main.rs`        — embassy executor, peripheral init, the acquisition + inference loop (cfg-split: hardware / `sim`, fp32 / `q8`)
- `src/sampler.rs`     — hardware acquisition task: ADXL345 FIFO + watermark interrupt → window channel
- `src/adxl345.rs`     — ADXL345 SPI driver (FIFO stream config + register/burst reads)
- `src/sim_source.rs`  — flash-baked `bearing_stream` source for the `--features sim` build
- `renode/run.sh`         — build the sim ELF + run it in Renode (`--test` robot test, `--q8` int8 build)
- `renode/stm32f411.repl` — F411 platform (specialises Renode's bundled STM32F4 description)
- `renode/edge-pm.resc`   — load + run the sim ELF, mirror USART2 to the console
- `renode/edge-pm.robot`  — CI test (fp32 sim): boot the ELF and assert the alert latch/clear log
- `renode/edge-pm-q8.robot` — CI test (int8 sim): same trajectory, integer-only on-target
- `memory.x`           — STM32F411RE linker layout (512K flash / 128K SRAM)
- `build.rs`           — puts `memory.x` on the linker search path
- `.cargo/config.toml` — target, linker args, flash/run runner

## Dependency notes (embassy 0.9 / stm32 0.6 wave)

Getting a self-consistent embassy set is fiddly. The working combination is **embassy-stm32
0.6** + **embassy-executor 0.9** + **embassy-time 0.4** (+ `embassy-time-queue-utils` 0.3,
pulled transitively). Gotchas, all linker/build errors otherwise:

- The async (DMA) `Uart::new` / `Spi::new` take their interrupt-binding argument *after* the
  DMA channels and require the **DMA stream** interrupts bound too (not just USART2) — see the
  `bind_interrupts!` block. In stm32 0.6 `Spi` also takes two generics (`Spi<'d, Async, Master>`).
- Async EXTI is bound manually too: `EXTI1 => exti::InterruptHandler<..::EXTI1>` in
  `bind_interrupts!` (cfg-gated to the hardware build, alongside the SPI DMA stream IRQs).
- embassy-time's integrated timer queue needs `__embassy_time_queue_item_from_waker`, which is
  only provided by **embassy-executor ≥ 0.9** (it owns the split-out `embassy-executor-timer-queue`
  crate). 0.7 predates the split and fails to link. 0.9 also dropped the `task-arena-size-*`
  feature (tasks are statically sized now).
- Renode's STM32 UART model does **not** raise the TX DMA transfer-complete interrupt, so
  `uart.write().await` (DMA) hangs forever — the log path uses `uart.blocking_write` instead.
