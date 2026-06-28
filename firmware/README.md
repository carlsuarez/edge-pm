# firmware — STM32F411 Cortex-M4F application

The hardware application: an `embassy-stm32` async executor brings up the ADXL345 over
SPI2 + DMA, and the real-time loop drives [`pmcore`](../pmcore)'s `features → model → alert`
pipeline, the on-board alert LED, and a UART2 debug log.

**Status: fp32 + int8.** Builds for `thumbv7em-none-eabihf` (dev + release), both configs
clippy clean, and fits the chip with room to spare.

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

## Acquisition

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

## Files

- `src/main.rs`        — embassy executor, peripheral init, the acquisition + inference loop (cfg-split: fp32 / `q8`)
- `src/sampler.rs`     — hardware acquisition task: ADXL345 FIFO + watermark interrupt → window channel
- `src/adxl345.rs`     — ADXL345 SPI driver (FIFO stream config + register/burst reads)
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
  `bind_interrupts!`, alongside the SPI DMA stream IRQs.
- embassy-time's integrated timer queue needs `__embassy_time_queue_item_from_waker`, which is
  only provided by **embassy-executor ≥ 0.9** (it owns the split-out `embassy-executor-timer-queue`
  crate). 0.7 predates the split and fails to link. 0.9 also dropped the `task-arena-size-*`
  feature (tasks are statically sized now).
