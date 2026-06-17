# firmware — STM32F411 Cortex-M4F shell (Milestone E)

The hardware application: an `embassy-stm32` async executor brings up the ADXL345 over
SPI2 + DMA, and the real-time loop drives [`pmcore`](../pmcore)'s `features → model → alert`
pipeline, the on-board alert LED, and a UART2 debug log. It runs the **same** `pmcore` code
the host simulator proved out — only the sample source differs (a real accelerometer here, a
CSV file in `host-sim`).

**Status: Renode-validated.** Builds for `thumbv7em-none-eabihf` (dev + release), clippy
clean, fits the chip with room to spare, and a `--features sim` build boots in **Renode**
emulation and reproduces the host gate's alert trajectory on the real Cortex-M4F ISA (see
[Emulation](#emulation-renode)).

| region | use | of | |
|--------|-----|----|--|
| flash  | ~80 KB (code + rodata + embedded model) | 512 KB | 16% |
| RAM    | ~46 KB (38.5 KB forward-pass arena + 6 KB double buffer + executor) | 128 KB | 36% |

The forward-pass arena and run state are carved **once** at boot from a `static` buffer, so
the acquisition loop allocates nothing — the same contract `pmcore` enforces.

## Pin map (Nucleo-F411RE)

- **LD2 alert LED** — PA5
- **USART2 debug log** — PA2 (TX) / PA3 (RX), DMA1 streams 6/5
- **ADXL345 over SPI2** — PB13 (SCK) / PB15 (MOSI) / PB14 (MISO), CS PB12, DMA1 streams 4/3

SPI2 rather than SPI1 because SPI1's SCK is PA5 — the on-board LED — on this board.

## Build

```sh
rustup target add thumbv7em-none-eabihf      # Cortex-M4F hard-float (once)
python ../tools/export_model.py --out ../models/bearing_cnn.bin   # the embedded model
cd firmware && cargo build                    # uses .cargo/config.toml + memory.x
```

The model is compiled into flash from `../models/bearing_cnn.bin` (gitignored, like the host
test fixtures), so generate it first. Its layer dimensions must match the `CFG` constant in
`src/main.rs` — the boot code asserts this against the loaded header.

## Emulation (Renode)

Renode has no SPI ADXL345 model (its `Sensors.ADXL345` is I2C-only), so the emulation build
swaps the sensor for a flash-baked sample stream instead of faking the SPI wire protocol:

```sh
python ../tools/make_stream.py --out ../models/bearing_stream   # demo model + stream + ref
python ../tools/export_sim_stream.py                            # stream -> .samples.bin blob

renode/run.sh          # build (--features sim) + boot in Renode + print the UART log
renode/run.sh --test   # build + run the headless robot test (asserts the trajectory)
```

`run.sh` assumes `renode` and `renode-test` are on `PATH`. **Note:** `cargo run` is *not* the
emulation path — the `.cargo/config.toml` runner is `probe-rs`, which flashes a real board;
Renode loads the ELF directly, which is what `run.sh` does. To drive Renode by hand instead:

```sh
cargo build --features sim
renode-test renode/edge-pm.robot                          # headless CI test
renode --console -e 'include @renode/edge-pm.resc; start' # interactive (in a TTY)
```

The `sim` feature (see `src/sim_source.rs`) reads `bearing_stream` from flash and drives the
**same** `windowing → features → CNN → alert` pipeline and UART log; it bakes the
`bearing_stream` demo model (whose hand-wired head latches `outer_race`) rather than the
deployed `bearing_cnn.bin`. The run reproduces the host gate exactly — UART shows
`ALERT outer_race conf=0.966` (the latch) then `CLEAR` (the 3-normal-window hysteresis) then
`sim stream complete`, and `conf=0.966` matches `bearing_stream.ref.bin`'s PyTorch softmax.
The robot test (`renode/edge-pm.robot`) asserts that trajectory; for a quick look without the
robot harness, `renode renode/edge-pm.resc` then `start` shows the same lines via the logging
UART analyzer.

This validates the on-target compute + control flow + UART; the ADXL345 SPI/DMA driver itself
is exercised on real hardware (and the boot `device_id()` DMA read does work in Renode against
a `Sensors.GenericSPISensor` stub, which is how the SPI path was smoke-tested).

## Files

- `src/main.rs`        — embassy executor, peripheral init, the acquisition + inference loop
- `src/adxl345.rs`     — ADXL345 SPI driver (register config + 3-axis burst read), hardware build
- `src/sim_source.rs`  — flash-baked `bearing_stream` source for the `--features sim` build
- `renode/run.sh`         — build the sim ELF + run it in Renode (`--test` for the robot test)
- `renode/stm32f411.repl` — F411 platform (specialises Renode's bundled STM32F4 description)
- `renode/edge-pm.resc`   — load + run the sim ELF, mirror USART2 to the console
- `renode/edge-pm.robot`  — CI test: boot the sim ELF and assert the alert latch/clear log
- `memory.x`           — STM32F411RE linker layout (512K flash / 128K SRAM)
- `build.rs`           — puts `memory.x` on the linker search path
- `.cargo/config.toml` — target, linker args, flash/run runner

## Dependency notes (embassy 0.9 / stm32 0.6 wave)

Getting a self-consistent embassy set is fiddly. The working combination is **embassy-stm32
0.6** + **embassy-executor 0.9** + **embassy-time 0.4** (+ `embassy-time-queue-utils` 0.3,
pulled transitively). Two gotchas, both linker errors otherwise:

- The async (DMA) `Uart::new` / `Spi::new` take their interrupt-binding argument *after* the
  DMA channels and require the **DMA stream** interrupts bound too (not just USART2) — see the
  `bind_interrupts!` block.
- embassy-time's integrated timer queue needs `__embassy_time_queue_item_from_waker`, which is
  only provided by **embassy-executor ≥ 0.9** (it owns the split-out `embassy-executor-timer-queue`
  crate). 0.7 predates the split and fails to link. 0.9 also dropped the `task-arena-size-*`
  feature (tasks are statically sized now).
