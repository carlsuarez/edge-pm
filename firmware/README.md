# firmware — STM32F411 Cortex-M4F shell (Milestone E)

The hardware application: `embassy-stm32` async executor, ADXL345 acquisition over SPI1 +
DMA into a static ring buffer, and the real-time loop that drives [`pmcore`](../pmcore)'s
`features → model → alert` pipeline, the alert LED (PA5), and the UART2 debug log.

**Status: placeholder.** This crate is *excluded from the workspace* (own compile target +
linker script) and is fleshed out in Milestone E, after the portable logic in `pmcore` is
proven on the host. It will first be validated in **Renode** (which models STM32F4 + SPI +
DMA + a file-fed sensor) before running on the board.

## Build (Milestone E)

```sh
rustup target add thumbv7em-none-eabihf      # Cortex-M4F hard-float
cd firmware && cargo build                    # uses .cargo/config.toml + memory.x
```

## Files

- `src/main.rs`   — embassy executor, peripheral init, real-time loop
- `src/adxl345.rs` — SPI driver + ADXL345 register configuration (added in M-E)
- `memory.x`      — STM32F411RE linker layout (512K flash / 128K SRAM)
- `build.rs`      — puts `memory.x` on the linker search path
- `.cargo/config.toml` — target, linker args, flash/run runner
