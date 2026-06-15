//! edge-pm firmware entry point — Milestone E (skeleton, not yet built).
//!
//! The hardware shell around [`pmcore`]: bring up SPI1 + DMA to the ADXL345, fill a
//! static ring buffer, wake on a window-complete interrupt, then run the portable
//! `features -> model -> alert` pipeline and drive the LED (PA5) + UART2 log. Built on
//! `embassy-stm32`'s async executor.
//!
//! This file is a placeholder so the layout is visible; the real implementation (and the
//! embassy/cortex-m dependencies it needs) arrives in Milestone E, validated first in
//! Renode. It is intentionally not part of the host workspace build.
//!
//! ```ignore
//! #![no_std]
//! #![no_main]
//!
//! // mod adxl345;  // SPI driver + register config
//!
//! use embassy_executor::Spawner;
//!
//! #[embassy_executor::main]
//! async fn main(_spawner: Spawner) {
//!     // 1. init clocks/peripherals  2. configure ADXL345 (1600 Hz, full-res)
//!     // 3. start SPI+DMA into the ring buffer
//!     // 4. per window: pmcore::features::extract -> model -> alert -> LED/UART
//! }
//! ```
