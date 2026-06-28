//! Acquisition task — drains the ADXL345 FIFO on each watermark interrupt and hands
//! completed [`WINDOW_LEN`]-sample windows to the inference loop.
//!
//! The ADXL345 buffers samples in hardware and raises its watermark interrupt every time the
//! FIFO fills to the configured level; this task sleeps on the corresponding STM32 EXTI line
//! ([`ExtiInput`]) via the driver's [`wait_and_drain`](adxl345_async::Adxl345::wait_and_drain)
//! and only wakes to copy the buffered burst out over SPI + DMA. The driver yields each
//! reading as an [`Accel`] (`x`/`y`/`z` raw counts); we transcribe it into pmcore's
//! [`Sample`] (`[i16; 3]`) as we fill the window.
//!
//! Windows are double-buffered through two channels: the task claims an empty buffer from
//! [`FREE_Q`](crate::FREE_Q), fills it, and once full sends it to [`FULL_Q`](crate::FULL_Q)
//! for the consumer — then immediately claims the next free buffer, so acquisition continues
//! while the CPU runs the CNN on the just-completed window. No raw pointers and no shared
//! mutable state: buffer ownership moves through the channels, and the executor provides the
//! synchronization.
//!
//! If the consumer falls behind, both buffers end up in flight and the next
//! `FREE_Q.receive()` blocks — back-pressure rather than a data race. (A true overrun would
//! then surface as dropped samples inside the ADXL345's own FIFO, observable via
//! FIFO_STATUS; that is not yet surfaced.)

use adxl345_async::{Accel, Adxl345, SpiBus};
use embassy_stm32::exti::ExtiInput;
use embassy_stm32::gpio::Output;
use embassy_stm32::mode::Async;
use embassy_stm32::spi::mode::Master;
use embassy_stm32::spi::Spi;
use embassy_time::Delay;
use embedded_hal_bus::spi::ExclusiveDevice;

use pmcore::features::WINDOW_LEN;

use crate::{FREE_Q, FULL_Q};

/// The concrete driver type the task owns: the async ADXL345 over SPI2 wrapped in an
/// `embedded-hal-bus` [`ExclusiveDevice`] (bus + CS + delay). Spelled out because an
/// `#[embassy_executor::task]` cannot be generic.
pub type AccelDev =
    Adxl345<SpiBus<ExclusiveDevice<Spi<'static, Async, Master>, Output<'static>, Delay>>>;

/// Owns the ADXL345 + its watermark interrupt line and feeds [`crate::FULL_Q`] one completed
/// window at a time.
#[embassy_executor::task]
pub async fn sampler_task(mut accel: AccelDev, mut int_pin: ExtiInput<'static, Async>) {
    // The FIFO is 32 entries deep, so one watermark burst never exceeds 32 readings.
    let mut batch = [Accel::default(); 32];
    let mut buf = FREE_Q.receive().await;
    let mut fill = 0usize;

    loop {
        // Sleep until the watermark interrupt asserts, then drain the buffered burst. A bus
        // error yields 0 samples and we simply wait for the next watermark (no hot spin).
        let n = accel
            .wait_and_drain(&mut int_pin, &mut batch)
            .await
            .unwrap_or(0);

        for a in &batch[..n] {
            // Accel (x/y/z raw counts) -> pmcore Sample ([i16; 3]).
            buf[fill] = [a.x, a.y, a.z];
            fill += 1;
            if fill == WINDOW_LEN {
                FULL_Q.send(buf).await;
                buf = FREE_Q.receive().await; // blocks (back-pressure) if the consumer is behind
                fill = 0;
            }
        }
    }
}
