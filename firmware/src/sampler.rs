//! Acquisition task — drains the ADXL345 FIFO on each watermark interrupt and hands
//! completed [`WINDOW_LEN`]-sample windows to the inference loop.
//!
//! The ADXL345 buffers samples in hardware and raises INT1 every [`WATERMARK`] samples
//! (see [`adxl345`](crate::adxl345)); this task sleeps on the corresponding STM32 EXTI line
//! ([`ExtiInput`]) and only wakes to copy the buffered burst out over SPI. Windows are
//! double-buffered through two channels: it claims an empty buffer from
//! [`FREE_Q`](crate::FREE_Q), fills it sample by sample, and once full sends it to
//! [`FULL_Q`](crate::FULL_Q) for the consumer — then immediately claims the next free
//! buffer, so acquisition continues while the CPU runs the CNN on the just-completed
//! window. No raw pointers and no shared mutable state: buffer ownership moves through the
//! channels, and the executor provides the synchronization.
//!
//! If the consumer falls behind, both buffers end up in flight and the next
//! `FREE_Q.receive()` blocks — back-pressure rather than a data race. (A true overrun would
//! then surface as dropped samples inside the ADXL345's own FIFO, observable via
//! FIFO_STATUS; surfacing that is left for a later milestone.)

use embassy_stm32::exti::ExtiInput;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::mode::Async;
use embassy_stm32::peripherals;
use embassy_stm32::spi::{Config as SpiConfig, Spi, MODE_3};
use embassy_stm32::time::Hertz;
use embassy_stm32::Peri;
use embassy_time::{Duration, Timer};

use pmcore::features::WINDOW_LEN;

use crate::adxl345::Adxl345;
use crate::{Irqs, FREE_Q, FULL_Q};

/// The ADXL345 needs a brief settle between consecutive FIFO reads (datasheet: ≥5 µs); a
/// full watermark burst is still drained in well under one sample period.
const FIFO_READ_GAP: Duration = Duration::from_micros(5);

/// The SPI2 peripheral + pins + DMA the ADXL345 hangs off, bundled so the task takes one
/// handle instead of seven positional arguments.
pub struct SpiPins {
    /// SPI2 peripheral.
    pub spi: Peri<'static, peripherals::SPI2>,
    /// SCK — PB13.
    pub sck: Peri<'static, peripherals::PB13>,
    /// MOSI — PB15.
    pub mosi: Peri<'static, peripherals::PB15>,
    /// MISO — PB14.
    pub miso: Peri<'static, peripherals::PB14>,
    /// Chip select — PB12.
    pub cs: Peri<'static, peripherals::PB12>,
    /// SPI2 TX DMA — DMA1 stream 4.
    pub txdma: Peri<'static, peripherals::DMA1_CH4>,
    /// SPI2 RX DMA — DMA1 stream 3.
    pub rxdma: Peri<'static, peripherals::DMA1_CH3>,
}

/// Owns SPI2 + the ADXL345 and feeds [`crate::FULL_Q`] one completed window at a time.
#[embassy_executor::task]
pub async fn sampler_task(pins: SpiPins, mut int1: ExtiInput<'static, Async>) {
    let mut cfg = SpiConfig::default();
    cfg.frequency = Hertz(4_000_000);
    cfg.mode = MODE_3;

    let spi = Spi::new(
        pins.spi, pins.sck, pins.mosi, pins.miso, pins.txdma, pins.rxdma, Irqs, cfg,
    );
    let cs = Output::new(pins.cs, Level::High, Speed::VeryHigh);

    let mut adxl = Adxl345::new(spi, cs);
    adxl.init_fifo().await;

    let mut buf = FREE_Q.receive().await;
    let mut fill = 0usize;

    loop {
        // Sleep until the FIFO reaches the watermark (INT1 is active-high). Level-wait, not
        // edge: if the line is still asserted after a drain, re-enter immediately.
        int1.wait_for_high().await;

        let pending = adxl.fifo_count().await;
        for _ in 0..pending {
            buf[fill] = adxl.read_fifo_sample().await;
            fill += 1;
            if fill == WINDOW_LEN {
                FULL_Q.send(buf).await;
                buf = FREE_Q.receive().await; // blocks (back-pressure) if the consumer is behind
                fill = 0;
            }
            Timer::after(FIFO_READ_GAP).await;
        }
    }
}
