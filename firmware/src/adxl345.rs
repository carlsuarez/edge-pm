//! ADXL345 driver — SPI 4-wire, FIFO stream mode with a watermark interrupt.
//!
//! The accelerometer runs at its 1600 Hz output data rate and buffers samples in its
//! 32-deep hardware FIFO. Once the FIFO reaches [`WATERMARK`] entries it asserts INT1
//! (active-high), which is wired to an STM32 EXTI line; the [`sampler`](crate::sampler)
//! task sleeps on that interrupt and only wakes to drain the buffered burst — reading
//! [`fifo_count`](Adxl345::fifo_count) samples with [`read_fifo_sample`](Adxl345::read_fifo_sample).
//! This replaces per-sample polling: the CPU is idle between watermark interrupts instead
//! of spinning at the 1600 Hz sample rate.

use embassy_stm32::gpio::Output;
use embassy_stm32::mode::Async;
use embassy_stm32::spi::mode::Master;
use embassy_stm32::spi::Spi;

// Register map (only the registers we touch).
const REG_BW_RATE: u8 = 0x2C;
const REG_POWER_CTL: u8 = 0x2D;
const REG_INT_ENABLE: u8 = 0x2E;
const REG_INT_MAP: u8 = 0x2F;
const REG_DATA_FORMAT: u8 = 0x31;
const REG_DATAX0: u8 = 0x32;
const REG_FIFO_CTL: u8 = 0x38;
const REG_FIFO_STATUS: u8 = 0x39;

// SPI command bits: bit7 = read, bit6 = multi-byte (address auto-increment).
const FLAG_READ: u8 = 0x80;
const FLAG_MULTI: u8 = 0x40;

/// WATERMARK bit position in INT_ENABLE / INT_MAP / INT_SOURCE.
const INT_WATERMARK: u8 = 0x02;

/// FIFO watermark: assert INT1 once this many samples are buffered. The FIFO is 32 deep, so
/// 16 leaves headroom for the drain latency before it would overflow (one entry = one
/// 3-axis sample).
pub const WATERMARK: u8 = 16;

/// Bytes per FIFO entry: x, y, z each a little-endian `i16`.
const SAMPLE_BYTES: usize = 6;

/// ADXL345 over SPI2, owning its chip-select line.
pub struct Adxl345<'d> {
    spi: Spi<'d, Async, Master>,
    cs: Output<'d>,
}

impl<'d> Adxl345<'d> {
    /// Wrap an already-configured SPI bus and CS output.
    pub fn new(spi: Spi<'d, Async, Master>, cs: Output<'d>) -> Self {
        Self { spi, cs }
    }

    /// Configure 1600 Hz, ±16 g full-resolution acquisition into the FIFO in stream mode,
    /// with the watermark interrupt routed (active-high) to INT1. Measurement is enabled
    /// last, so no samples are produced until the FIFO + interrupt path is fully set up.
    pub async fn init_fifo(&mut self) {
        self.write(REG_DATA_FORMAT, 0x0B).await; // full-res, ±16 g, INT active-high
        self.write(REG_BW_RATE, 0x0F).await; // 1600 Hz ODR
        self.write(REG_INT_MAP, 0x00).await; // all interrupts -> INT1 (watermark bit clear)
        self.write(REG_FIFO_CTL, 0x80 | (WATERMARK & 0x1F)).await; // stream mode + watermark
        self.write(REG_INT_ENABLE, INT_WATERMARK).await; // enable the watermark interrupt
        self.write(REG_POWER_CTL, 0x08).await; // measurement mode -> acquisition starts
    }

    /// Number of samples currently in the FIFO (`0..=32`), from FIFO_STATUS bits `[5:0]`.
    pub async fn fifo_count(&mut self) -> u8 {
        let mut buf = [REG_FIFO_STATUS | FLAG_READ, 0];
        self.cs.set_low();
        let _ = self.spi.transfer_in_place(&mut buf).await;
        self.cs.set_high();
        buf[1] & 0x3F
    }

    /// Pop one 3-axis sample from the FIFO — a multi-byte read of DATAX0..DATAZ1, which the
    /// device answers from the head of the FIFO and then advances.
    pub async fn read_fifo_sample(&mut self) -> [i16; 3] {
        let mut buf = [0u8; 1 + SAMPLE_BYTES];
        buf[0] = REG_DATAX0 | FLAG_READ | FLAG_MULTI;
        self.cs.set_low();
        let _ = self.spi.transfer_in_place(&mut buf).await;
        self.cs.set_high();
        [
            i16::from_le_bytes([buf[1], buf[2]]),
            i16::from_le_bytes([buf[3], buf[4]]),
            i16::from_le_bytes([buf[5], buf[6]]),
        ]
    }

    /// Write one configuration register (single-byte, write bit clear).
    async fn write(&mut self, reg: u8, val: u8) {
        let tx = [reg & 0x3F, val];
        self.cs.set_low();
        let _ = self.spi.write(&tx).await;
        self.cs.set_high();
    }
}
