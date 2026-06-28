//! edge-pm firmware — the STM32F411CE "Black Pill" hardware shell around [`pmcore`].
//!
//! Boots the `embassy-stm32` async executor and runs the `features → model → alert`
//! pipeline. Each completed 512-sample window is classified by the on-device 1-D CNN; the
//! alert state machine drives the on-board LED (PC13) and logs state transitions over the
//! **ST-Link debug probe** via `defmt` RTT (no UART — see [`handle_outcome`]).
//!
//! Samples come from a real ADXL345 in FIFO stream mode: its watermark interrupt wakes the
//! [`sampler`] task, which drains the FIFO over SPI2 + DMA and passes completed windows to
//! the inference loop through [`FULL_Q`]/[`FREE_Q`].
//!
//! Pin map (Black Pill STM32F411CEU6):
//! - PC13 alert LED       — active-low on this board (the LED is wired to 3V3)
//! - ADXL345 over SPI2    — PB13 (SCK) / PB15 (MOSI) / PB14 (MISO), CS PB12, DMA1
//! - ADXL345 INT          — PB1 (EXTI1), active-high FIFO watermark
//! - logs                 — RTT over SWD (PA13 SWDIO / PA14 SWCLK), printed by `probe-rs run`
//!
//! (SPI2 rather than SPI1 because SPI1's SCK is PA5.)
//!
//! The model is compiled into flash; the forward-pass arena and the run state are carved
//! **once** at boot from a static buffer, so the steady-state loop allocates nothing — the
//! same allocation-free contract `pmcore` enforces.

#![no_std]
#![no_main]

mod sampler;

use defmt::info;

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::spi::{Config as SpiConfig, Spi, MODE_3};
use embassy_stm32::time::Hertz;
use embassy_stm32::{bind_interrupts, peripherals};
use embedded_hal_bus::spi::ExclusiveDevice;

// defmt-rtt installs the RTT log transport; panic-probe prints panics over the same channel.
use defmt_rtt as _;
use panic_probe as _;
use static_cell::StaticCell;

use embassy_stm32::exti::ExtiInput;
use embassy_stm32::gpio::Pull;
use embassy_sync::blocking_mutex::raw::ThreadModeRawMutex;
use embassy_sync::channel::Channel;

use crate::sampler::sampler_task;
use pmcore::alert::{AlertMachine, State};
use pmcore::features::{Sample, DEFAULT_SAMPLE, WINDOW_LEN};
use pmcore::model::ModelConfig;
use pmcore::pipeline::Outcome;

use adxl345_async::{Adxl345, Config as AccelConfig, DataRate, FifoMode, IntRoute, Range};

// The model representation is fixed at build time by the `q8` feature, so the load path,
// working set, and per-window step are cfg-selected (no runtime dispatch) — this is what lets
// the int8 build carve only its ~4×-smaller integer buffer and skip the fp32 activation arena.
#[cfg(feature = "q8")]
use pmcore::model::QuantizedWeights;
#[cfg(not(feature = "q8"))]
use pmcore::model::Weights;
#[cfg(not(feature = "q8"))]
use pmcore::pipeline::process_window;
#[cfg(feature = "q8")]
use pmcore::pipeline::process_window_q8;
use pmcore::{Arena, RunState};

bind_interrupts!(struct Irqs {
    DMA1_STREAM4 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH4>;
    DMA1_STREAM3 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH3>;
    EXTI1 => embassy_stm32::exti::InterruptHandler<embassy_stm32::interrupt::typelevel::EXTI1>;
});

// defmt log timestamp: microseconds since boot, read from the embassy time driver.
defmt::timestamp!("{=u64:us}", embassy_time::Instant::now().as_micros());

#[repr(align(4))]
struct Aligned<const N: usize>([u8; N]);

// The firmware classifies live ADXL data against the deployed checkpoint; the `q8` feature
// selects the int8 (v2) version of the model.
#[cfg(not(feature = "q8"))]
macro_rules! model_file {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bearing_cnn.bin")
    };
}
#[cfg(feature = "q8")]
macro_rules! model_file {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bearing_cnn_q8.bin")
    };
}

static MODEL: Aligned<{ include_bytes!(model_file!()).len() }> =
    Aligned(*include_bytes!(model_file!()));

const CFG: ModelConfig = ModelConfig {
    c1: 16,
    k1: 7,
    s1: 2,
    c2: 32,
    k2: 5,
    s2: 2,
};

/// fp32 forward-pass arena (fp32 build only): every activation buffer is carved from this
/// once at boot, so the steady-state loop allocates nothing.
#[cfg(not(feature = "q8"))]
static SCRATCH: StaticCell<[f32; CFG.buf_len()]> = StaticCell::new();

/// Int8 working buffer for the integer-only forward pass (q8 build only) — holds the
/// de-interleaved window, both conv int8 outputs, and the int8 dense input
/// ([`ModelConfig::buf_len`] elements, but `i8` not `f32`), ~4× smaller than the fp32 arena in
/// bytes. Carved once at boot.
#[cfg(feature = "q8")]
static QSCRATCH: StaticCell<[i8; CFG.buf_len()]> = StaticCell::new();

// two window buffers that circulate through the free/full channels
static WINDOW_BUF1: StaticCell<[Sample; WINDOW_LEN]> = StaticCell::new();
static WINDOW_BUF2: StaticCell<[Sample; WINDOW_LEN]> = StaticCell::new();
/// Empty window buffers waiting to be filled by the [`sampler`] task.
pub(crate) static FREE_Q: Channel<ThreadModeRawMutex, &'static mut [Sample; WINDOW_LEN], 2> =
    Channel::new();
/// Filled windows handed from the [`sampler`] task to the inference loop.
pub(crate) static FULL_Q: Channel<ThreadModeRawMutex, &'static mut [Sample; WINDOW_LEN], 2> =
    Channel::new();

/// Drive the LED from the alert state and log only state *transitions* over the probe (RTT).
///
/// The Black Pill's on-board PC13 LED is active-low (the pin sinks current from 3V3), so
/// `set_low` lights it. The log line is emitted only on a change, so steady-state acquisition
/// stays silent on the wire.
fn handle_outcome(out: &Outcome, prev: &mut State, led: &mut Output<'static>) {
    match out.state {
        State::Alert { .. } => led.set_low(), // LED on
        State::Normal => led.set_high(),      // LED off
    }
    if out.state != *prev {
        *prev = out.state;
    }

    match out.state {
        State::Alert { class } => info!("ALERT {} conf={}", class.name(), out.confidence),
        State::Normal => info!("NORMAL"),
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_stm32::init(Default::default());

    // Busy-spin (NOT an async Timer) so the probe-rs RTT reader can attach after reset before
    // the first log line — a Timer would idle the core into WFE during the attach window and
    // gate SWD access. ~0.25 s at the 16 MHz HSI default clock.
    cortex_m::asm::delay(4_000_000);
    info!("edge-pm: boot");

    // PC13 LED is active-low; start high (off).
    let mut led = Output::new(p.PC13, Level::High, Speed::Low);
    let cs = Output::new(p.PB12, Level::High, Speed::VeryHigh);

    let mut spi_cfg = SpiConfig::default();
    spi_cfg.frequency = Hertz(4_000_000);
    spi_cfg.mode = MODE_3;
    let spi = Spi::new(
        p.SPI2, p.PB13, p.PB15, p.PB14, p.DMA1_CH4, p.DMA1_CH3, Irqs, spi_cfg,
    );

    let dev = ExclusiveDevice::new(spi, cs, embassy_time::Delay)
        .expect("ExclusiveDevice (CS set-high) failed");

    let config = AccelConfig::new()
        .range(Range::G16)
        .rate(DataRate::Hz3200)
        .full_res(true)
        .fifo(FifoMode::Stream { watermark: 16 })
        .int_route(IntRoute::Int2);

    let accel = Adxl345::new_spi(dev, config).await.unwrap();
    info!("adxl345 ready (DEVID ok), FIFO stream @ 3200 Hz, watermark 16");

    // Load the baked model and carve its working set once, mirroring the host startup. The
    // representation is cfg-fixed: the fp32 build carves the fp32 arena + RunState; the int8
    // build carves only the (much smaller) integer working buffer.
    #[cfg(not(feature = "q8"))]
    let (cfg, weights) = Weights::load(&MODEL.0).unwrap();
    #[cfg(feature = "q8")]
    let (cfg, weights) = QuantizedWeights::load(&MODEL.0).unwrap();
    assert_eq!(cfg, CFG);

    let mut state = {
        #[cfg(not(feature = "q8"))]
        let scratch = SCRATCH.init([0.0; CFG.buf_len()]);
        #[cfg(feature = "q8")]
        let scratch = QSCRATCH.init([0; CFG.buf_len()]);

        let mut arena = Arena::new(scratch);
        RunState::new(&mut arena, &cfg).unwrap()
    };
    let mut alert = AlertMachine::new();
    let mut prev = State::Normal;

    FREE_Q
        .send(WINDOW_BUF1.init([DEFAULT_SAMPLE; WINDOW_LEN]))
        .await;
    FREE_Q
        .send(WINDOW_BUF2.init([DEFAULT_SAMPLE; WINDOW_LEN]))
        .await;

    // ADXL345 watermark INT -> PB1 (EXTI1); active-high.
    let int_pin = ExtiInput::new(p.PB1, p.EXTI1, Pull::Down, Irqs);
    spawner.spawn(sampler_task(accel, int_pin)).unwrap();
    info!("sampler started; classifying live windows");

    loop {
        let window = FULL_Q.receive().await;
        #[cfg(not(feature = "q8"))]
        let out = process_window(&cfg, &weights, window, &mut state, &mut alert);
        #[cfg(feature = "q8")]
        let out = process_window_q8(&cfg, &weights, window, &mut state, &mut alert);
        handle_outcome(&out, &mut prev, &mut led);
        FREE_Q.send(window).await; // return the buffer to circulation
    }
}
