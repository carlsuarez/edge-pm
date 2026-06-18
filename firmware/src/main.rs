//! edge-pm firmware — the STM32F411 hardware shell around [`pmcore`].
//!
//! Boots the `embassy-stm32` async executor and runs the same `features → model → alert`
//! pipeline the host simulator proved out. Each completed 512-sample window is classified by
//! the on-device 1-D CNN; the alert state machine drives the on-board LED (LD2 / PA5) and a
//! one-line log over the ST-Link virtual COM port (USART2).
//!
//! Two sample sources, selected at build time:
//! - **hardware (default):** a real ADXL345 in FIFO stream mode. Its watermark interrupt
//!   (INT1) wakes the [`sampler`] task, which drains the FIFO over SPI2 + DMA and passes
//!   completed windows to the inference loop through [`FULL_Q`]/[`FREE_Q`].
//! - **`--features sim`:** a flash-baked `bearing_stream` blob (see [`sim_source`]) — no
//!   sensor needed, so the whole pipeline + UART log can be validated in Renode / CI.
//!
//! Pin map (Nucleo-F411RE):
//! - LD2 alert LED        — PA5
//! - USART2 debug log     — PA2 (TX) / PA3 (RX), DMA1
//! - ADXL345 over SPI2    — PB13 (SCK) / PB15 (MOSI) / PB14 (MISO), CS PB12, DMA1
//! - ADXL345 INT1         — PB1 (EXTI1), active-high FIFO watermark
//!
//! (SPI2 rather than SPI1 because SPI1's SCK is PA5 — the on-board LED — on this board.)
//!
//! The model is compiled into flash; the forward-pass arena and the run state are carved
//! **once** at boot from a static buffer, so the steady-state loop allocates nothing — the
//! same allocation-free contract `pmcore` enforces.

#![no_std]
#![no_main]

#[cfg(not(feature = "sim"))]
mod adxl345;
#[cfg(not(feature = "sim"))]
mod sampler;
#[cfg(feature = "sim")]
mod sim_source;

use core::fmt::Write as _;

use embassy_executor::Spawner;
use embassy_stm32::gpio::{Level, Output, Speed};
use embassy_stm32::mode::Async;
use embassy_stm32::usart::{Config as UartConfig, Uart};
use embassy_stm32::{bind_interrupts, peripherals};

use panic_halt as _;
use static_cell::StaticCell;

#[cfg(not(feature = "sim"))]
use embassy_stm32::exti::ExtiInput;
#[cfg(not(feature = "sim"))]
use embassy_stm32::gpio::Pull;
#[cfg(not(feature = "sim"))]
use embassy_sync::blocking_mutex::raw::ThreadModeRawMutex;
#[cfg(not(feature = "sim"))]
use embassy_sync::channel::Channel;
#[cfg(feature = "sim")]
use embassy_time::{Duration, Timer};

#[cfg(not(feature = "sim"))]
use crate::sampler::{sampler_task, SpiPins};
use pmcore::alert::{AlertMachine, State};
use pmcore::features::{Sample, DEFAULT_SAMPLE, WINDOW_LEN};
use pmcore::model::ModelConfig;
use pmcore::pipeline::Outcome;

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
    USART2 => embassy_stm32::usart::InterruptHandler<peripherals::USART2>;
    DMA1_STREAM6 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH6>;
    DMA1_STREAM5 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH5>;
    #[cfg(not(feature = "sim"))]
    DMA1_STREAM4 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH4>;
    #[cfg(not(feature = "sim"))]
    DMA1_STREAM3 => embassy_stm32::dma::InterruptHandler<peripherals::DMA1_CH3>;
    #[cfg(not(feature = "sim"))]
    EXTI1 => embassy_stm32::exti::InterruptHandler<embassy_stm32::interrupt::typelevel::EXTI1>;
});

#[repr(align(4))]
struct Aligned<const N: usize>([u8; N]);

// The hardware build classifies live ADXL data; the sim build replays a stream that was
// generated against its own checked-in model, so each picks the matching checkpoint — and
// the `q8` feature selects the int8 (v2) version of whichever model the build uses.
#[cfg(all(not(feature = "sim"), not(feature = "q8")))]
macro_rules! model_file {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bearing_cnn.bin")
    };
}
#[cfg(all(not(feature = "sim"), feature = "q8"))]
macro_rules! model_file {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bearing_cnn_q8.bin")
    };
}
#[cfg(all(feature = "sim", not(feature = "q8")))]
macro_rules! model_file {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/bearing_stream.bin")
    };
}
#[cfg(all(feature = "sim", feature = "q8"))]
macro_rules! model_file {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/bearing_stream_q8.bin"
        )
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

// --- sim: one reusable window buffer (single producer/consumer, in the main task) --------
#[cfg(feature = "sim")]
static SIM_WINDOW: StaticCell<[Sample; WINDOW_LEN]> = StaticCell::new();

// --- hardware: two window buffers that circulate through the free/full channels ----------
#[cfg(not(feature = "sim"))]
static WINDOW_BUF1: StaticCell<[Sample; WINDOW_LEN]> = StaticCell::new();
#[cfg(not(feature = "sim"))]
static WINDOW_BUF2: StaticCell<[Sample; WINDOW_LEN]> = StaticCell::new();
/// Empty window buffers waiting to be filled by the [`sampler`] task.
#[cfg(not(feature = "sim"))]
pub(crate) static FREE_Q: Channel<ThreadModeRawMutex, &'static mut [Sample; WINDOW_LEN], 2> =
    Channel::new();
/// Filled windows handed from the [`sampler`] task to the inference loop.
#[cfg(not(feature = "sim"))]
pub(crate) static FULL_Q: Channel<ThreadModeRawMutex, &'static mut [Sample; WINDOW_LEN], 2> =
    Channel::new();

/// Drive the LED from the alert state and log only state *transitions* over USART2.
///
/// A blocking (not DMA) write: the line is short and emitted only on a change, so the
/// few-µs stall is irrelevant — and it keeps the log off the UART TX DMA, which Renode does
/// not raise a transfer-complete interrupt for (an async `.write()` would hang there).
fn handle_outcome(
    out: &Outcome,
    prev: &mut State,
    led: &mut Output<'static>,
    uart: &mut Uart<'static, Async>,
) {
    match out.state {
        State::Alert { .. } => led.set_high(),
        State::Normal => led.set_low(),
    }
    if out.state != *prev {
        let mut msg: heapless::String<64> = heapless::String::new();
        match out.state {
            State::Alert { class } => {
                let _ = write!(msg, "ALERT {} conf={:.3}\r\n", class.name(), out.confidence);
            }
            State::Normal => {
                let _ = write!(msg, "CLEAR\r\n");
            }
        }
        let _ = uart.blocking_write(msg.as_bytes());
        *prev = out.state;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_stm32::init(Default::default());

    let mut led = Output::new(p.PA5, Level::Low, Speed::Low);

    let mut uart = Uart::new(
        p.USART2,
        p.PA3,
        p.PA2,
        p.DMA1_CH6,
        p.DMA1_CH5,
        Irqs,
        UartConfig::default(),
    )
    .unwrap();
    let _ = uart.blocking_write(b"edge-pm: boot\r\n");

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

    // --- sim build: flash-baked stream, no SPI/EXTI (runs in Renode / CI) ----------------
    #[cfg(feature = "sim")]
    {
        let _ = &spawner; // no tasks spawned in the sim build
        let _ = uart.blocking_write(b"edge-pm: sim source (baked bearing_stream)\r\n");

        let window = SIM_WINDOW.init([DEFAULT_SAMPLE; WINDOW_LEN]);
        let mut src = sim_source::SimSource::new();
        let mut fill = 0usize;
        loop {
            let Some(sample) = src.next_sample() else {
                let _ = uart.blocking_write(b"edge-pm: sim stream complete\r\n");
                break;
            };
            window[fill] = sample;
            fill += 1;
            if fill == WINDOW_LEN {
                #[cfg(not(feature = "q8"))]
                let out = process_window(&cfg, &weights, window, &mut state, &mut alert);
                #[cfg(feature = "q8")]
                let out = process_window_q8(&cfg, &weights, window, &mut state, &mut alert);
                handle_outcome(&out, &mut prev, &mut led, &mut uart);
                fill = 0;
            }
        }

        // The baked stream is finite; park so the executor's entry task never returns.
        loop {
            Timer::after(Duration::from_secs(3600)).await;
        }
    }

    // --- hardware build: ADXL345 FIFO + watermark IRQ ------------------------------------
    #[cfg(not(feature = "sim"))]
    {
        FREE_Q
            .send(WINDOW_BUF1.init([DEFAULT_SAMPLE; WINDOW_LEN]))
            .await;
        FREE_Q
            .send(WINDOW_BUF2.init([DEFAULT_SAMPLE; WINDOW_LEN]))
            .await;

        // ADXL345 INT1 -> PB1 (EXTI1); the watermark interrupt is active-high.
        let int1 = ExtiInput::new(p.PB1, p.EXTI1, Pull::Down, Irqs);
        let pins = SpiPins {
            spi: p.SPI2,
            sck: p.PB13,
            mosi: p.PB15,
            miso: p.PB14,
            cs: p.PB12,
            txdma: p.DMA1_CH4,
            rxdma: p.DMA1_CH3,
        };
        spawner.spawn(sampler_task(pins, int1)).unwrap();
        let _ = uart.blocking_write(b"edge-pm: adxl345 fifo source\r\n");

        loop {
            let window = FULL_Q.receive().await;
            #[cfg(not(feature = "q8"))]
            let out = process_window(&cfg, &weights, window, &mut state, &mut alert);
            #[cfg(feature = "q8")]
            let out = process_window_q8(&cfg, &weights, window, &mut state, &mut alert);
            handle_outcome(&out, &mut prev, &mut led, &mut uart);
            FREE_Q.send(window).await; // return the buffer to circulation
        }
    }
}
