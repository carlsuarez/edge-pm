//! Flash-baked accelerometer source for the `sim` build (Renode / CI bring-up).
//!
//! Under `--features sim` the firmware has no ADXL345; it reads samples from this baked
//! stream instead, so the whole `windowing → features → CNN → alert` pipeline and the UART
//! log can be exercised in emulation with no SPI or sensor model. The data is the exact same
//! `bearing_stream` the host gate (`host-sim replay`) and the PyTorch reference consume, so a
//! sim run reproduces their alert trajectory — just on the real Cortex-M4F instruction set.
//!
//! Regenerate the blob with `python tools/export_sim_stream.py` (little-endian `i16`, x,y,z
//! interleaved — the same layout [`Sample`] uses).

use pmcore::features::Sample;

/// The baked stream: raw little-endian `i16`, x,y,z interleaved (6 bytes per sample).
static STREAM: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/bearing_stream.samples.bin"
));

/// Sequential reader over [`STREAM`]; hands back one 3-axis sample per call until exhausted.
pub struct SimSource {
    pos: usize,
}

impl SimSource {
    /// Start at the beginning of the baked stream.
    pub fn new() -> Self {
        Self { pos: 0 }
    }

    /// The next `[i16; 3]` sample, or `None` once the stream is exhausted.
    pub fn next_sample(&mut self) -> Option<Sample> {
        let base = self.pos * 6;
        if base + 6 > STREAM.len() {
            return None;
        }
        self.pos += 1;
        let rd = |off: usize| i16::from_le_bytes([STREAM[base + off], STREAM[base + off + 1]]);
        Some([rd(0), rd(2), rd(4)])
    }
}
