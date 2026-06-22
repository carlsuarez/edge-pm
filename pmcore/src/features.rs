//! Stage 2 — feature extraction over one acquisition window.
//!
//! Each window is [`WINDOW_LEN`] samples of a 3-axis ([`N_AXES`]) accelerometer. Two
//! kinds of per-window feature are computed per axis and concatenated into a flat
//! [`FEATURE_LEN`]-long `f32` vector that feeds [`crate::model`]:
//!
//! **Time-domain statistics** (3 per axis):
//! * **RMS energy** — `√(mean(xᵢ²))`, overall vibration level.
//! * **Crest factor** — `peak / RMS`, how impulsive the signal is (early fault marker).
//! * **Kurtosis** — 4th standardized moment, spikiness of the distribution.
//!
//! **Spectral bands** ([`FFT_BANDS_PER_AXIS`] per axis): a Hann-windowed real FFT
//! ([`engine::dsp`]) of the axis, with magnitude summed into log-spaced frequency bands and
//! `log1p`-compressed. Bearing faults excite characteristic frequencies (ball-pass tones)
//! that the time-domain stats cannot localize; the bands give the classifier that spectral
//! evidence while staying RPM-agnostic (fixed Hz bands, valid for the near-fixed-speed
//! target). `log1p` keeps the band magnitudes in the same numeric regime as the stats so the
//! downstream int8 path's single shared feature scale stays well-conditioned.
//!
//! [`extract`] lays the vector out as two blocks: first the time-domain stats grouped by
//! axis (`[rms, crest, kurtosis]` for axis 0, then 1, then 2), then the spectral bands
//! grouped by axis. The split keeps the original 9 stat values at the same indices.
//!
//! The definitions here mirror `tools/verify_features.py` exactly, so the firmware's
//! `extract` and that numpy reference agree to `f32` tolerance on identical input — the
//! Milestone B correctness gate.

use engine::dsp;

/// Samples per acquisition window (~160 ms at the ADXL345's 3200 Hz max ODR — the rate the
/// training data is decimated to, see `tools/train_adxl355.py`).
pub const WINDOW_LEN: usize = 512;

// The spectral features transform a whole window at once, so the window length must be the
// FFT's fixed input length.
const _: () = assert!(WINDOW_LEN == dsp::FFT_LEN);

/// Accelerometer axes (X, Y, Z).
pub const N_AXES: usize = 3;

/// Time-domain scalar features computed per axis: RMS, crest factor, kurtosis.
pub const FEATURES_PER_AXIS: usize = 3;

/// FFT magnitude bands computed per axis (log-spaced over the AC spectrum).
pub const FFT_BANDS_PER_AXIS: usize = 5;

/// Log-spaced FFT band edges, as bin indices into the [`dsp::FFT_BINS`]-long magnitude
/// spectrum. Band `b` spans bins `BAND_EDGES[b]..BAND_EDGES[b + 1]`; bin 0 (DC) is skipped.
/// Derived offline as a geometric series from bin 1 to [`dsp::FFT_BINS`]
/// (`round(1 · (256/1)^(b/5))`) and hard-coded so the Rust and the numpy reference in
/// `tools/verify_features.py` bin identically. At 3200 Hz (6.25 Hz/bin) the bands cover
/// ≈6–19, 19–56, 56–175, 175–525, 525–1600 Hz — the middle bands straddle the shaft and
/// ball-pass fault frequencies.
const BAND_EDGES: [usize; FFT_BANDS_PER_AXIS + 1] = [1, 3, 9, 28, 84, dsp::FFT_BINS];

/// Length of the flat feature vector handed to the model: per-axis time-domain stats, then
/// per-axis spectral bands.
pub const FEATURE_LEN: usize = N_AXES * (FEATURES_PER_AXIS + FFT_BANDS_PER_AXIS);

/// One acquisition sample: a reading on each axis, as delivered by the ADXL345.
pub type Sample = [i16; N_AXES];

/// A zeroed sample, used to initialize window buffers before acquisition fills them.
pub const DEFAULT_SAMPLE: Sample = [0; N_AXES];

/// Extract the per-window feature vector from a raw sample window.
///
/// Fills `out` as two blocks: the per-axis time-domain stats
/// (`out[0..3]` = axis 0's `[rms, crest, kurtosis]`, `out[3..6]` = axis 1, …), then the
/// per-axis spectral bands (`out[9..14]` = axis 0's [`FFT_BANDS_PER_AXIS`] band energies,
/// `out[14..19]` = axis 1, …). Allocation-free: each axis is de-interleaved into one
/// `WINDOW_LEN`-long stack buffer, reused across axes, plus a Hann/FFT scratch in
/// [`fft_bands`].
pub fn extract(window: &[Sample; WINDOW_LEN], out: &mut [f32; FEATURE_LEN]) {
    let (stats, bands) = out.split_at_mut(N_AXES * FEATURES_PER_AXIS);
    for axis in 0..N_AXES {
        let col: [i16; WINDOW_LEN] = window.map(|row| row[axis]);

        let stat = &mut stats[axis * FEATURES_PER_AXIS..][..FEATURES_PER_AXIS];
        stat[0] = rms(&col);
        stat[1] = crest(&col);
        stat[2] = kurtosis(&col);

        let band = &mut bands[axis * FFT_BANDS_PER_AXIS..][..FFT_BANDS_PER_AXIS];
        fft_bands(&col, band);
    }
}

/// Per-axis spectral band features: Hann-windowed real FFT magnitude summed into the
/// [`BAND_EDGES`] log-spaced bands, each `log1p`-compressed.
///
/// `out` must be [`FFT_BANDS_PER_AXIS`] long. Allocation-free: a `WINDOW_LEN`-long `f32`
/// scratch (the windowed signal, transformed in place) and a [`dsp::FFT_BINS`]-long
/// magnitude buffer live on the stack.
fn fft_bands(col: &[i16; WINDOW_LEN], out: &mut [f32]) {
    // Apply a periodic Hann window (denominator N, matching the numpy reference) to cut
    // spectral leakage, into the FFT's in-place scratch buffer.
    let mut sig = [0.0f32; WINDOW_LEN];
    for (n, (s, &c)) in sig.iter_mut().zip(col.iter()).enumerate() {
        let phase = 2.0 * core::f32::consts::PI * n as f32 / WINDOW_LEN as f32;
        *s = c as f32 * (0.5 - 0.5 * libm::cosf(phase));
    }

    let mut mag = [0.0f32; dsp::FFT_BINS];
    dsp::rfft512_mag(&mut sig, &mut mag);

    for (b, slot) in out.iter_mut().enumerate() {
        let sum: f32 = mag[BAND_EDGES[b]..BAND_EDGES[b + 1]].iter().sum();
        *slot = libm::log1pf(sum);
    }
}

/// Root-mean-square amplitude: `√(mean(xᵢ²))` — the overall vibration energy in a window.
///
/// The sum of squares is accumulated exactly in `i64` (a window of `i16` samples cannot
/// overflow it), so only the final `√` rounds and RMS stays exact to `f32` at any
/// amplitude.
///
/// # Panics
/// In debug builds, if `x` is empty.
pub fn rms(x: &[i16]) -> f32 {
    debug_assert!(!x.is_empty());
    let ss: i64 = x
        .iter()
        .map(|&v| {
            let v = v as i64;
            v * v
        })
        .sum();
    libm::sqrtf(ss as f32 / x.len() as f32)
}

/// Crest factor: `peak / RMS`, the peak being the largest absolute sample.
///
/// Rises as the signal turns impulsive — sharp spikes over a low background — which is an
/// early bearing-fault marker. Returns 0 for an all-zero (silent) window, where RMS is 0
/// and the ratio is otherwise undefined.
///
/// # Panics
/// In debug builds, if `x` is empty.
pub fn crest(x: &[i16]) -> f32 {
    debug_assert!(!x.is_empty());
    let r = rms(x);
    if r == 0.0 {
        return 0.0;
    }
    let peak = x.iter().map(|&v| v.unsigned_abs()).max().unwrap() as f32;
    peak / r
}

/// Pearson kurtosis: `m₄ / m₂²` — the 4th central moment over the squared variance.
///
/// A scale-invariant measure of how heavy-tailed / spiky the distribution is: ≈3 for
/// Gaussian vibration, climbing well above 3 as impulsive fault transients appear. Uses
/// the biased (population, ÷N) moments and a two-pass computation (mean first, then the
/// central moments) for numerical stability. Returns 0 for a constant window, where the
/// variance is 0 and kurtosis is undefined.
///
/// # Panics
/// In debug builds, if `x` is empty.
pub fn kurtosis(x: &[i16]) -> f32 {
    debug_assert!(!x.is_empty());
    let n = x.len() as f32;

    let mean = x.iter().map(|&v| v as f32).sum::<f32>() / n;

    let (m2, m4) = x.iter().fold((0f32, 0f32), |(m2, m4), &v| {
        let d = v as f32 - mean;
        let d2 = d * d;
        (m2 + d2, m4 + d2 * d2)
    });

    let variance = m2 / n;
    if variance == 0.0 {
        return 0.0;
    }
    (m4 / n) / (variance * variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Approximate float comparison for feature outputs.
    fn close(a: f32, b: f32) -> bool {
        libm::fabsf(a - b) < 1e-4
    }

    #[test]
    fn rms_of_constant_is_its_magnitude() {
        assert!(close(rms(&[7, 7, 7, 7]), 7.0));
        assert!(close(rms(&[-7, -7, -7, -7]), 7.0));
    }

    #[test]
    fn rms_matches_hand_value() {
        // √((9 + 16) / 2) = √12.5
        assert!(close(rms(&[3, 4]), libm::sqrtf(12.5)));
    }

    #[test]
    fn crest_of_alternating_unit_is_one() {
        assert!(close(crest(&[1, -1, 1, -1]), 1.0));
    }

    #[test]
    fn crest_rises_for_an_impulsive_signal() {
        // peak 4, rms √(16/4) = 2 → crest 2
        assert!(close(crest(&[0, 0, 4, 0]), 2.0));
    }

    #[test]
    fn crest_of_a_silent_window_is_zero() {
        assert_eq!(crest(&[0, 0, 0, 0]), 0.0);
    }

    #[test]
    fn kurtosis_of_two_point_signal_is_one() {
        // A ±1 split is the minimum-kurtosis distribution, exactly 1.
        assert!(close(kurtosis(&[-1, 1, -1, 1]), 1.0));
    }

    #[test]
    fn kurtosis_matches_hand_value() {
        // x = [0,0,0,4]: mean 1, m2 = 3, m4 = 21 → 21/9
        assert!(close(kurtosis(&[0, 0, 0, 4]), 21.0 / 9.0));
    }

    #[test]
    fn kurtosis_of_constant_is_zero() {
        assert_eq!(kurtosis(&[5, 5, 5, 5]), 0.0);
    }

    #[test]
    fn extract_lays_out_per_axis_features() {
        // axis 0: alternating ±1          → rms 1, crest 1, kurt 1
        // axis 1: constant 5              → rms 5, crest 1, kurt 0
        // axis 2: repeating [0,0,0,4]     → rms 2, crest 2, kurt 21/9
        let window: [Sample; WINDOW_LEN] = core::array::from_fn(|t| {
            [
                if t % 2 == 0 { -1 } else { 1 },
                5,
                if t % 4 == 3 { 4 } else { 0 },
            ]
        });
        let mut out = [0.0f32; FEATURE_LEN];
        extract(&window, &mut out);
        assert!(close(out[0], 1.0) && close(out[1], 1.0) && close(out[2], 1.0));
        assert!(close(out[3], 5.0) && close(out[4], 1.0) && close(out[5], 0.0));
        assert!(close(out[6], 2.0) && close(out[7], 2.0) && close(out[8], 21.0 / 9.0));
    }

    // First feature index of axis `a`'s spectral band block.
    fn band_base(a: usize) -> usize {
        N_AXES * FEATURES_PER_AXIS + a * FFT_BANDS_PER_AXIS
    }

    #[test]
    fn fft_bands_locate_a_tone_in_its_band() {
        // A tone at bin 40 (≈125 Hz) falls in band 3 (bins 28..84). Put it on axis 0 only.
        let window: [Sample; WINDOW_LEN] = core::array::from_fn(|n| {
            let phase = 2.0 * core::f32::consts::PI * 40.0 * n as f32 / WINDOW_LEN as f32;
            [(1000.0 * libm::cosf(phase)) as i16, 0, 0]
        });
        let mut out = [0.0f32; FEATURE_LEN];
        extract(&window, &mut out);

        let axis0 = &out[band_base(0)..band_base(0) + FFT_BANDS_PER_AXIS];
        let argmax = (0..FFT_BANDS_PER_AXIS)
            .max_by(|&a, &b| axis0[a].total_cmp(&axis0[b]))
            .unwrap();
        assert_eq!(argmax, 3, "tone should dominate band 3, got bands {axis0:?}");

        // Silent axes 1 and 2 have ~zero band energy.
        for a in 1..N_AXES {
            for &v in &out[band_base(a)..band_base(a) + FFT_BANDS_PER_AXIS] {
                assert!(v < 1e-3, "silent axis {a} band non-zero: {v}");
            }
        }
    }

    #[test]
    fn fft_bands_of_silent_window_are_zero() {
        let window = [DEFAULT_SAMPLE; WINDOW_LEN];
        let mut out = [0.0f32; FEATURE_LEN];
        extract(&window, &mut out);
        for &v in &out[N_AXES * FEATURES_PER_AXIS..] {
            assert_eq!(v, 0.0);
        }
    }
}
