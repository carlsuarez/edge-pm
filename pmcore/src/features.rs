//! Stage 2 — feature extraction over one acquisition window.
//!
//! Each window is [`WINDOW_LEN`] samples of a 3-axis ([`N_AXES`]) accelerometer. The
//! per-window scalar features are computed per axis and concatenated into a flat
//! [`FEATURE_LEN`]-long `f32` vector that feeds [`crate::model`]:
//!
//! * **RMS energy** — `√(mean(xᵢ²))`, overall vibration level.
//! * **Crest factor** — `peak / RMS`, how impulsive the signal is (early fault marker).
//! * **Kurtosis** — 4th standardized moment, spikiness of the distribution.
//!
//! [`extract`] writes them grouped by axis — `[rms, crest, kurtosis]` for axis 0, then
//! axis 1, then axis 2. An optional 256-point FFT magnitude spectrum is a later addition.
//!
//! The definitions here mirror `tools/verify_features.py` exactly, so the firmware's
//! `extract` and that numpy reference agree to `f32` tolerance on identical input — the
//! Milestone B correctness gate.

/// Samples per acquisition window (~320 ms at 1600 Hz).
pub const WINDOW_LEN: usize = 512;

/// Accelerometer axes (X, Y, Z).
pub const N_AXES: usize = 3;

/// Scalar features computed per axis: RMS, crest factor, kurtosis.
pub const FEATURES_PER_AXIS: usize = 3;

/// Length of the flat feature vector handed to the model: per-axis features, all axes.
pub const FEATURE_LEN: usize = N_AXES * FEATURES_PER_AXIS;

/// One acquisition sample: a reading on each axis, as delivered by the ADXL345.
pub type Sample = [i16; N_AXES];

/// Extract the per-window feature vector from a raw sample window.
///
/// Fills `out` with the per-axis RMS, crest factor, and kurtosis of `window`, grouped by
/// axis (`out[0..3]` = axis 0's `[rms, crest, kurtosis]`, `out[3..6]` = axis 1, …).
/// Allocation-free: each axis is de-interleaved into one `WINDOW_LEN`-long stack buffer,
/// reused across axes.
pub fn extract(window: &[Sample; WINDOW_LEN], out: &mut [f32; FEATURE_LEN]) {
    for (i, chunk) in out.chunks_exact_mut(FEATURES_PER_AXIS).enumerate() {
        let col: [i16; WINDOW_LEN] = core::array::from_fn(|j| window[j][i]);
        chunk[0] = rms(&col);
        chunk[1] = crest(&col);
        chunk[2] = kurtosis(&col);
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
}
