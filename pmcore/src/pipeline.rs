//! Stage 1 glue — windowing and the sample-buffer handoff, platform-agnostic parts.
//!
//! On-device, the ADXL345 streams 3-axis samples into a static ring buffer via DMA, and
//! the CPU wakes on a window-complete interrupt once [`WINDOW_LEN`](crate::features::WINDOW_LEN)
//! samples have accumulated. The DMA/interrupt wiring is firmware-specific (it lives in
//! `firmware/`), but the windowing bookkeeping — tracking fill level, presenting a
//! completed window as a `&[Sample; WINDOW_LEN]`, and double-buffering so acquisition
//! continues during feature extraction — is portable and lives here so it can be exercised
//! on the host with a file-fed sample stream.
//!
//! This module also hosts [`process_window`], the end-to-end per-window step
//! (extract → forward → decide) that both `host-sim` and the firmware share, so the wiring
//! between the four stages exists in exactly one place.

use crate::alert::{top_class, AlertMachine, State};
use crate::features::{self, Sample, FEATURE_LEN, WINDOW_LEN, N_AXES};
use crate::model::{Class, Model, ModelError, N_CLASSES};

/// Double-buffered windower over a 3-axis sample stream.
///
/// [`push`](Self::push) one sample at a time; every [`WINDOW_LEN`] pushes it returns the
/// just-completed window and flips to its other buffer, so the returned window stays valid
/// while acquisition continues filling the alternate buffer (the firmware's DMA fills one
/// half while the CPU processes the other — modelled here so the host path uses the same
/// API). Owns both buffers inline: no allocation, `const`-constructible for a firmware
/// static.
pub struct Windower {
    buffers: [[Sample; WINDOW_LEN]; 2],
    /// Samples written into the active buffer so far (`0..WINDOW_LEN`).
    fill: usize,
    /// Index (`0`/`1`) of the buffer currently being filled.
    active: usize,
}

impl Windower {
    /// An empty windower, both buffers zeroed, filling buffer 0.
    pub const fn new() -> Self {
        Self {
            buffers: [[[0; N_AXES]; WINDOW_LEN]; 2],
            fill: 0,
            active: 0,
        }
    }

    /// Append one sample. Returns the completed window the instant it fills (and then
    /// resumes into the other buffer), otherwise [`None`].
    pub fn push(&mut self, sample: Sample) -> Option<&[Sample; WINDOW_LEN]> {
        self.buffers[self.active][self.fill] = sample;
        self.fill += 1;
        if self.fill == WINDOW_LEN {
            let completed = self.active;
            self.active ^= 1; // flip to the spare buffer for the next window
            self.fill = 0;
            Some(&self.buffers[completed])
        } else {
            None
        }
    }

    /// Samples accumulated in the in-progress window (`0..WINDOW_LEN`).
    pub fn fill_level(&self) -> usize {
        self.fill
    }

    /// Discard the in-progress window (e.g. after a stream gap).
    pub fn reset(&mut self) {
        self.fill = 0;
    }
}

impl Default for Windower {
    fn default() -> Self {
        Self::new()
    }
}

/// The result of running one window through the full pipeline.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    /// The Stage-2 feature vector for the window.
    pub features: [f32; FEATURE_LEN],
    /// The Stage-3 softmax probabilities, one per [`Class`].
    pub probs: [f32; N_CLASSES],
    /// The most-likely class (argmax of `probs`).
    pub class: Class,
    /// The probability of `class`.
    pub confidence: f32,
    /// The alert state after this window was folded into the machine.
    pub state: State,
}

/// Run one completed window through every stage: extract features, run the model forward
/// pass, and advance the alert state machine.
///
/// `scratch` is the model's working buffer (size [`ModelConfig::arena_floats`]); it is the
/// only memory this touches beyond the stack, so the call is allocation-free and identical
/// on host and device.
///
/// [`ModelConfig::arena_floats`]: crate::model::ModelConfig::arena_floats
pub fn process_window(
    model: &Model,
    window: &[Sample; WINDOW_LEN],
    scratch: &mut [f32],
    alert: &mut AlertMachine,
) -> Result<Outcome, ModelError> {
    let mut feats = [0.0f32; FEATURE_LEN];
    features::extract(window, &mut feats);

    let mut probs = [0.0f32; N_CLASSES];
    model.forward(window, &feats, scratch, &mut probs)?;

    let state = alert.update(&probs);
    let (class, confidence) = top_class(&probs);

    Ok(Outcome {
        features: feats,
        probs,
        class,
        confidence,
        state,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(i: usize) -> Sample {
        let v = i as i16;
        [v, v.wrapping_mul(2), v.wrapping_mul(3)]
    }

    #[test]
    fn windower_emits_one_window_every_window_len_samples() {
        let mut w = Windower::new();
        for i in 0..WINDOW_LEN - 1 {
            assert!(w.push(sample(i)).is_none());
        }
        assert_eq!(w.fill_level(), WINDOW_LEN - 1);

        let win = w.push(sample(WINDOW_LEN - 1)).expect("window completes");
        // The completed window holds the samples in order.
        assert_eq!(win[0], sample(0));
        assert_eq!(win[WINDOW_LEN - 1], sample(WINDOW_LEN - 1));
        // Fill counter wrapped for the next window.
        assert_eq!(w.fill_level(), 0);
    }

    #[test]
    fn windower_double_buffers_so_back_to_back_windows_dont_alias() {
        let mut w = Windower::new();
        for i in 0..WINDOW_LEN {
            w.push(sample(i));
        }
        // Second window into the other buffer; capture its pointer and contents.
        let mut second_ptr = core::ptr::null();
        for i in 0..WINDOW_LEN {
            if let Some(win) = w.push(sample(1000 + i)) {
                second_ptr = win.as_ptr();
                assert_eq!(win[0], sample(1000));
            }
        }
        // Re-fill a third window: it reuses buffer 0 (the first window's storage), proving
        // the windower alternates exactly two buffers rather than growing.
        let mut third_ptr = core::ptr::null();
        for i in 0..WINDOW_LEN {
            if let Some(win) = w.push(sample(2000 + i)) {
                third_ptr = win.as_ptr();
            }
        }
        assert!(!second_ptr.is_null() && !third_ptr.is_null());
        assert_ne!(second_ptr, third_ptr); // consecutive windows live in distinct buffers
    }

    #[test]
    fn reset_discards_the_in_progress_window() {
        let mut w = Windower::new();
        for i in 0..10 {
            w.push(sample(i));
        }
        assert_eq!(w.fill_level(), 10);
        w.reset();
        assert_eq!(w.fill_level(), 0);
    }
}
