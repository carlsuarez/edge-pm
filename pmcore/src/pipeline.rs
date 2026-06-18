//! Stage 1 glue — the per-window pipeline step shared by `host-sim` and the firmware.
//!
//! How a completed window arrives is platform-specific and lives in the caller:
//! - the firmware's `sampler` task drains the ADXL345 FIFO on each watermark interrupt and
//!   hands full windows across an `embassy_sync` channel,
//! - `host-sim` chops a recorded sample stream into
//!   [`WINDOW_LEN`](crate::features::WINDOW_LEN)-long slices.
//!
//! Once a window exists, both feed it through [`process_window`] — the end-to-end
//! `extract → forward → decide` step — so the wiring between the four stages exists in
//! exactly one place.

use crate::alert::{top_class, AlertMachine, State};
use crate::features::{self, Sample, FEATURE_LEN, WINDOW_LEN};
use crate::model::{forward, forward_q8, Class, ModelConfig, QuantizedWeights, Weights, N_CLASSES};
use crate::RunState;

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
/// This is the **steady-state loop body** — on-device it is called once per window for the
/// life of the program. It allocates nothing and reuses the caller-owned working set: the
/// [`Weights`] and the [`RunState`] (whose arena is carved exactly once at startup by
/// [`RunState::new`], *not* here) are borrowed, never built. The probabilities are written to
/// a caller-owned buffer each call. Identical on host and device.
pub fn process_window(
    config: &ModelConfig,
    weights: &Weights,
    window: &[Sample; WINDOW_LEN],
    state: &mut RunState,
    alert: &mut AlertMachine,
) -> Outcome {
    let mut feats = [0.0f32; FEATURE_LEN];
    features::extract(window, &mut feats);

    let mut probs = [0.0f32; N_CLASSES];
    forward(config, weights, state, window, &feats, &mut probs);
    decide(feats, probs, alert)
}

/// The **integer-only** int8 twin of [`process_window`]: identical pipeline driven by
/// [`forward_q8`] over [`QuantizedWeights`].
///
/// Differs only in that the model runs in integer arithmetic, so it takes a
/// [`RunState<i8>`](RunState) (the int8 working set, carved once at startup) instead of the fp32
/// [`RunState`] activation arena. Feature extraction and the alert decision are shared with
/// the fp32 path.
pub fn process_window_q8(
    config: &ModelConfig,
    weights: &QuantizedWeights,
    window: &[Sample; WINDOW_LEN],
    state: &mut RunState<i8>,
    alert: &mut AlertMachine,
) -> Outcome {
    let mut feats = [0.0f32; FEATURE_LEN];
    features::extract(window, &mut feats);

    let mut probs = [0.0f32; N_CLASSES];
    forward_q8(config, weights, state, window, &feats, &mut probs);

    decide(feats, probs, alert)
}

/// Shared tail of both `process_window` variants: take the probabilities the forward pass
/// produced, advance the alert machine, and bundle the [`Outcome`].
fn decide(
    features: [f32; FEATURE_LEN],
    probs: [f32; N_CLASSES],
    alert: &mut AlertMachine,
) -> Outcome {
    let alert_state = alert.update(&probs);
    let (class, confidence) = top_class(&probs);

    Outcome {
        features,
        probs,
        class,
        confidence,
        state: alert_state,
    }
}
