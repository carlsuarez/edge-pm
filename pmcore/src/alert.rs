//! Stage 4 — the decision state machine over the model's output.
//!
//! Each window the model yields a probability per [`Class`](crate::model::Class). The node
//! enters [`State::Alert`] when confidence in any fault class exceeds
//! [`ALERT_CONFIDENCE`], and returns to [`State::Normal`] only after
//! [`NORMAL_WINDOWS_TO_CLEAR`] consecutive normal-class windows — hysteresis that keeps a
//! single noisy window from flapping the alert. In firmware the state drives the LED blink
//! pattern and the UART log line; here the logic is platform-agnostic and host-testable.
//!
//! The transitions (Milestone D) are driven one window at a time by
//! [`AlertMachine::update`]; [`crate::pipeline::process_window`] feeds it the model output.

use crate::model::{Class, N_CLASSES};

/// Fault-class confidence above which the node raises an alert.
pub const ALERT_CONFIDENCE: f32 = 0.80;

/// Consecutive normal-class windows required to clear an alert (hysteresis).
pub const NORMAL_WINDOWS_TO_CLEAR: u32 = 3;

/// Whether the node is currently flagging a fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// No fault flagged.
    Normal,
    /// A fault class crossed [`ALERT_CONFIDENCE`]; `class` is the latched fault.
    Alert {
        /// The fault class that triggered (and sustains) the alert.
        class: Class,
    },
}

/// The most-likely class for a window and its probability.
///
/// Argmax over the softmax output, ties broken toward the lower index (same rule as
/// [`engine::math::argmax`]). The four probabilities sum to one, so at most one class can
/// exceed 0.5 — and thus the [`ALERT_CONFIDENCE`] = 0.80 threshold — making "the dominant
/// class" and "the class that crosses the threshold" the same question.
pub fn top_class(probs: &[f32; N_CLASSES]) -> (Class, f32) {
    let (idx, max_val) = engine::math::argmax(probs);
    // `argmax` returns `0..N_CLASSES`, which `from_index` covers exactly; `Normal` is an
    // unreachable fallback that keeps this panic-free.
    (Class::from_index(idx).unwrap_or(Class::Normal), max_val)
}

/// The `NORMAL ⇄ ALERT` decision state machine with clear-side hysteresis.
///
/// Fed one window's class probabilities per [`update`](Self::update). It latches into
/// [`State::Alert`] the moment a fault class exceeds the confidence threshold, and only
/// clears after [`NORMAL_WINDOWS_TO_CLEAR`] *consecutive* windows classify as
/// [`Class::Normal`] — so a lone noisy window neither raises a false alert (it must cross
/// 0.80) nor clears a real one (the streak resets on any non-normal window). Holds no
/// buffers; trivially `Copy`-free and `const`-constructible for a static in firmware.
#[derive(Clone, Copy, Debug)]
pub struct AlertMachine {
    state: State,
    /// Consecutive normal-class windows seen while in [`State::Alert`].
    normal_streak: u32,
    /// Confidence threshold; [`ALERT_CONFIDENCE`] by default, tunable for experiments.
    confidence: f32,
}

impl AlertMachine {
    /// A machine in [`State::Normal`] using the default [`ALERT_CONFIDENCE`].
    pub const fn new() -> Self {
        Self::with_confidence(ALERT_CONFIDENCE)
    }

    /// Like [`new`](Self::new) but with a custom fault-confidence threshold (host
    /// experiments / sensitivity sweeps; firmware uses the default).
    pub const fn with_confidence(confidence: f32) -> Self {
        Self {
            state: State::Normal,
            normal_streak: 0,
            confidence,
        }
    }

    /// The current state.
    pub fn state(&self) -> State {
        self.state
    }

    /// Advance the machine by one window's probabilities and return the new state.
    pub fn update(&mut self, probs: &[f32; N_CLASSES]) -> State {
        let (top, conf) = top_class(probs);
        let crosses = top.is_fault() && conf > self.confidence;
        match self.state {
            State::Normal => {
                if crosses {
                    self.state = State::Alert { class: top };
                    self.normal_streak = 0;
                }
            }
            State::Alert { .. } => {
                if top == Class::Normal {
                    self.normal_streak += 1;
                    if self.normal_streak >= NORMAL_WINDOWS_TO_CLEAR {
                        self.state = State::Normal;
                        self.normal_streak = 0;
                    }
                } else {
                    // A fault window sustains the alert and breaks the clear streak; if it
                    // clears the threshold too, re-latch onto the now-dominant fault.
                    self.normal_streak = 0;
                    if crosses {
                        self.state = State::Alert { class: top };
                    }
                }
            }
        }
        self.state
    }
}

impl Default for AlertMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alert_latches_the_triggering_class() {
        // The alert state carries which fault tripped it — what the LED/UART layer reads.
        let s = State::Alert {
            class: Class::OuterRace,
        };
        assert_ne!(s, State::Normal);
        let State::Alert { class } = s else {
            panic!("expected Alert");
        };
        assert_eq!(class.name(), "outer_race");
        assert!(class.is_fault());
    }

    /// Probabilities with `conf` on class `idx` and the rest split evenly — a valid softmax
    /// distribution whose argmax is `idx`.
    fn probs(idx: usize, conf: f32) -> [f32; N_CLASSES] {
        let mut p = [(1.0 - conf) / (N_CLASSES as f32 - 1.0); N_CLASSES];
        p[idx] = conf;
        p
    }

    #[test]
    fn top_class_is_argmax_with_its_probability() {
        let (c, conf) = top_class(&probs(2, 0.7));
        assert_eq!(c, Class::OuterRace);
        assert!((conf - 0.7).abs() < 1e-6);
    }

    #[test]
    fn stays_normal_below_threshold() {
        let mut m = AlertMachine::new();
        // Fault is the top class but only at 0.70 — under the 0.80 bar.
        assert_eq!(m.update(&probs(1, 0.70)), State::Normal);
        assert_eq!(m.update(&probs(1, 0.80)), State::Normal); // strictly-greater, 0.80 is not "exceeds"
    }

    #[test]
    fn raises_alert_when_a_fault_exceeds_threshold() {
        let mut m = AlertMachine::new();
        assert_eq!(
            m.update(&probs(1, 0.90)),
            State::Alert {
                class: Class::InnerRace
            }
        );
    }

    #[test]
    fn a_confident_normal_window_never_alerts() {
        let mut m = AlertMachine::new();
        // Class 0 (Normal) at 0.99 is not a fault — no alert regardless of confidence.
        assert_eq!(m.update(&probs(0, 0.99)), State::Normal);
    }

    #[test]
    fn clears_only_after_three_consecutive_normal_windows() {
        let mut m = AlertMachine::new();
        m.update(&probs(2, 0.95)); // -> Alert(OuterRace)
        assert!(matches!(m.state(), State::Alert { .. }));

        assert!(matches!(m.update(&probs(0, 0.6)), State::Alert { .. })); // normal #1
        assert!(matches!(m.update(&probs(0, 0.6)), State::Alert { .. })); // normal #2
        assert_eq!(m.update(&probs(0, 0.6)), State::Normal); // normal #3 -> clear
    }

    #[test]
    fn one_fault_window_resets_the_clear_streak() {
        let mut m = AlertMachine::new();
        m.update(&probs(2, 0.95)); // -> Alert(OuterRace)
        m.update(&probs(0, 0.6)); // normal #1
        m.update(&probs(0, 0.6)); // normal #2
        m.update(&probs(2, 0.5)); // fault window (below thr) — streak resets, still latched
        assert!(matches!(m.state(), State::Alert { .. }));
        // Need three fresh normals again, not one.
        m.update(&probs(0, 0.6));
        m.update(&probs(0, 0.6));
        assert!(matches!(m.state(), State::Alert { .. }));
        assert_eq!(m.update(&probs(0, 0.6)), State::Normal);
    }

    #[test]
    fn re_latches_onto_a_newly_dominant_fault() {
        let mut m = AlertMachine::new();
        assert_eq!(
            m.update(&probs(1, 0.90)),
            State::Alert {
                class: Class::InnerRace
            }
        );
        // A different fault now crosses the threshold: the latched class follows it.
        assert_eq!(
            m.update(&probs(3, 0.85)),
            State::Alert {
                class: Class::RollingElement
            }
        );
    }
}
