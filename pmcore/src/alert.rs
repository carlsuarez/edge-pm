//! Stage 4 — the decision state machine over the model's output.
//!
//! Each window the model yields a probability per [`Class`](crate::model::Class). The node
//! enters [`State::Alert`] when confidence in any fault class exceeds
//! [`ALERT_CONFIDENCE`], and returns to [`State::Normal`] only after
//! [`NORMAL_WINDOWS_TO_CLEAR`] consecutive normal-class windows — hysteresis that keeps a
//! single noisy window from flapping the alert. In firmware the state drives the LED blink
//! pattern and the UART log line; here the logic is platform-agnostic and host-testable.
//!
//! State-machine transitions land in **Milestone D**.

use crate::model::Class;

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
}
