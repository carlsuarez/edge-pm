//! Stage 3 — the 1-D CNN bearing-health classifier.
//!
//! Loads a model exported offline (PyTorch, CWRU Bearing Dataset) as a flat binary —
//! a header of layer dimensions followed by raw `f32` (or, later, int8) weight arrays in
//! row-major order, the same loading convention as tiny-infer's Llama checkpoint — and
//! runs a forward pass over a [`crate::features`] vector to a probability distribution
//! over the four [`Class`]es.
//!
//! The forward pass is built from [`engine::nn`]'s `conv1d` → `relu` → `global_avg_pool`
//! for the convolutional stack, then a dense layer + [`engine::math::softmax`] head. All
//! working buffers come from a caller-provided [`engine::Arena`], sized once at startup.
//!
//! Model format, loader, and forward pass land in **Milestone C**.

/// Number of bearing-health classes the model discriminates.
pub const N_CLASSES: usize = 4;

/// Bearing-health classes, in the model's output-index order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Healthy bearing.
    Normal,
    /// Inner-race fault.
    InnerRace,
    /// Outer-race fault.
    OuterRace,
    /// Rolling-element fault.
    RollingElement,
}

impl Class {
    /// The class for a model output index `0..`[`N_CLASSES`], if in range.
    pub fn from_index(i: usize) -> Option<Class> {
        Some(match i {
            0 => Class::Normal,
            1 => Class::InnerRace,
            2 => Class::OuterRace,
            3 => Class::RollingElement,
            _ => return None,
        })
    }

    /// Lower-case name used in the UART alert log (`class=<name>`).
    pub fn name(self) -> &'static str {
        match self {
            Class::Normal => "normal",
            Class::InnerRace => "inner_race",
            Class::OuterRace => "outer_race",
            Class::RollingElement => "rolling_element",
        }
    }

    /// Whether this class represents a fault (anything but [`Class::Normal`]).
    pub fn is_fault(self) -> bool {
        self != Class::Normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_index_roundtrips_and_names() {
        for i in 0..N_CLASSES {
            let c = Class::from_index(i).unwrap();
            assert!(!c.name().is_empty());
        }
        assert_eq!(Class::from_index(N_CLASSES), None);
        assert_eq!(Class::Normal.name(), "normal");
        assert!(!Class::Normal.is_fault());
        assert!(Class::OuterRace.is_fault());
    }
}
