//! Per-window run state: every activation buffer, carved from the [`Arena`] once.
//!
//! [`RunState`] mirrors tiny-infer's `RunState`: it owns *all* of the intermediate
//! activation buffers the forward pass reads and writes, each a disjoint sub-slice of a
//! single caller-provided [`Arena`]. It holds no weights and no [`ModelConfig`] — the free
//! [`forward`](crate::model::forward) function is handed the state, the weights, and the
//! config as three separate arguments. The buffer sizes come straight from the config and
//! sum to exactly [`ModelConfig::arena_floats`], so a forward pass allocates nothing.
//!
//! [`Arena`]: engine::Arena
//! [`ModelConfig::arena_floats`]: crate::model::ModelConfig::arena_floats

use engine::{Arena, EngineError};

use crate::features::{Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use crate::model::{ModelConfig, N_CLASSES};

/// Why a [`RunState`] accessor rejected its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStateError {
    /// The requested channel index is `>= N_AXES`.
    BadChannel,
}

/// All mutable activation buffers a single forward pass reads and writes.
///
/// Every field borrows a disjoint sub-slice of one arena block (`'buf`); because
/// [`Arena::alloc`] ties each slice to the block's lifetime rather than to the arena
/// borrow, they can all be held at once. Reused in place for every window, so the
/// steady-state pipeline allocates nothing.
pub struct RunState<'buf> {
    /// De-interleaved input, `[N_AXES, WINDOW_LEN]` (channel-major, `f32`).
    pub x: &'buf mut [f32],
    /// conv1 → relu output, `[c1, l1]`.
    pub x_c1: &'buf mut [f32],
    /// conv2 → relu output, `[c2, l2]`.
    pub x_c2: &'buf mut [f32],
    /// Global-average-pool result with the fused features appended, `[c2 + FEATURE_LEN]`.
    pub glb_avg_pool: &'buf mut [f32],
    /// Dense-head output, `[N_CLASSES]` — class probabilities after
    /// [`forward`](crate::model::forward) (the closing softmax writes them here).
    pub logits: &'buf mut [f32],
}

impl<'buf> RunState<'buf> {
    /// Carve every activation buffer out of `arena` for `config`.
    ///
    /// Allocates in a fixed order whose total is exactly [`ModelConfig::arena_floats`].
    ///
    /// # Errors
    /// [`EngineError::ArenaOverflow`] if the arena is smaller than the budget (size it with
    /// [`ModelConfig::arena_floats`] to guarantee a fit).
    ///
    /// [`ModelConfig::arena_floats`]: crate::model::ModelConfig::arena_floats
    pub fn new(
        arena: &mut Arena<'buf>,
        config: &ModelConfig,
    ) -> Result<RunState<'buf>, EngineError> {
        Ok(RunState {
            x: arena.alloc(N_AXES * WINDOW_LEN)?,
            x_c1: arena.alloc(config.c1 * config.l1())?,
            x_c2: arena.alloc(config.c2 * config.l2())?,
            glb_avg_pool: arena.alloc(config.c2 + FEATURE_LEN)?,
            logits: arena.alloc(N_CLASSES)?,
        })
    }

    /// De-interleave a window of 3-axis samples into the channel-major [`x`](Self::x) buffer
    /// (axis `ch` occupies `x[ch*WINDOW_LEN..][..WINDOW_LEN]`), ready for `conv1d`.
    pub fn x_from_window(&mut self, window: &[Sample; WINDOW_LEN]) {
        for ch in 0..N_AXES {
            let row = &mut self.x[ch * WINDOW_LEN..][..WINDOW_LEN];
            for (slot, sample) in row.iter_mut().zip(window.iter()) {
                *slot = sample[ch] as f32;
            }
        }
    }

    /// Borrow one de-interleaved axis of the input buffer, `x[channel*WINDOW_LEN..][..WINDOW_LEN]`.
    ///
    /// # Errors
    /// [`RunStateError::BadChannel`] if `channel >= N_AXES`.
    pub fn get_channel(&self, channel: usize) -> Result<&[f32], RunStateError> {
        if channel >= N_AXES {
            return Err(RunStateError::BadChannel);
        }

        Ok(&self.x[channel * WINDOW_LEN..][..WINDOW_LEN])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::vec;

    fn cfg() -> ModelConfig {
        ModelConfig {
            c1: 16,
            k1: 7,
            s1: 2,
            c2: 32,
            k2: 5,
            s2: 2,
        }
    }

    #[test]
    fn carves_all_buffers_and_consumes_exactly_the_budget() {
        let c = cfg();
        let total = c.arena_floats();
        let mut buf = vec![0.0f32; total];
        let mut arena = Arena::new(&mut buf);
        let s = RunState::new(&mut arena, &c).unwrap();

        assert_eq!(s.x.len(), N_AXES * WINDOW_LEN);
        assert_eq!(s.x_c1.len(), c.c1 * c.l1());
        assert_eq!(s.x_c2.len(), c.c2 * c.l2());
        assert_eq!(s.glb_avg_pool.len(), c.c2 + FEATURE_LEN);
        assert_eq!(s.logits.len(), N_CLASSES);
        // The budget is exact: the activations fill the arena with nothing left over.
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn too_small_arena_overflows() {
        let c = cfg();
        let mut buf = vec![0.0f32; c.arena_floats() - 1];
        let mut arena = Arena::new(&mut buf);
        assert!(matches!(
            RunState::new(&mut arena, &c),
            Err(EngineError::ArenaOverflow { .. })
        ));
    }

    #[test]
    fn x_from_window_de_interleaves_by_axis() {
        let c = cfg();
        let mut buf = vec![0.0f32; c.arena_floats()];
        let mut arena = Arena::new(&mut buf);
        let mut s = RunState::new(&mut arena, &c).unwrap();

        let window: [Sample; WINDOW_LEN] = core::array::from_fn(|t| [t as i16, -(t as i16), 100]);
        s.x_from_window(&window);
        // Each axis is contiguous: x[axis*WINDOW_LEN + t] == window[t][axis].
        assert_eq!(s.x[0], 0.0);
        assert_eq!(s.x[5], 5.0);
        assert_eq!(s.x[WINDOW_LEN + 5], -5.0);
        assert_eq!(s.x[2 * WINDOW_LEN + 5], 100.0);
    }
}
