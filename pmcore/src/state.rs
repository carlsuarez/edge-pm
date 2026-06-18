//! Per-window run state: every activation buffer, carved from the [`Arena`] once.
//!
//! [`RunState`] mirrors tiny-infer's `RunState`: it owns *all* of the intermediate
//! activation buffers the forward pass reads and writes, each a disjoint sub-slice of a
//! single caller-provided [`Arena`]. It holds no weights and no [`ModelConfig`] — the free
//! [`forward`](crate::model::forward) function is handed the state, the weights, and the
//! config as separate arguments.
//!
//! It is generic over the activation element type `T` (defaulting to `f32`): the float
//! forward pass runs on `RunState<f32>` carved from an `Arena<f32>`, and the integer-only
//! pass on `RunState<i8>` carved from an `Arena<i8>` — same layout, same code, only the type
//! differs. The buffer sizes come straight from the config and sum to exactly
//! [`ModelConfig::buf_len`], so a forward pass allocates nothing.
//!
//! [`Arena`]: engine::Arena
//! [`ModelConfig::buf_len`]: crate::model::ModelConfig::buf_len

use engine::{Arena, EngineError};

use crate::features::{Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use crate::model::ModelConfig;

/// Why a [`RunState`] accessor rejected its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStateError {
    /// The requested channel index is `>= N_AXES`.
    BadChannel,
}

/// All mutable activation buffers a single forward pass reads and writes, over the element
/// type `T` (defaults to `f32`).
///
/// Every field borrows a disjoint sub-slice of one arena block (`'buf`); because
/// [`Arena::alloc`] ties each slice to the block's lifetime rather than to the arena
/// borrow, they can all be held at once. Reused in place for every window, so the
/// steady-state pipeline allocates nothing. The float (`RunState<f32>`) and integer-only
/// (`RunState<i8>`) paths share this one type; only the dense **logits** differ in width
/// (`f32` vs `i32`) and so are kept as a forward-pass local rather than a field here.
pub struct RunState<'buf, T = f32> {
    /// Input, `[N_AXES, WINDOW_LEN]` (channel-major): de-interleaved `f32` on the float
    /// path, de-interleaved + quantized int8 on the integer path.
    pub x: &'buf mut [T],
    /// conv1 → relu output, `[c1, l1]`.
    pub x_c1: &'buf mut [T],
    /// conv2 → relu output, `[c2, l2]`.
    pub x_c2: &'buf mut [T],
    /// Global-average-pool result with the fused features appended, `[c2 + FEATURE_LEN]` —
    /// the dense head's input.
    pub fc_in: &'buf mut [T],
}

impl<'buf, T: Copy + Default> RunState<'buf, T> {
    /// Carve every activation buffer out of `arena` for `config`, in a fixed order whose
    /// total is exactly [`ModelConfig::buf_len`].
    ///
    /// # Errors
    /// [`EngineError::ArenaOverflow`] if the arena is smaller than the budget (size it with
    /// [`ModelConfig::buf_len`] to guarantee a fit).
    ///
    /// [`ModelConfig::buf_len`]: crate::model::ModelConfig::buf_len
    pub fn new(
        arena: &mut Arena<'buf, T>,
        config: &ModelConfig,
    ) -> Result<RunState<'buf, T>, EngineError> {
        Ok(RunState {
            x: arena.alloc(N_AXES * WINDOW_LEN)?,
            x_c1: arena.alloc(config.c1 * config.l1())?,
            x_c2: arena.alloc(config.c2 * config.l2())?,
            fc_in: arena.alloc(config.c2 + FEATURE_LEN)?,
        })
    }
}

impl<'buf> RunState<'buf, f32> {
    /// De-interleave a window of 3-axis samples into the channel-major [`x`](Self::x) buffer
    /// (axis `ch` occupies `x[ch*WINDOW_LEN..][..WINDOW_LEN]`), ready for `conv1d`. Float
    /// path only — the integer path de-interleaves *and* quantizes in `forward_q8`.
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
        let total = c.buf_len();
        let mut buf = vec![0.0f32; total];
        let mut arena = Arena::new(&mut buf);
        let s = RunState::new(&mut arena, &c).unwrap();

        assert_eq!(s.x.len(), N_AXES * WINDOW_LEN);
        assert_eq!(s.x_c1.len(), c.c1 * c.l1());
        assert_eq!(s.x_c2.len(), c.c2 * c.l2());
        assert_eq!(s.fc_in.len(), c.c2 + FEATURE_LEN);
        // The budget is exact: the activations fill the arena with nothing left over.
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn carves_an_int8_state_from_the_same_budget() {
        // The same generic carves an i8 working set from an i8 arena of the same length.
        let c = cfg();
        let mut buf = vec![0i8; c.buf_len()];
        let mut arena = Arena::new(&mut buf);
        let s: RunState<i8> = RunState::new(&mut arena, &c).unwrap();
        assert_eq!(s.x.len(), N_AXES * WINDOW_LEN);
        assert_eq!(s.fc_in.len(), c.c2 + FEATURE_LEN);
        assert_eq!(arena.remaining(), 0);
    }

    #[test]
    fn too_small_arena_overflows() {
        let c = cfg();
        let mut buf = vec![0.0f32; c.buf_len() - 1];
        let mut arena = Arena::new(&mut buf);
        assert!(matches!(
            RunState::new(&mut arena, &c),
            Err(EngineError::ArenaOverflow { .. })
        ));
    }

    #[test]
    fn x_from_window_de_interleaves_by_axis() {
        let c = cfg();
        let mut buf = vec![0.0f32; c.buf_len()];
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
