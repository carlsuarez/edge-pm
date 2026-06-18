//! # pmcore
//!
//! The portable, `no_std`, allocation-free core of the edge-pm predictive-maintenance
//! node. Everything here runs **identically** on the host (driven by [`host-sim`] over
//! recorded data) and on the STM32F411 firmware — the only difference is who feeds it
//! samples. Keeping the signal-processing and inference logic in this shared library is
//! what lets the whole pipeline be developed and unit-tested on a laptop before any
//! board, real or emulated, is involved.
//!
//! The four stages of the pipeline map to the four modules:
//!
//! * [`features`] — turn a window of raw accelerometer samples into a feature vector
//!   (RMS, crest factor, kurtosis, optionally an FFT magnitude spectrum).
//! * [`model`] — run the 1-D CNN forward pass over that vector (built on
//!   [`engine::nn`]'s `conv1d` / `relu` / `global_avg_pool` plus a dense + softmax head)
//!   to a probability over the four bearing-health [`Class`](model::Class)es.
//! * [`pipeline`] — the platform-agnostic windowing/handoff glue between stages.
//! * [`alert`] — the `NORMAL ⇄ ALERT` decision state machine over the model's output.
//!
//! Allocation-free throughout: every working buffer is caller-provided (on-device, carved
//! once from a static [`engine::Arena`]; on the host, a plain stack/heap buffer).
//!
//! [`host-sim`]: ../host_sim/index.html

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod alert;
pub mod features;
pub mod model;
pub mod pipeline;
pub mod state;
pub use state::RunState;

/// Re-export of the engine's bump [`Arena`](engine::Arena): callers carve a [`RunState`] from
/// one at startup (sized by [`ModelConfig::buf_len`](model::ModelConfig::buf_len))
/// and reuse it for every window, so they need the type without depending on `engine` directly.
pub use engine::Arena;
