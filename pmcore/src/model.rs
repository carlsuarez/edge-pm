//! Stage 3 — the 1-D CNN bearing-health classifier.
//!
//! # Architecture: a hybrid 1-D CNN
//!
//! The convolutional stack runs on the **raw acquisition window** — `N_AXES` channels of
//! `WINDOW_LEN` samples — which is where a convolution is meaningful: it slides over the
//! time axis of the vibration signal, where translation invariance actually holds. The
//! `WINDOW_LEN`-wide signal is fed through two `Conv1D → ReLU` blocks (each with a stride
//! that shrinks the time axis) and a `GlobalAveragePool` that collapses time to one value
//! per channel. The pooled vector is then **concatenated with the 9 hand-crafted features**
//! from [`crate::features`] and passed through a dense layer + softmax to a probability over
//! the four [`Class`]es.
//!
//! Fusing the learned conv features with the classical statistical features (RMS, crest,
//! kurtosis) is a common, defensible design for bearing diagnosis: the conv stack captures
//! waveform shape, the hand-crafted features capture distributional shape (impulsiveness),
//! and the dense head learns to weigh both. It also ties the pipeline together — Stage 2's
//! output is not just logged, it feeds the classifier.
//!
//! # The three pieces (mirroring tiny-infer)
//!
//! Loading and running a model is split into three things that **own each other not at
//! all**: a [`ModelConfig`] of layer dimensions (with the sizing helpers), a [`Weights`]
//! bundle of zero-copy `f32` views into the checkpoint, and a [`RunState`] of activation
//! buffers carved from an [`engine::Arena`]. The forward pass is the free function
//! [`forward`], handed all three. Sizing, weights, and scratch are therefore independent
//! and individually testable.
//!
//! ```text
//! window[N_AXES, WINDOW_LEN]
//!   → conv1d → relu        [c1, l1]
//!   → conv1d → relu        [c2, l2]
//!   → global_avg_pool      [c2]
//!   → concat features      [c2 + FEATURE_LEN]
//!   → dense (matmul+bias)  [N_CLASSES]
//!   → softmax              [N_CLASSES]   (probabilities over the four classes)
//! ```
//!
//! # On-disk format
//!
//! A 64-byte header (`"epm1"` magic, version, then the layer dimensions) followed by the
//! raw little-endian `f32` weights in PyTorch order — the same load-from-flash convention
//! as tiny-infer's checkpoints. See `tools/export_model.py`.

use engine::nn::conv1d_out_len;
use engine::{math, nn};

use crate::features::{Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use crate::RunState;

/// Number of bearing-health classes the model discriminates.
pub const N_CLASSES: usize = 4;

/// Magic at the head of an exported model: `"epm1"`.
const MAGIC: [u8; 4] = *b"epm1";

/// On-disk format version this loader understands.
const VERSION: i32 = 1;

/// Fixed header size in bytes; weights begin here (4-aligned).
const HEADER_BYTES: usize = 64;

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

/// Dimensions of the two convolutional layers (everything else is fixed by the pipeline:
/// `N_AXES` input channels, `WINDOW_LEN` input samples, `FEATURE_LEN` fused features,
/// `N_CLASSES` outputs).
///
/// A plain `Copy` value object: it owns no weights and no buffers, just the arithmetic
/// needed to size them ([`l1`](Self::l1) / [`l2`](Self::l2) / [`arena_floats`](Self::arena_floats)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConfig {
    /// conv1 output channels.
    pub c1: usize,
    /// conv1 kernel length.
    pub k1: usize,
    /// conv1 stride.
    pub s1: usize,
    /// conv2 output channels.
    pub c2: usize,
    /// conv2 kernel length.
    pub k2: usize,
    /// conv2 stride.
    pub s2: usize,
}

impl ModelConfig {
    /// conv1 output length over the `WINDOW_LEN` input.
    pub const fn l1(&self) -> usize {
        conv1d_out_len(WINDOW_LEN, self.k1, self.s1)
    }

    /// conv2 output length over conv1's output.
    pub const fn l2(&self) -> usize {
        conv1d_out_len(self.l1(), self.k2, self.s2)
    }

    /// `f32` elements a forward pass carves from the arena — size a static arena with this.
    pub const fn arena_floats(&self) -> usize {
        N_AXES * WINDOW_LEN          // de-interleaved input
            + self.c1 * self.l1()    // conv1 output
            + self.c2 * self.l2()    // conv2 output
            + self.c2 + FEATURE_LEN  // pooled + fused features
            + N_CLASSES // logits
    }

    /// Total `f32` weights the on-disk format carries for this config.
    const fn weight_floats(&self) -> usize {
        self.c1 * N_AXES * self.k1 + self.c1            // conv1 weight + bias
            + self.c2 * self.c1 * self.k2 + self.c2     // conv2 weight + bias
            + N_CLASSES * (self.c2 + FEATURE_LEN) + N_CLASSES // dense weight + bias
    }

    /// Parse and validate the 64-byte header, returning just the layer dimensions.
    ///
    /// Checks the `"epm1"` magic, the version, and that the fixed dimensions (input
    /// channels / window / features / classes) match the compiled pipeline. It does *not*
    /// look at the weight payload — [`Weights::load`] does the size and alignment checks.
    ///
    /// # Errors
    /// [`ModelError::TooShort`], [`ModelError::BadMagic`], [`ModelError::UnsupportedVersion`],
    /// or [`ModelError::ConfigMismatch`].
    pub fn parse(bytes: &[u8]) -> Result<ModelConfig, ModelError> {
        if bytes.len() < HEADER_BYTES {
            return Err(ModelError::TooShort);
        }
        if bytes[..4] != MAGIC {
            return Err(ModelError::BadMagic);
        }
        let version = rd_i32(bytes, 4);
        if version != VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }
        // Fixed dimensions must match the compiled pipeline.
        if rd_i32(bytes, 8) as usize != N_AXES
            || rd_i32(bytes, 12) as usize != WINDOW_LEN
            || rd_i32(bytes, 40) as usize != FEATURE_LEN
            || rd_i32(bytes, 44) as usize != N_CLASSES
        {
            return Err(ModelError::ConfigMismatch);
        }
        Ok(ModelConfig {
            c1: rd_i32(bytes, 16) as usize,
            k1: rd_i32(bytes, 20) as usize,
            s1: rd_i32(bytes, 24) as usize,
            c2: rd_i32(bytes, 28) as usize,
            k2: rd_i32(bytes, 32) as usize,
            s2: rd_i32(bytes, 36) as usize,
        })
    }
}

/// Why loading a model failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// Fewer than `HEADER_BYTES` of input, or the weights are truncated.
    TooShort,
    /// Leading bytes are not the `"epm1"` magic.
    BadMagic,
    /// Header version this loader does not understand.
    UnsupportedVersion(i32),
    /// A fixed dimension (input channels / window / features / classes) did not match the
    /// compiled pipeline, so the model and firmware disagree.
    ConfigMismatch,
    /// File length does not equal the header + the weights the config implies.
    SizeMismatch {
        /// Bytes the config implies.
        expected: usize,
        /// Bytes actually present.
        got: usize,
    },
    /// The weight region is not 4-byte aligned (cannot view it as `f32`). On a little-endian
    /// target with the model in aligned flash this never happens.
    Misaligned,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ModelError::TooShort => write!(f, "model file is too short"),
            ModelError::BadMagic => write!(f, "not an edge-pm model (bad magic)"),
            ModelError::UnsupportedVersion(v) => write!(f, "unsupported model version {v}"),
            ModelError::ConfigMismatch => {
                write!(f, "model dimensions do not match the compiled pipeline")
            }
            ModelError::SizeMismatch { expected, got } => {
                write!(
                    f,
                    "model size mismatch: expected {expected} bytes, got {got}"
                )
            }
            ModelError::Misaligned => write!(f, "model weight region is not 4-byte aligned"),
        }
    }
}

/// Zero-copy views of each weight tensor in a loaded model.
///
/// Like tiny-infer's `Weights`, every field borrows a sub-slice straight out of the
/// checkpoint's `f32` region — nothing is copied or reshaped. Holds no [`ModelConfig`]; the
/// config is parsed alongside it (see [`Weights::load`]) and passed to [`forward`]
/// separately.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'w> {
    /// conv1 weight, `[c1, N_AXES, k1]`.
    pub conv1_w: &'w [f32],
    /// conv1 bias, `[c1]`.
    pub conv1_b: &'w [f32],
    /// conv2 weight, `[c2, c1, k2]`.
    pub conv2_w: &'w [f32],
    /// conv2 bias, `[c2]`.
    pub conv2_b: &'w [f32],
    /// dense weight, `[N_CLASSES, c2 + FEATURE_LEN]`.
    pub fc_w: &'w [f32],
    /// dense bias, `[N_CLASSES]`.
    pub fc_b: &'w [f32],
}

impl<'w> Weights<'w> {
    /// Carve the six weight tensors out of the checkpoint's `f32` region — the bytes after
    /// the 64-byte header, reinterpreted as `f32` — in declaration (PyTorch) order.
    ///
    /// # Errors
    /// [`ModelError::SizeMismatch`] if `floats` is shorter than `config` requires.
    pub fn new(floats: &'w [f32], config: &ModelConfig) -> Result<Weights<'w>, ModelError> {
        let needed = config.weight_floats();
        if floats.len() < needed {
            return Err(ModelError::SizeMismatch {
                expected: needed * 4,
                got: core::mem::size_of_val(floats),
            });
        }

        // Bump a cursor through the slice, taking each tensor in declaration order.
        let mut o = 0;
        let mut take = |n: usize| {
            let s = &floats[o..o + n];
            o += n;
            s
        };
        let conv1_w = take(config.c1 * N_AXES * config.k1);
        let conv1_b = take(config.c1);
        let conv2_w = take(config.c2 * config.c1 * config.k2);
        let conv2_b = take(config.c2);
        let fc_w = take(N_CLASSES * (config.c2 + FEATURE_LEN));
        let fc_b = take(N_CLASSES);

        Ok(Weights {
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc_w,
            fc_b,
        })
    }

    /// Parse a model from its on-disk bytes (header + weights): validate the header via
    /// [`ModelConfig::parse`], check the exact file size and 4-byte alignment, then slice
    /// each weight tensor in place. Returns the config and the views side by side — neither
    /// owns the other.
    ///
    /// `bytes` must be 4-byte aligned (the model in flash is; on the host, read into an
    /// aligned buffer). Little-endian, matching the export and both target CPUs.
    pub fn load(bytes: &'w [u8]) -> Result<(ModelConfig, Weights<'w>), ModelError> {
        let config = ModelConfig::parse(bytes)?;

        let expected = HEADER_BYTES + config.weight_floats() * 4;
        if bytes.len() != expected {
            return Err(ModelError::SizeMismatch {
                expected,
                got: bytes.len(),
            });
        }

        let floats: &[f32] =
            bytemuck::try_cast_slice(&bytes[HEADER_BYTES..]).map_err(|_| ModelError::Misaligned)?;
        let weights = Weights::new(floats, &config)?;
        Ok((config, weights))
    }
}

/// Read a little-endian `i32` at byte offset `off`.
fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Run the forward pass: classify `window` + its `features` into the class **probabilities**
/// (a softmax over the four [`Class`]es), written into [`state.logits`](RunState::logits).
///
/// Computes the result from the three independent pieces — the [`ModelConfig`], the
/// [`Weights`], and the [`RunState`] scratch — touching no memory beyond `state`'s
/// arena-carved buffers, so it allocates nothing and is infallible. The buffer is named
/// `logits` for the dense output it holds mid-pass; after the closing softmax it holds the
/// normalized probabilities.
pub fn forward(
    config: &ModelConfig,
    weights: &Weights,
    state: &mut RunState,
    window: &[Sample; WINDOW_LEN],
    features: &[f32; FEATURE_LEN],
) {
    state.x_from_window(window);

    // conv1 → relu
    nn::conv1d(
        state.x_c1,
        state.x,
        weights.conv1_w,
        Some(weights.conv1_b),
        N_AXES,
        config.c1,
        WINDOW_LEN,
        config.k1,
        config.s1,
    );
    nn::relu(state.x_c1);

    // conv2 → relu
    nn::conv1d(
        state.x_c2,
        state.x_c1,
        weights.conv2_w,
        Some(weights.conv2_b),
        config.c1,
        config.c2,
        config.l1(),
        config.k2,
        config.s2,
    );
    nn::relu(state.x_c2);

    // global average pool → concat the fused features
    let (pooled, feat_tail) = state.glb_avg_pool.split_at_mut(config.c2);
    nn::global_avg_pool(pooled, state.x_c2, config.c2, config.l2());
    feat_tail.copy_from_slice(features);

    // dense + bias → logits → softmax
    math::matmul(
        state.logits,
        state.glb_avg_pool,
        weights.fc_w,
        config.c2 + FEATURE_LEN,
        N_CLASSES,
    );
    math::add_bias(state.logits, weights.fc_b);
    math::softmax(state.logits);
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use engine::Arena;
    use std::vec;
    use std::vec::Vec;

    fn close(a: f32, b: f32) -> bool {
        libm::fabsf(a - b) < 1e-5
    }

    // Build an on-disk model blob for `cfg` from the six weight tensors (in order).
    fn build_blob(cfg: &ModelConfig, weights: &[f32]) -> Vec<u8> {
        assert_eq!(weights.len(), cfg.weight_floats());
        let mut b = vec![0u8; HEADER_BYTES];
        b[..4].copy_from_slice(&MAGIC);
        let put = |b: &mut [u8], off: usize, v: i32| {
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut b, 4, VERSION);
        put(&mut b, 8, N_AXES as i32);
        put(&mut b, 12, WINDOW_LEN as i32);
        put(&mut b, 16, cfg.c1 as i32);
        put(&mut b, 20, cfg.k1 as i32);
        put(&mut b, 24, cfg.s1 as i32);
        put(&mut b, 28, cfg.c2 as i32);
        put(&mut b, 32, cfg.k2 as i32);
        put(&mut b, 36, cfg.s2 as i32);
        put(&mut b, 40, FEATURE_LEN as i32);
        put(&mut b, 44, N_CLASSES as i32);
        for &w in weights {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    // Run the full forward pass over a freshly-carved arena, returning the probabilities
    // (forward applies the closing softmax itself).
    fn run(
        cfg: &ModelConfig,
        w: &Weights,
        window: &[Sample; WINDOW_LEN],
        feats: &[f32; FEATURE_LEN],
    ) -> [f32; N_CLASSES] {
        let mut scratch = vec![0.0f32; cfg.arena_floats()];
        let mut arena = Arena::new(&mut scratch);
        let mut state = RunState::new(&mut arena, cfg).unwrap();
        forward(cfg, w, &mut state, window, feats);
        let mut probs = [0.0f32; N_CLASSES];
        probs.copy_from_slice(state.logits);
        probs
    }

    #[test]
    fn class_index_roundtrips_and_names() {
        for i in 0..N_CLASSES {
            assert!(!Class::from_index(i).unwrap().name().is_empty());
        }
        assert_eq!(Class::from_index(N_CLASSES), None);
        assert!(!Class::Normal.is_fault());
        assert!(Class::OuterRace.is_fault());
    }

    #[test]
    fn arena_floats_matches_hand_count() {
        let cfg = ModelConfig {
            c1: 16,
            k1: 7,
            s1: 2,
            c2: 32,
            k2: 5,
            s2: 2,
        };
        // l1 = (512-7)/2+1 = 253 ; l2 = (253-5)/2+1 = 125
        assert_eq!(cfg.l1(), 253);
        assert_eq!(cfg.l2(), 125);
        let expect = 3 * 512 + 16 * 253 + 32 * 125 + (32 + 9) + 4;
        assert_eq!(cfg.arena_floats(), expect);
    }

    #[test]
    fn load_rejects_bad_magic_and_size() {
        let cfg = ModelConfig {
            c1: 2,
            k1: 3,
            s1: 1,
            c2: 2,
            k2: 3,
            s2: 1,
        };
        let blob = build_blob(&cfg, &vec![0.0; cfg.weight_floats()]);

        let mut bad = blob.clone();
        bad[0] = b'X';
        assert_eq!(Weights::load(&bad).err(), Some(ModelError::BadMagic));

        let truncated = &blob[..blob.len() - 4];
        assert!(matches!(
            Weights::load(truncated),
            Err(ModelError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn load_returns_config_and_views_that_partition_the_payload() {
        let cfg = ModelConfig {
            c1: 2,
            k1: 3,
            s1: 1,
            c2: 2,
            k2: 3,
            s2: 1,
        };
        // Fill the weights with their own index so each view reveals where it was carved.
        let w: Vec<f32> = (0..cfg.weight_floats()).map(|i| i as f32).collect();
        let blob = build_blob(&cfg, &w);
        let (parsed, views) = Weights::load(&blob).unwrap();

        assert_eq!(parsed, cfg);
        assert_eq!(views.conv1_w.len(), cfg.c1 * N_AXES * cfg.k1);
        assert_eq!(views.conv1_w[0], 0.0);
        assert_eq!(views.conv1_b[0], (cfg.c1 * N_AXES * cfg.k1) as f32);
        assert_eq!(views.fc_b.len(), N_CLASSES);
        // The last tensor ends exactly at the payload end.
        assert_eq!(views.fc_b[N_CLASSES - 1], (cfg.weight_floats() - 1) as f32);
    }

    #[test]
    fn forward_plumbing_is_softmax_of_the_dense_bias() {
        // All conv weights/biases zero → conv outputs 0 → relu 0 → pool 0; the dense
        // weights are zero too, so the logits equal the dense bias regardless of input.
        // The output is then exactly softmax(bias) — a check of the whole plumbing
        // (shapes, pooling, feature concat, bias add, softmax) with a known answer.
        let cfg = ModelConfig {
            c1: 2,
            k1: 3,
            s1: 1,
            c2: 2,
            k2: 3,
            s2: 1,
        };
        let mut w = vec![0.0f32; cfg.weight_floats()];
        // Set only the dense bias (last N_CLASSES floats): [0, ln2, 0, 0].
        let n = w.len();
        w[n - N_CLASSES..].copy_from_slice(&[0.0, core::f32::consts::LN_2, 0.0, 0.0]);
        let blob = build_blob(&cfg, &w);
        let (config, weights) = Weights::load(&blob).unwrap();

        let window = [[123i16, -7, 9]; WINDOW_LEN]; // arbitrary — output must ignore it
        let feats = [1.0f32; FEATURE_LEN];
        let probs = run(&config, &weights, &window, &feats);

        // softmax([0, ln2, 0, 0]) = [1, 2, 1, 1] / 5
        assert!(close(probs[0], 0.2));
        assert!(close(probs[1], 0.4));
        assert!(close(probs[2], 0.2));
        assert!(close(probs[3], 0.2));
    }

    #[test]
    fn forward_outputs_a_valid_distribution() {
        // Deterministic non-trivial weights → probs must be a finite distribution.
        let cfg = ModelConfig {
            c1: 4,
            k1: 5,
            s1: 3,
            c2: 4,
            k2: 3,
            s2: 2,
        };
        let w: Vec<f32> = (0..cfg.weight_floats())
            .map(|i| ((i as f32 * 0.123).sin()) * 0.05)
            .collect();
        let blob = build_blob(&cfg, &w);
        let (config, weights) = Weights::load(&blob).unwrap();

        let window: [Sample; WINDOW_LEN] =
            core::array::from_fn(|t| [(t as i16 % 50) - 25, 10, -((t as i16) % 7)]);
        let feats = [0.5f32; FEATURE_LEN];
        let probs = run(&config, &weights, &window, &feats);

        let sum: f32 = probs.iter().sum();
        assert!(close(sum, 1.0), "probs sum to {sum}");
        assert!(probs
            .iter()
            .all(|&p| (0.0..=1.0).contains(&p) && p.is_finite()));
    }
}
