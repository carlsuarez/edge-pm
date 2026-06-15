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
//! # Forward pass (built on [`engine::nn`] + [`engine::math`])
//!
//! ```text
//! window[N_AXES, WINDOW_LEN]
//!   → conv1d → relu        [c1, l1]
//!   → conv1d → relu        [c2, l2]
//!   → global_avg_pool      [c2]
//!   → concat features      [c2 + FEATURE_LEN]
//!   → dense (matmul+bias)  [N_CLASSES]
//!   → softmax              [N_CLASSES]
//! ```
//!
//! Every working buffer is carved once from a caller-provided [`engine::Arena`] sized by
//! [`ModelConfig::arena_floats`]; the forward pass allocates nothing.
//!
//! # On-disk format
//!
//! A 64-byte header (`"epm1"` magic, version, then the layer dimensions) followed by the
//! raw little-endian `f32` weights in PyTorch order — the same load-from-flash convention
//! as tiny-infer's checkpoints. See `tools/export_model.py`.

use engine::nn::conv1d_out_len;
use engine::{math, nn, Arena};

use crate::features::{Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};

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
    /// The `scratch` buffer handed to [`Model::infer`] is smaller than
    /// [`ModelConfig::arena_floats`].
    ScratchTooSmall,
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
            ModelError::ScratchTooSmall => write!(f, "inference scratch buffer is too small"),
        }
    }
}

/// A loaded model: the config plus zero-copy views of each weight tensor.
#[derive(Clone, Copy, Debug)]
pub struct Model<'w> {
    config: ModelConfig,
    conv1_w: &'w [f32],
    conv1_b: &'w [f32],
    conv2_w: &'w [f32],
    conv2_b: &'w [f32],
    fc_w: &'w [f32],
    fc_b: &'w [f32],
}

impl<'w> Model<'w> {
    /// Parse a model from its on-disk bytes (header + weights), validating magic, version,
    /// the fixed dimensions, and the file size, then slicing each weight tensor in place.
    ///
    /// `bytes` must be 4-byte aligned (the model in flash is; on the host, read into an
    /// aligned buffer). Little-endian, matching the export and both target CPUs.
    pub fn load(bytes: &'w [u8]) -> Result<Model<'w>, ModelError> {
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
        let config = ModelConfig {
            c1: rd_i32(bytes, 16) as usize,
            k1: rd_i32(bytes, 20) as usize,
            s1: rd_i32(bytes, 24) as usize,
            c2: rd_i32(bytes, 28) as usize,
            k2: rd_i32(bytes, 32) as usize,
            s2: rd_i32(bytes, 36) as usize,
        };

        let expected = HEADER_BYTES + config.weight_floats() * 4;
        if bytes.len() != expected {
            return Err(ModelError::SizeMismatch {
                expected,
                got: bytes.len(),
            });
        }

        let weights: &[f32] =
            bytemuck::try_cast_slice(&bytes[HEADER_BYTES..]).map_err(|_| ModelError::Misaligned)?;

        // Slice the tensors in declaration order.
        let mut o = 0;
        let mut take = |n: usize| {
            let s = &weights[o..o + n];
            o += n;
            s
        };
        let conv1_w = take(config.c1 * N_AXES * config.k1);
        let conv1_b = take(config.c1);
        let conv2_w = take(config.c2 * config.c1 * config.k2);
        let conv2_b = take(config.c2);
        let fc_w = take(N_CLASSES * (config.c2 + FEATURE_LEN));
        let fc_b = take(N_CLASSES);

        Ok(Model {
            config,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            fc_w,
            fc_b,
        })
    }

    /// The model's layer dimensions.
    pub fn config(&self) -> ModelConfig {
        self.config
    }

    /// Run the forward pass: classify one window + its features into `probs` (a softmax over
    /// the four [`Class`]es). `scratch` must hold at least [`ModelConfig::arena_floats`]
    /// elements; every working buffer is carved from it and nothing else allocates.
    ///
    /// # Errors
    /// [`ModelError::ScratchTooSmall`] if `scratch` is shorter than the arena budget.
    pub fn forward(
        &self,
        window: &[Sample; WINDOW_LEN],
        features: &[f32; FEATURE_LEN],
        scratch: &mut [f32],
        probs: &mut [f32; N_CLASSES],
    ) -> Result<(), ModelError> {
        let c = &self.config;
        let (l1, l2) = (c.l1(), c.l2());

        let mut arena = Arena::new(scratch);

        // De-interleave the window into channel-major f32 `[N_AXES, WINDOW_LEN]`.
        let input = arena
            .alloc(N_AXES * WINDOW_LEN)
            .map_err(|_| ModelError::ScratchTooSmall)?;
        for ch in 0..N_AXES {
            let row = &mut input[ch * WINDOW_LEN..][..WINDOW_LEN];
            for (slot, sample) in row.iter_mut().zip(window.iter()) {
                *slot = sample[ch] as f32;
            }
        }

        // conv1 → relu
        let a = arena
            .alloc(c.c1 * l1)
            .map_err(|_| ModelError::ScratchTooSmall)?;
        nn::conv1d(
            a,
            input,
            self.conv1_w,
            Some(self.conv1_b),
            N_AXES,
            c.c1,
            WINDOW_LEN,
            c.k1,
            c.s1,
        );
        nn::relu(a);

        // conv2 → relu
        let b = arena
            .alloc(c.c2 * l2)
            .map_err(|_| ModelError::ScratchTooSmall)?;
        nn::conv1d(
            b,
            a,
            self.conv2_w,
            Some(self.conv2_b),
            c.c1,
            c.c2,
            l1,
            c.k2,
            c.s2,
        );
        nn::relu(b);

        // global average pool → concat the fused features
        let cat = arena
            .alloc(c.c2 + FEATURE_LEN)
            .map_err(|_| ModelError::ScratchTooSmall)?;
        nn::global_avg_pool(&mut cat[..c.c2], b, c.c2, l2);
        cat[c.c2..].copy_from_slice(features);

        // dense + bias → softmax
        let logits = arena
            .alloc(N_CLASSES)
            .map_err(|_| ModelError::ScratchTooSmall)?;
        math::matmul(logits, cat, self.fc_w, c.c2 + FEATURE_LEN, N_CLASSES);
        math::add_bias(logits, self.fc_b);
        math::softmax(logits);

        probs.copy_from_slice(logits);
        Ok(())
    }
}

/// Read a little-endian `i32` at byte offset `off`.
fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
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
        assert_eq!(Model::load(&bad).err(), Some(ModelError::BadMagic));

        let truncated = &blob[..blob.len() - 4];
        assert!(matches!(
            Model::load(truncated),
            Err(ModelError::SizeMismatch { .. })
        ));
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
        let model = Model::load(&blob).unwrap();

        let window = [[123i16, -7, 9]; WINDOW_LEN]; // arbitrary — output must ignore it
        let feats = [1.0f32; FEATURE_LEN];
        let mut scratch = vec![0.0f32; cfg.arena_floats()];
        let mut probs = [0.0f32; N_CLASSES];
        model
            .forward(&window, &feats, &mut scratch, &mut probs)
            .unwrap();

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
        let model = Model::load(&blob).unwrap();

        let window: [Sample; WINDOW_LEN] =
            core::array::from_fn(|t| [(t as i16 % 50) - 25, 10, -((t as i16) % 7)]);
        let feats = [0.5f32; FEATURE_LEN];
        let mut scratch = vec![0.0f32; cfg.arena_floats()];
        let mut probs = [0.0f32; N_CLASSES];
        model
            .forward(&window, &feats, &mut scratch, &mut probs)
            .unwrap();

        let sum: f32 = probs.iter().sum();
        assert!(close(sum, 1.0), "probs sum to {sum}");
        assert!(probs
            .iter()
            .all(|&p| (0.0..=1.0).contains(&p) && p.is_finite()));
    }
}
