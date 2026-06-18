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

/// On-disk format version for fp32 weights ([`Weights`]).
const VERSION_F32: i32 = 1;

/// On-disk format version for int8 W8A8 quantized weights ([`QuantizedWeights`]).
const VERSION_Q8: i32 = 2;

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
/// needed to size them ([`l1`](Self::l1) / [`l2`](Self::l2) / [`buf_len`](Self::buf_len)).
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

    /// Elements a forward pass carves from its [`Arena`](engine::Arena) — size a static arena
    /// with this. Counts elements, not bytes, so it sizes **both** the `f32` arena of the
    /// float path (`[f32; buf_len]`) and the `i8` working set of the integer-only path
    /// (`[i8; buf_len]`): the de-interleaved window + both conv outputs + the dense input.
    /// The dense logits are a forward-pass local (their width differs by path), not carved here.
    pub const fn buf_len(&self) -> usize {
        N_AXES * WINDOW_LEN          // de-interleaved input
            + self.c1 * self.l1()    // conv1 output
            + self.c2 * self.l2()    // conv2 output
            + self.c2 + FEATURE_LEN // pooled + fused features (dense input)
    }

    /// Total `f32` weights the on-disk format carries for this config.
    const fn weight_floats(&self) -> usize {
        self.c1 * N_AXES * self.k1 + self.c1            // conv1 weight + bias
            + self.c2 * self.c1 * self.k2 + self.c2     // conv2 weight + bias
            + N_CLASSES * (self.c2 + FEATURE_LEN) + N_CLASSES // dense weight + bias
    }

    /// `i8` weight values an integer-only (v2) checkpoint carries — the three weight tensors,
    /// quantized. See [`QuantizedWeights`].
    const fn q_weight_i8s(&self) -> usize {
        self.c1 * N_AXES * self.k1                    // conv1 weight
            + self.c2 * self.c1 * self.k2             // conv2 weight
            + N_CLASSES * (self.c2 + FEATURE_LEN) // dense weight
    }

    /// `i32` values an integer-only checkpoint carries: per-output-channel `(bias, mult,
    /// shift)` for conv1 and conv2, plus the per-tensor pool `(mult, shift)`.
    const fn q_i32_count(&self) -> usize {
        3 * self.c1            // conv1 bias + mult + shift
            + 3 * self.c2      // conv2 bias + mult + shift
            + 2 // pool mult + shift
    }

    /// `f32` values an integer-only checkpoint carries: the two input-boundary activation
    /// scales (`s_in0` for the window, `s_fc_in` for the fused features), the per-class output
    /// dequant scale, and the per-class dense **bias** (`f32`, added at the final dequant — so
    /// it stays exact even when a weight row is all-zero) — `2 + 2 * N_CLASSES`.
    const fn q_f32_floats(&self) -> usize {
        2 + 2 * N_CLASSES
    }

    /// Peek the on-disk format version (`1` = fp32 [`Weights`], `2` = int8
    /// [`QuantizedWeights`]) from the header, without parsing the rest.
    ///
    /// # Errors
    /// [`ModelError::TooShort`] or [`ModelError::BadMagic`].
    pub fn version(bytes: &[u8]) -> Result<i32, ModelError> {
        if bytes.len() < HEADER_BYTES {
            return Err(ModelError::TooShort);
        }
        if bytes[..4] != MAGIC {
            return Err(ModelError::BadMagic);
        }
        Ok(rd_i32(bytes, 4))
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
        if version != VERSION_F32 && version != VERSION_Q8 {
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
        let version = rd_i32(bytes, 4);
        if version != VERSION_F32 {
            return Err(ModelError::UnsupportedVersion(version));
        }

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
/// (a softmax over the four [`Class`]es), written into `probs`.
///
/// Computes the result from the independent pieces — the [`ModelConfig`], the [`Weights`],
/// and the [`RunState`] scratch — touching no memory beyond `state`'s arena-carved buffers
/// and `probs`, so it allocates nothing and is infallible. The dense logits live in `probs`
/// itself (the closing softmax normalizes them in place), mirroring [`forward_q8`].
pub fn forward(
    config: &ModelConfig,
    weights: &Weights,
    state: &mut RunState,
    window: &[Sample; WINDOW_LEN],
    features: &[f32; FEATURE_LEN],
    probs: &mut [f32; N_CLASSES],
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

    // global average pool → concat the fused features (into the dense-input buffer)
    let (pooled, feat_tail) = state.fc_in.split_at_mut(config.c2);
    nn::global_avg_pool(pooled, state.x_c2, config.c2, config.l2());
    feat_tail.copy_from_slice(features);

    // dense + bias → logits (in `probs`) → softmax in place
    math::matmul(
        probs,
        state.fc_in,
        weights.fc_w,
        config.c2 + FEATURE_LEN,
        N_CLASSES,
    );
    math::add_bias(probs, weights.fc_b);
    math::softmax(probs);
}

/// Zero-copy views of an **integer-only** (static) int8 model — the quantized twin of
/// [`Weights`].
///
/// This is true integer-only inference (CMSIS-NN / TFLite style), **not** dynamic
/// quantization: weights *and* activations are int8, accumulation is `i32`, and the
/// per-output-channel `i32` accumulator is rescaled to the next layer's int8 domain by a
/// **fixed-point** multiplier (`mult`, `shift`) — no float in the hot path. The activation
/// scales (`s_in0`, `s_fc_in`) and the per-class output scale are baked at export from a
/// calibration pass; [`forward_q8`] never recomputes a scale. Each conv's `bias` is `i32`,
/// pre-scaled to its accumulator domain. Loaded from the v2 on-disk format — see
/// [`QuantizedWeights::load`].
///
/// The only floats touched anywhere near the model are the input/output boundaries: the
/// window is quantized with `s_in0`, the fused features with `s_fc_in`, and the four final
/// `i32` logits are dequantized with `fc_out_scale` for the closing softmax.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedWeights<'w> {
    /// conv1 int8 weight, `[c1, N_AXES, k1]`.
    pub conv1_w: &'w [i8],
    /// conv1 `i32` bias `[c1]`, pre-scaled to the `s_in0 · s_w1[o]` accumulator domain.
    pub conv1_bias: &'w [i32],
    /// conv1 per-output-channel requant multiplier `[c1]` (Q31 mantissa).
    pub conv1_mult: &'w [i32],
    /// conv1 per-output-channel requant shift `[c1]` (signed binary exponent).
    pub conv1_shift: &'w [i32],
    /// conv2 int8 weight, `[c2, c1, k2]`.
    pub conv2_w: &'w [i8],
    /// conv2 `i32` bias `[c2]`.
    pub conv2_bias: &'w [i32],
    /// conv2 per-output-channel requant multiplier `[c2]`.
    pub conv2_mult: &'w [i32],
    /// conv2 per-output-channel requant shift `[c2]`.
    pub conv2_shift: &'w [i32],
    /// Global-average-pool requant multiplier (per-tensor; folds in the `1/l2` average and
    /// the `s_c2 → s_fc_in` rescale).
    pub pool_mult: i32,
    /// Global-average-pool requant shift (per-tensor).
    pub pool_shift: i32,
    /// dense int8 weight, `[N_CLASSES, c2 + FEATURE_LEN]`.
    pub fc_w: &'w [i8],
    /// dense `f32` bias `[N_CLASSES]`, added at the final dequant (kept `f32` so it stays
    /// exact even for an all-zero weight row, where the `s_fc_in · s_wfc[o]` domain collapses).
    pub fc_bias: &'w [f32],
    /// Per-class dequant scale `[N_CLASSES]` (`s_fc_in · s_wfc[o]`): turns the final `i32`
    /// weight-dot logits back into real values for the softmax.
    pub fc_out_scale: &'w [f32],
    /// Activation scale for quantizing the raw `i16` window to int8 (conv1 input).
    pub s_in0: f32,
    /// Activation scale for quantizing the fused `f32` features to int8 (dense input).
    pub s_fc_in: f32,
}

impl<'w> QuantizedWeights<'w> {
    /// Parse a v2 (integer-only int8) model from its on-disk bytes.
    ///
    /// Layout, after the shared 64-byte header (`version = 2`):
    /// 1. a 4-aligned `f32` block — `s_in0`, `s_fc_in`, `fc_out_scale[N_CLASSES]`,
    ///    `fc_bias[N_CLASSES]`;
    /// 2. a 4-aligned `i32` block — `conv1_bias/mult/shift` (each `[c1]`), `conv2_bias/mult/
    ///    shift` (each `[c2]`), `pool_mult`, `pool_shift`;
    /// 3. a trailing `i8` block — `conv1_w`, `conv2_w`, `fc_w`.
    ///
    /// The `f32` and `i32` blocks lead so both stay 4-aligned for the [`bytemuck`] cast; the
    /// `i8` block has no alignment requirement.
    ///
    /// # Errors
    /// Header errors from [`ModelConfig::parse`], [`ModelError::UnsupportedVersion`] if not v2,
    /// [`ModelError::SizeMismatch`] on a wrong length, or [`ModelError::Misaligned`].
    pub fn load(bytes: &'w [u8]) -> Result<(ModelConfig, QuantizedWeights<'w>), ModelError> {
        let config = ModelConfig::parse(bytes)?;
        let version = rd_i32(bytes, 4);
        if version != VERSION_Q8 {
            return Err(ModelError::UnsupportedVersion(version));
        }

        let f32_floats = config.q_f32_floats();
        let i32_count = config.q_i32_count();
        let i8_count = config.q_weight_i8s();
        let f32_end = HEADER_BYTES + f32_floats * 4;
        let i32_end = f32_end + i32_count * 4;
        let expected = i32_end + i8_count;
        if bytes.len() != expected {
            return Err(ModelError::SizeMismatch {
                expected,
                got: bytes.len(),
            });
        }

        let f32_block: &[f32] = bytemuck::try_cast_slice(&bytes[HEADER_BYTES..f32_end])
            .map_err(|_| ModelError::Misaligned)?;
        let i32_block: &[i32] = bytemuck::try_cast_slice(&bytes[f32_end..i32_end])
            .map_err(|_| ModelError::Misaligned)?;
        // u8→i8 is a 1-byte reinterpret with no alignment constraint, so this never fails.
        let i8_block: &[i8] = bytemuck::cast_slice(&bytes[i32_end..]);

        let (c1, c2) = (config.c1, config.c2);

        let mut fo = 0;
        let mut take_f = |n: usize| {
            let s = &f32_block[fo..fo + n];
            fo += n;
            s
        };
        let s_in0 = take_f(1)[0];
        let s_fc_in = take_f(1)[0];
        let fc_out_scale = take_f(N_CLASSES);
        let fc_bias = take_f(N_CLASSES);

        let mut io = 0;
        let mut take_i32 = |n: usize| {
            let s = &i32_block[io..io + n];
            io += n;
            s
        };
        let conv1_bias = take_i32(c1);
        let conv1_mult = take_i32(c1);
        let conv1_shift = take_i32(c1);
        let conv2_bias = take_i32(c2);
        let conv2_mult = take_i32(c2);
        let conv2_shift = take_i32(c2);
        let pool_mult = take_i32(1)[0];
        let pool_shift = take_i32(1)[0];

        let mut wo = 0;
        let mut take_w = |n: usize| {
            let s = &i8_block[wo..wo + n];
            wo += n;
            s
        };
        let conv1_w = take_w(c1 * N_AXES * config.k1);
        let conv2_w = take_w(c2 * c1 * config.k2);
        let fc_w = take_w(N_CLASSES * (c2 + FEATURE_LEN));

        Ok((
            config,
            QuantizedWeights {
                conv1_w,
                conv1_bias,
                conv1_mult,
                conv1_shift,
                conv2_w,
                conv2_bias,
                conv2_mult,
                conv2_shift,
                pool_mult,
                pool_shift,
                fc_w,
                fc_bias,
                fc_out_scale,
                s_in0,
                s_fc_in,
            },
        ))
    }
}

/// A loaded model's weights in whichever representation the checkpoint carried.
///
/// Mirrors tiny-infer's `ModelWeights`: [`ModelWeights::load`] dispatches on the on-disk
/// version so a caller can accept either an fp32 (v1) or an integer-only int8 (v2) checkpoint
/// and pick the matching pipeline step ([`process_window`](crate::pipeline::process_window) vs
/// [`process_window_q8`](crate::pipeline::process_window_q8)).
#[derive(Clone, Copy, Debug)]
pub enum ModelWeights<'w> {
    /// fp32 weights (v1).
    F32(Weights<'w>),
    /// integer-only int8 weights (v2).
    Q8(QuantizedWeights<'w>),
}

impl<'w> ModelWeights<'w> {
    /// Load a checkpoint, choosing the representation by its header version (`1` → fp32,
    /// `2` → integer-only int8).
    ///
    /// # Errors
    /// Whatever [`Weights::load`] / [`QuantizedWeights::load`] return, or
    /// [`ModelError::UnsupportedVersion`] for any other version.
    pub fn load(bytes: &'w [u8]) -> Result<(ModelConfig, ModelWeights<'w>), ModelError> {
        match ModelConfig::version(bytes)? {
            VERSION_F32 => {
                let (cfg, w) = Weights::load(bytes)?;
                Ok((cfg, ModelWeights::F32(w)))
            }
            VERSION_Q8 => {
                let (cfg, w) = QuantizedWeights::load(bytes)?;
                Ok((cfg, ModelWeights::Q8(w)))
            }
            v => Err(ModelError::UnsupportedVersion(v)),
        }
    }
}

/// Symmetric int8 quantize of one value at a fixed scale: `round(v / scale)`, clamped to
/// `[-127, 127]` (`0` if `scale == 0`). Used only at the two input boundaries (the window
/// and the fused features); everything inside the forward pass is already int8.
fn quantize_to_i8(v: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        0
    } else {
        libm::roundf(v / scale).clamp(-127.0, 127.0) as i8
    }
}

/// Run the **integer-only** (static int8) forward pass — the quantized twin of [`forward`].
///
/// Same pipeline and output (softmax probabilities written into `probs`), but the whole CNN
/// runs in integer arithmetic: the window is quantized to int8 (`s_in0`), both convs and the
/// pool accumulate in `i32` and requantize with the baked fixed-point multipliers (the convs'
/// `ReLU` is fused via `out_min = 0`), the fused features are quantized to int8 (`s_fc_in`),
/// and the dense head produces `i32` logits that are dequantized with `fc_out_scale` for the
/// closing softmax. **No float between layers** — the int8 activations never leave the integer
/// domain until that final dequantize.
///
/// `state` is the caller-owned int8 working set ([`RunState<i8>`](RunState)), carved once via
/// [`RunState::new`] and
/// reused for every window. Allocates nothing.
pub fn forward_q8(
    config: &ModelConfig,
    weights: &QuantizedWeights,
    state: &mut RunState<i8>,
    window: &[Sample; WINDOW_LEN],
    features: &[f32; FEATURE_LEN],
    probs: &mut [f32; N_CLASSES],
) {
    // De-interleave + quantize the raw i16 window to int8 (conv1 input, scale s_in0).
    for ch in 0..N_AXES {
        let row = &mut state.x[ch * WINDOW_LEN..][..WINDOW_LEN];
        for (slot, sample) in row.iter_mut().zip(window.iter()) {
            *slot = quantize_to_i8(sample[ch] as f32, weights.s_in0);
        }
    }

    // conv1 → fused ReLU (int8 in, int8 out)
    nn::conv1d_i8(
        state.x_c1,
        state.x,
        weights.conv1_w,
        weights.conv1_bias,
        weights.conv1_mult,
        weights.conv1_shift,
        0,
        127,
        N_AXES,
        config.c1,
        WINDOW_LEN,
        config.k1,
        config.s1,
    );

    // conv2 → fused ReLU
    nn::conv1d_i8(
        state.x_c2,
        state.x_c1,
        weights.conv2_w,
        weights.conv2_bias,
        weights.conv2_mult,
        weights.conv2_shift,
        0,
        127,
        config.c1,
        config.c2,
        config.l1(),
        config.k2,
        config.s2,
    );

    // global average pool (int8 → int8 at s_fc_in) into the head of the dense input, then
    // quantize the fused features into the tail at the same scale. The two writes touch
    // disjoint halves of `fc_in` in turn, so no split is needed.
    nn::global_avg_pool_i8(
        &mut state.fc_in[..config.c2],
        state.x_c2,
        config.c2,
        config.l2(),
        weights.pool_mult,
        weights.pool_shift,
        -127,
        127,
    );
    for (slot, &f) in state.fc_in[config.c2..].iter_mut().zip(features) {
        *slot = quantize_to_i8(f, weights.s_fc_in);
    }

    // dense → i32 weight-dot logits → dequantize per class and add the f32 bias → softmax.
    let mut logits = [0i32; N_CLASSES];
    nn::matmul_i8(
        &mut logits,
        state.fc_in,
        weights.fc_w,
        config.c2 + FEATURE_LEN,
        N_CLASSES,
    );
    for o in 0..N_CLASSES {
        probs[o] = logits[o] as f32 * weights.fc_out_scale[o] + weights.fc_bias[o];
    }
    math::softmax(probs);
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
        put(&mut b, 4, VERSION_F32);
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
        let mut scratch = vec![0.0f32; cfg.buf_len()];
        let mut arena = Arena::new(&mut scratch);
        let mut state = RunState::new(&mut arena, cfg).unwrap();
        let mut probs = [0.0f32; N_CLASSES];
        forward(cfg, w, &mut state, window, feats, &mut probs);
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
        // x + x_c1 + x_c2 + fc_in (no logits buffer: probs are a caller-owned out-param).
        let expect = 3 * 512 + 16 * 253 + 32 * 125 + (32 + 9);
        assert_eq!(cfg.buf_len(), expect);
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

    #[test]
    fn forward_q8_tracks_fp32() {
        // The integer-only int8 forward pass must reproduce the fp32 pass up to quantization
        // error: same argmax, probabilities within a small tolerance. This reimplements the
        // export's static quantization in-Rust — per-output-channel weight scales, activation
        // scales *calibrated from the fp32 activations*, i32 biases, and fixed-point requant
        // multipliers — then runs both `forward` and `forward_q8` on the same window.
        let cfg = ModelConfig {
            c1: 8,
            k1: 5,
            s1: 3,
            c2: 8,
            k2: 3,
            s2: 2,
        };
        let wf: Vec<f32> = (0..cfg.weight_floats())
            .map(|i| libm::sinf(i as f32 * 0.123) * 0.1)
            .collect();
        let blob = build_blob(&cfg, &wf);
        let (config, weights) = Weights::load(&blob).unwrap();

        let window: [Sample; WINDOW_LEN] =
            core::array::from_fn(|t| [(t as i16 % 50) - 25, 10, -((t as i16) % 7)]);
        let feats = [0.5f32; FEATURE_LEN];

        // fp32 reference; the run leaves the intermediate activations in the state buffers,
        // which is our one-window "calibration set" for the activation scales.
        let mut scratch = vec![0.0f32; config.buf_len()];
        let mut arena = Arena::new(&mut scratch);
        let mut state = RunState::new(&mut arena, &config).unwrap();
        let mut probs_f = [0.0f32; N_CLASSES];
        forward(&config, &weights, &mut state, &window, &feats, &mut probs_f);

        // Activation scales (per-tensor) calibrated from the captured fp32 activations.
        let win_abs = window
            .iter()
            .flatten()
            .fold(0i16, |m, &v| m.max(v.abs())) as f32;
        let s_in0 = win_abs / 127.0;
        let s_c1 = max_abs(state.x_c1) / 127.0;
        let s_c2 = max_abs(state.x_c2) / 127.0;
        let s_fc_in = max_abs(state.fc_in) / 127.0;

        // Per-output-channel weight quantization.
        let (c1w, s_w1) = quantize_rows(weights.conv1_w, config.c1, N_AXES * config.k1);
        let (c2w, s_w2) = quantize_rows(weights.conv2_w, config.c2, config.c1 * config.k2);
        let (fcw, s_wfc) = quantize_rows(weights.fc_w, N_CLASSES, config.c2 + FEATURE_LEN);

        // i32 biases (pre-scaled to each layer's accumulator domain) + requant multipliers.
        let mut c1_bias = vec![0i32; config.c1];
        let mut c1_mult = vec![0i32; config.c1];
        let mut c1_shift = vec![0i32; config.c1];
        for o in 0..config.c1 {
            c1_bias[o] = libm::roundf(weights.conv1_b[o] / (s_in0 * s_w1[o])) as i32;
            let (m, s) = quantize_multiplier(s_in0 * s_w1[o] / s_c1);
            c1_mult[o] = m;
            c1_shift[o] = s;
        }
        let mut c2_bias = vec![0i32; config.c2];
        let mut c2_mult = vec![0i32; config.c2];
        let mut c2_shift = vec![0i32; config.c2];
        for o in 0..config.c2 {
            c2_bias[o] = libm::roundf(weights.conv2_b[o] / (s_c1 * s_w2[o])) as i32;
            let (m, s) = quantize_multiplier(s_c1 * s_w2[o] / s_c2);
            c2_mult[o] = m;
            c2_shift[o] = s;
        }
        let (pool_mult, pool_shift) = quantize_multiplier(s_c2 / (config.l2() as f32 * s_fc_in));
        // The dense bias stays f32 and is added *after* the per-class dequant, so it survives an
        // all-zero weight row (where the `s_fc_in · s_wfc[o]` accumulator domain collapses).
        let fc_bias: Vec<f32> = weights.fc_b.to_vec();
        let fc_out_scale: Vec<f32> = (0..N_CLASSES).map(|o| s_fc_in * s_wfc[o]).collect();

        let qw = QuantizedWeights {
            conv1_w: &c1w,
            conv1_bias: &c1_bias,
            conv1_mult: &c1_mult,
            conv1_shift: &c1_shift,
            conv2_w: &c2w,
            conv2_bias: &c2_bias,
            conv2_mult: &c2_mult,
            conv2_shift: &c2_shift,
            pool_mult,
            pool_shift,
            fc_w: &fcw,
            fc_bias: &fc_bias,
            fc_out_scale: &fc_out_scale,
            s_in0,
            s_fc_in,
        };

        // The int8 forward runs over its own int8 working set, carved once like the firmware.
        let mut ibuf = vec![0i8; config.buf_len()];
        let mut iarena = Arena::new(&mut ibuf);
        let mut istate = RunState::new(&mut iarena, &config).unwrap();
        let mut probs_q = [0.0f32; N_CLASSES];
        forward_q8(&config, &qw, &mut istate, &window, &feats, &mut probs_q);

        let sum: f32 = probs_q.iter().sum();
        assert!(close(sum, 1.0), "int8 probs sum to {sum}");
        assert_eq!(
            argmax(&probs_f),
            argmax(&probs_q),
            "int8 changed the argmax: fp32 {probs_f:?} vs int8 {probs_q:?}"
        );
        let max_abs_diff = probs_f
            .iter()
            .zip(&probs_q)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 0.05,
            "int8 diverges from fp32: max abs {max_abs_diff} (fp32 {probs_f:?} vs int8 {probs_q:?})"
        );
    }

    fn argmax(p: &[f32; N_CLASSES]) -> usize {
        p.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    fn max_abs(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, &v| m.max(libm::fabsf(v)))
    }

    // Reference QuantizeMultiplier (offline): real M>0 → (Q31 mantissa, signed shift).
    fn quantize_multiplier(m: f32) -> (i32, i32) {
        if m == 0.0 {
            return (0, 0);
        }
        let exp = libm::floorf(libm::log2f(m)) as i32 + 1;
        let frac = (m as f64) / 2f64.powi(exp);
        let mut q = (frac * (1i64 << 31) as f64).round() as i64;
        let mut shift = exp;
        if q == (1i64 << 31) {
            q /= 2;
            shift += 1;
        }
        (q as i32, shift)
    }

    // Quantize one weight tensor per output row (group_size = cols): int8 values + per-row scale.
    fn quantize_rows(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
        let mut q = vec![0i8; w.len()];
        let mut scales = vec![0.0f32; rows];
        for o in 0..rows {
            let row = &w[o * cols..][..cols];
            let scale = max_abs(row) / 127.0;
            scales[o] = scale;
            for j in 0..cols {
                q[o * cols + j] = quantize_to_i8(row[j], scale);
            }
        }
        (q, scales)
    }
}
