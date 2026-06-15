//! Stage 1 glue — windowing and the sample-buffer handoff, platform-agnostic parts.
//!
//! On-device, the ADXL345 streams 3-axis samples into a static ring buffer via DMA, and
//! the CPU wakes on a window-complete interrupt once [`WINDOW_LEN`](crate::features::WINDOW_LEN)
//! samples have accumulated. The DMA/interrupt wiring is firmware-specific (it lives in
//! `firmware/`), but the windowing bookkeeping — tracking fill level, presenting a
//! completed window as a `&[Sample; WINDOW_LEN]` to [`crate::features::extract`], and
//! double-buffering so acquisition continues during feature extraction — is portable and
//! lives here so it can be exercised on the host with a file-fed sample stream.
//!
//! Windowing/double-buffer logic lands in **Milestone D**; the host replay path
//! (`host-sim`) drives it with recorded CWRU data.
