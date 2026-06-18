//! Milestone F gate — pmcore's int8 (W8A8) forward pass vs the fp32 PyTorch reference.
//!
//! Loads the v2 quantized model `bearing_cnn_q8.bin` (written by
//! `tools/export_model.py --quantize`) and runs `process_window_q8` on the same window the
//! Milestone-C fp32 gate uses, asserting the int8 result **preserves the argmax** and stays
//! within a small tolerance of the fp32 PyTorch probabilities in `bearing_cnn.ref.bin`.
//! Quantization perturbs the probabilities but must not change the decision. If the fixtures
//! are absent the test skips itself, so the suite still passes on a clean checkout.
//!
//! Regenerate the fixtures with:
//!   python tools/export_model.py --out models/bearing_cnn.bin --quantize

use std::path::PathBuf;

use pmcore::alert::AlertMachine;
use pmcore::features::{N_AXES, WINDOW_LEN};
use pmcore::model::{ModelWeights, N_CLASSES};
use pmcore::pipeline::process_window_q8;
use pmcore::{Arena, RunState};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn rd_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
fn q8_forward_tracks_pytorch_reference() {
    let dir = models_dir();
    let model_p = dir.join("bearing_cnn_q8.bin");
    let window_p = dir.join("bearing_cnn.window.csv");
    let ref_p = dir.join("bearing_cnn.ref.bin");
    if !(model_p.exists() && window_p.exists() && ref_p.exists()) {
        eprintln!("skipping: q8 fixtures not present (run tools/export_model.py --quantize)");
        return;
    }

    // Load the v2 int8 model.
    let bytes = std::fs::read(&model_p).unwrap();
    let (config, weights) = ModelWeights::load(&bytes).expect("q8 model loads");
    let qw = match weights {
        ModelWeights::Q8(qw) => qw,
        ModelWeights::F32(_) => panic!("bearing_cnn_q8.bin should be a v2 int8 model"),
    };

    // Parse the window CSV.
    let text = std::fs::read_to_string(&window_p).unwrap();
    let mut window = [[0i16; N_AXES]; WINDOW_LEN];
    let mut rows = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split(',');
        for axis in window[rows].iter_mut() {
            *axis = cols.next().unwrap().trim().parse().unwrap();
        }
        rows += 1;
    }
    assert_eq!(rows, WINDOW_LEN, "window must have {WINDOW_LEN} rows");

    // Reference fp32 probs (u32 n_features, u32 n_classes, f32 features, f32 probs).
    let r = std::fs::read(&ref_p).unwrap();
    let n_features = rd_u32(&r, 0) as usize;
    assert_eq!(rd_u32(&r, 4) as usize, N_CLASSES);
    let probs_off = 8 + n_features * 4;
    let ref_probs: Vec<f32> = (0..N_CLASSES)
        .map(|i| rd_f32(&r, probs_off + i * 4))
        .collect();

    // Run the integer-only forward pass over its int8 working set.
    let mut ibuf = vec![0i8; config.buf_len()];
    let mut iarena = Arena::new(&mut ibuf);
    let mut istate = RunState::new(&mut iarena, &config).unwrap();
    let mut alert = AlertMachine::new();
    let out = process_window_q8(&config, &qw, &window, &mut istate, &mut alert);

    // The int8 decision must match fp32, and the probabilities must stay close.
    assert_eq!(
        argmax(&ref_probs),
        argmax(&out.probs),
        "q8 changed the argmax: fp32 {ref_probs:?} vs q8 {:?}",
        out.probs
    );
    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in out.probs.iter().zip(&ref_probs).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(d < 0.02, "class {i}: q8 {a} vs fp32 {b} (abs {d:.2e})");
    }
    let sum: f32 = out.probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "q8 probs sum to {sum}");
    eprintln!("q8 forward tracks PyTorch (max abs prob diff {max_abs:.2e})");
}
