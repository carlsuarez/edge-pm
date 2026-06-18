//! Milestone C gate — pmcore's forward pass vs PyTorch laptop inference.
//!
//! Runs against fixtures produced by `tools/export_model.py` under `models/`:
//! `bearing_cnn.bin` (the exported model), `bearing_cnn.window.csv` (a deterministic input
//! window), and `bearing_cnn.ref.bin` (the reference features + softmax probs PyTorch
//! produced on that window). If the fixtures are absent the test skips itself, so the suite
//! still passes on a clean checkout (same pattern as tiny-infer's parity tests).
//!
//! Regenerate the fixtures with:
//!   python tools/export_model.py --out models/bearing_cnn.bin

use std::path::PathBuf;

use pmcore::alert::AlertMachine;
use pmcore::features::{extract, FEATURE_LEN, N_AXES, WINDOW_LEN};
use pmcore::model::{Weights, N_CLASSES};
use pmcore::pipeline::process_window;
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

#[test]
fn forward_pass_matches_pytorch_reference() {
    let dir = models_dir();
    let model_p = dir.join("bearing_cnn.bin");
    let window_p = dir.join("bearing_cnn.window.csv");
    let ref_p = dir.join("bearing_cnn.ref.bin");
    if !(model_p.exists() && window_p.exists() && ref_p.exists()) {
        eprintln!("skipping: export fixtures not present (run tools/export_model.py)");
        return;
    }

    // Load the model.
    let bytes = std::fs::read(&model_p).unwrap();
    let (config, weights) = Weights::load(&bytes).expect("model loads");

    // Parse the window CSV into a sample array.
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

    // Reference: features + probs PyTorch produced (u32 n_features, u32 n_classes, f32s).
    let r = std::fs::read(&ref_p).unwrap();
    assert_eq!(rd_u32(&r, 0) as usize, FEATURE_LEN);
    assert_eq!(rd_u32(&r, 4) as usize, N_CLASSES);
    let ref_feats: Vec<f32> = (0..FEATURE_LEN).map(|i| rd_f32(&r, 8 + i * 4)).collect();
    let probs_off = 8 + FEATURE_LEN * 4;
    let ref_probs: Vec<f32> = (0..N_CLASSES)
        .map(|i| rd_f32(&r, probs_off + i * 4))
        .collect();

    // pmcore: extract features, then run the model.
    let mut feats = [0.0f32; FEATURE_LEN];
    extract(&window, &mut feats);

    // The Milestone B features must already agree with the numpy ones in the reference.
    for (i, (&a, &b)) in feats.iter().zip(&ref_feats).enumerate() {
        let rel = (a - b).abs() / b.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "feature {i}: pmcore {a} vs ref {b} (rel {rel:.2e})"
        );
    }

    let mut scratch = vec![0.0f32; config.buf_len()];
    let mut arena = Arena::new(&mut scratch);
    let mut state = RunState::new(&mut arena, &config).unwrap();
    let mut alert = AlertMachine::new();
    let probs = process_window(&config, &weights, &window, &mut state, &mut alert).probs;

    // End-to-end: pmcore probs vs PyTorch probs.
    let mut max_abs = 0.0f32;
    for (i, (&a, &b)) in probs.iter().zip(&ref_probs).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(
            d < 1e-3,
            "class {i}: pmcore {a} vs pytorch {b} (abs {d:.2e})"
        );
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "probs sum to {sum}");
    eprintln!("forward pass matches PyTorch (max abs prob diff {max_abs:.2e})");
}
