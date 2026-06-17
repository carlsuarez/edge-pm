//! Milestone D gate — the full replay pipeline + alert FSM vs an independent reference.
//!
//! Chops a recorded sample stream into windows, drives `pmcore`'s [`process_window`] +
//! [`AlertMachine`] over each, and checks two things against fixtures from
//! `tools/make_stream.py`: per-window softmax probs match PyTorch, and the
//! `NORMAL ⇄ ALERT` trajectory matches the pure-Python `fsm_run` mirror of `pmcore::alert`
//! window for window. It also asserts the run actually latches an alert and later clears it,
//! so the hysteresis path is genuinely exercised end-to-end (not just in the in-crate unit
//! tests).
//!
//! Fixtures under `models/` (gitignored, so this skips on a clean checkout):
//! `bearing_stream.bin` / `.csv` / `.ref.bin`. Regenerate with:
//!   python tools/make_stream.py --out models/bearing_stream

use std::path::PathBuf;

use pmcore::alert::{AlertMachine, State};
use pmcore::features::{Sample, N_AXES, WINDOW_LEN};
use pmcore::model::{Class, Weights, N_CLASSES};
use pmcore::pipeline::process_window;
use pmcore::{Arena, RunState};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("models")
}

fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn rd_f32(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// The model output index for a class — the encoding `fsm_run` records in `latched`.
fn class_index(c: Class) -> i32 {
    match c {
        Class::Normal => 0,
        Class::InnerRace => 1,
        Class::OuterRace => 2,
        Class::RollingElement => 3,
    }
}

#[test]
fn replay_matches_pytorch_probs_and_fsm_trajectory() {
    let dir = models_dir();
    let model_p = dir.join("bearing_stream.bin");
    let csv_p = dir.join("bearing_stream.csv");
    let ref_p = dir.join("bearing_stream.ref.bin");
    if !(model_p.exists() && csv_p.exists() && ref_p.exists()) {
        eprintln!("skipping: stream fixtures not present (run tools/make_stream.py)");
        return;
    }

    let bytes = std::fs::read(&model_p).unwrap();
    let (config, weights) = Weights::load(&bytes).expect("model loads");

    // Parse the stream CSV (`x,y,z` per row) into samples.
    let text = std::fs::read_to_string(&csv_p).unwrap();
    let mut samples: Vec<Sample> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split(',');
        let mut s = [0i16; N_AXES];
        for axis in s.iter_mut() {
            *axis = cols.next().unwrap().trim().parse().unwrap();
        }
        samples.push(s);
    }

    // Reference layout (little-endian): i32 n_windows, i32 n_classes, f32 thr,
    // then n_windows*n_classes f32 probs, then n_windows*(i32 state, i32 latched).
    let r = std::fs::read(&ref_p).unwrap();
    let n_windows = rd_i32(&r, 0) as usize;
    assert_eq!(rd_i32(&r, 4) as usize, N_CLASSES, "n_classes");
    let thr = rd_f32(&r, 8);
    let probs_off = 12;
    let track_off = probs_off + n_windows * N_CLASSES * 4;
    assert_eq!(
        samples.len(),
        n_windows * WINDOW_LEN,
        "stream length must be exactly {n_windows} windows"
    );

    let mut scratch = vec![0.0f32; config.arena_floats()];
    let mut arena = Arena::new(&mut scratch);
    let mut state = RunState::new(&mut arena, &config).unwrap();
    // Match the FSM the reference was generated with.
    let mut alert = AlertMachine::with_confidence(thr);

    let mut saw_alert = false;
    let mut saw_clear_after_alert = false;
    let mut was_alerting = false;
    let mut max_prob_diff = 0.0f32;

    for (w, chunk) in samples.chunks_exact(WINDOW_LEN).enumerate() {
        let window: &[Sample; WINDOW_LEN] = chunk.try_into().unwrap();
        let out = process_window(&config, &weights, window, &mut state, &mut alert);

        // 1. Per-window probs vs PyTorch.
        for (k, &p) in out.probs.iter().enumerate() {
            let rp = rd_f32(&r, probs_off + (w * N_CLASSES + k) * 4);
            let d = (p - rp).abs();
            max_prob_diff = max_prob_diff.max(d);
            assert!(
                d < 1e-3,
                "window {w} class {k}: pmcore {p} vs ref {rp} (abs {d:.2e})"
            );
        }

        // 2. FSM trajectory vs the pure-Python mirror.
        let ref_state = rd_i32(&r, track_off + w * 8);
        let ref_latched = rd_i32(&r, track_off + w * 8 + 4);
        let (got_state, got_latched) = match out.state {
            State::Normal => (0, -1),
            State::Alert { class } => (1, class_index(class)),
        };
        assert_eq!(got_state, ref_state, "window {w}: alert state");
        assert_eq!(got_latched, ref_latched, "window {w}: latched class");

        let alerting = got_state == 1;
        saw_alert |= alerting;
        saw_clear_after_alert |= was_alerting && !alerting;
        was_alerting = alerting;
    }

    assert!(saw_alert, "stream never latched an alert");
    assert!(
        saw_clear_after_alert,
        "alert never cleared (hysteresis path not exercised)"
    );
    eprintln!(
        "replay parity ok: {n_windows} windows match PyTorch probs (max abs diff {max_prob_diff:.2e}) + FSM trajectory"
    );
}
