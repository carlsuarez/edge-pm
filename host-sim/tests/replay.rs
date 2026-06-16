//! Milestone D gate — the full replay pipeline + alert FSM vs an independent reference.
//!
//! Drives `pmcore`'s [`Windower`] + [`process_window`] + [`AlertMachine`] over a recorded
//! sample stream and checks two things against fixtures from `tools/make_stream.py`:
//! per-window softmax probs match PyTorch (the forward pass, now over a multi-window stream
//! rather than the single Milestone C window), and the `NORMAL ⇄ ALERT` trajectory matches
//! the pure-Python `fsm_run` mirror step for step. It also asserts the run actually latches
//! an alert and later clears it, so the hysteresis path is genuinely exercised end-to-end
//! (not just in the in-crate unit tests).
//!
//! Fixtures under `models/` (gitignored, so this skips on a clean checkout):
//! `bearing_stream.bin` / `.csv` / `.ref.bin`. Regenerate with:
//!   python tools/make_stream.py --out models/bearing_stream

use std::path::PathBuf;

use pmcore::alert::{AlertMachine, State};
use pmcore::features::{Sample, N_AXES};
use pmcore::model::{Class, Weights, N_CLASSES};
use pmcore::pipeline::{process_window, Windower};
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

#[test]
fn replay_matches_reference_and_latches_then_clears() {
    let dir = models_dir();
    let model_p = dir.join("bearing_stream.bin");
    let stream_p = dir.join("bearing_stream.csv");
    let ref_p = dir.join("bearing_stream.ref.bin");
    if !(model_p.exists() && stream_p.exists() && ref_p.exists()) {
        eprintln!("skipping: stream fixtures not present (run tools/make_stream.py)");
        return;
    }

    let bytes = std::fs::read(&model_p).unwrap();
    let (config, weights) = Weights::load(&bytes).expect("model loads");

    // Parse the stream CSV into a flat sample vector.
    let text = std::fs::read_to_string(&stream_p).unwrap();
    let mut samples: Vec<Sample> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split(',');
        let mut s = [0i16; N_AXES];
        for axis in s.iter_mut() {
            *axis = cols.next().unwrap().trim().parse().unwrap();
        }
        samples.push(s);
    }

    // Reference: i32 n_windows, i32 n_classes, f32 threshold, then per-window probs, then
    // the per-window (i32 state, i32 latched) FSM trajectory.
    let r = std::fs::read(&ref_p).unwrap();
    let n = rd_i32(&r, 0) as usize;
    assert_eq!(rd_i32(&r, 4) as usize, N_CLASSES);
    let thr = rd_f32(&r, 8);
    let probs_off = 12;
    let track_off = probs_off + n * N_CLASSES * 4;
    let ref_probs = |w: usize, c: usize| rd_f32(&r, probs_off + (w * N_CLASSES + c) * 4);
    let ref_state = |w: usize| rd_i32(&r, track_off + w * 8);
    let ref_latched = |w: usize| rd_i32(&r, track_off + w * 8 + 4);

    // Drive the real pipeline at the reference threshold. The run state is carved once and
    // reused for every window, exactly as the firmware loop does.
    let mut windower = Windower::new();
    let mut alert = AlertMachine::with_confidence(thr);
    let mut scratch = vec![0.0f32; config.arena_floats()];
    let mut arena = Arena::new(&mut scratch);
    let mut state = RunState::new(&mut arena, &config).unwrap();

    let mut w = 0usize;
    let mut max_abs = 0.0f32;
    let mut prev = State::Normal;
    let mut latched_once = false;
    let mut cleared_once = false;

    for s in &samples {
        let Some(window) = windower.push(*s) else {
            continue;
        };
        let out = process_window(&config, &weights, window, &mut state, &mut alert);

        // Probs vs PyTorch.
        for c in 0..N_CLASSES {
            let d = (out.probs[c] - ref_probs(w, c)).abs();
            max_abs = max_abs.max(d);
            assert!(d < 1e-3, "win {w} class {c}: {} vs {} (abs {d:.2e})", out.probs[c], ref_probs(w, c));
        }

        // State vs the Python FSM mirror.
        match out.state {
            State::Normal => assert_eq!(ref_state(w), 0, "win {w}: expected Normal"),
            State::Alert { class } => {
                assert_eq!(ref_state(w), 1, "win {w}: expected Alert");
                let exp = Class::from_index(ref_latched(w) as usize).unwrap();
                assert_eq!(class, exp, "win {w}: latched class mismatch");
            }
        }

        // Track that a latch and a clear both occur.
        match (prev, out.state) {
            (State::Normal, State::Alert { .. }) => latched_once = true,
            (State::Alert { .. }, State::Normal) => cleared_once = true,
            _ => {}
        }
        prev = out.state;
        w += 1;
    }

    assert_eq!(w, n, "replayed {w} windows, reference has {n}");
    assert!(latched_once, "expected the run to latch an alert");
    assert!(cleared_once, "expected the alert to clear via hysteresis");
    eprintln!(
        "replay matches reference over {n} windows (max abs prob diff {max_abs:.2e}); latch+clear exercised"
    );
}
