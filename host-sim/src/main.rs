//! edge-pm host simulator — the host-first development harness.
//!
//! Runs the platform-agnostic [`pmcore`] pipeline on the laptop over recorded or synthetic
//! data, so feature extraction, the model forward pass, and the decision logic are all
//! validated before any board (emulated in Renode, then real) runs them. This is the same
//! code path the firmware uses; only the sample source differs.
//!
//! ```text
//! host-sim                          print the pipeline configuration
//! host-sim features <window.csv>    run Stage 2 — print the 9 features
//! host-sim infer <model.bin> <csv>  run Stages 2+3 — print the class probabilities
//! host-sim replay <model.bin> <csv> run all stages over a sample stream + alert FSM
//! ```
//!
//! A window CSV is one `x,y,z` sample per row, [`WINDOW_LEN`] rows. A stream CSV is the same
//! format but arbitrarily long — `replay` chops it into windows and drives the full Stage
//! 1→4 pipeline (windowing → features → model → alert state machine). The `features` and
//! `infer` outputs are directly comparable to `tools/verify_features.py` and
//! `tools/export_model.py`; `replay` to `tools/make_stream.py` — the Milestone B/C/D gates.

use std::process::ExitCode;

use pmcore::alert::{AlertMachine, State, ALERT_CONFIDENCE};
use pmcore::features::{extract, Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use pmcore::model::{Class, Weights, N_CLASSES};
use pmcore::pipeline::{process_window, Windower};
use pmcore::{Arena, RunState};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["features", window] => cmd_features(window),
        ["infer", model, window] => cmd_infer(model, window),
        ["replay", model, stream] => cmd_replay(model, stream, ALERT_CONFIDENCE),
        ["replay", model, stream, "--alert-confidence", c] => match c.parse::<f32>() {
            Ok(conf) => cmd_replay(model, stream, conf),
            Err(e) => Err(format!("--alert-confidence: {e}")),
        },
        [] | ["-h"] | ["--help"] => {
            banner();
            return ExitCode::SUCCESS;
        }
        _ => Err(
            "usage: host-sim [features <window.csv> | infer <model.bin> <csv> | \
                  replay <model.bin> <stream.csv> [--alert-confidence <f>]]"
                .into(),
        ),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Stage 2: print the 9-element feature vector for a window.
fn cmd_features(window_path: &str) -> Result<(), String> {
    let window = parse_window(window_path)?;
    let mut feats = [0.0f32; FEATURE_LEN];
    extract(&window, &mut feats);
    println!("{}", join(&feats));
    Ok(())
}

/// Stages 2+3: extract features, run the model, print the class probabilities.
fn cmd_infer(model_path: &str, window_path: &str) -> Result<(), String> {
    let bytes = std::fs::read(model_path).map_err(|e| format!("{model_path}: {e}"))?;
    let (config, weights) = Weights::load(&bytes).map_err(|e| format!("{model_path}: {e}"))?;

    let window = parse_window(window_path)?;

    // Carve the run state once, exactly as the firmware does at startup.
    let mut scratch = vec![0.0f32; config.arena_floats()];
    let mut arena = Arena::new(&mut scratch);
    let mut state = RunState::new(&mut arena, &config).map_err(|e| format!("{e:?}"))?;

    // Reuse the shared per-window step; the alert machine is a throwaway here (this command
    // reports a single window's probabilities, not a running alert state).
    let mut alert = AlertMachine::new();
    let out = process_window(&config, &weights, &window, &mut state, &mut alert);

    println!("{}", join(&out.probs));
    eprintln!(
        "class={} confidence={:.4}",
        out.class.name(),
        out.confidence
    );
    Ok(())
}

/// All stages: chop a sample stream into windows and drive the full pipeline + alert FSM.
fn cmd_replay(model_path: &str, stream_path: &str, confidence: f32) -> Result<(), String> {
    let bytes = std::fs::read(model_path).map_err(|e| format!("{model_path}: {e}"))?;
    let (config, weights) = Weights::load(&bytes).map_err(|e| format!("{model_path}: {e}"))?;
    let samples = parse_stream(stream_path)?;

    let mut windower = Windower::new();
    let mut alert = AlertMachine::with_confidence(confidence);

    // Carve the run state once up front; every window reuses it (no per-window allocation),
    // mirroring how the firmware sets up its static arena at boot and loops forever.
    let mut scratch = vec![0.0f32; config.arena_floats()];
    let mut arena = Arena::new(&mut scratch);
    let mut state = RunState::new(&mut arena, &config).map_err(|e| format!("{e:?}"))?;

    eprintln!(
        "replay: {} samples, alert-confidence {:.2}",
        samples.len(),
        confidence
    );
    let mut windows = 0usize;
    let mut alerts = 0usize;
    let mut prev = State::Normal;

    for s in &samples {
        let Some(window) = windower.push(*s) else {
            continue;
        };
        let out = process_window(&config, &weights, window, &mut state, &mut alert);
        println!(
            "win {:>3}  {}  class={:<14} conf={:.4}  state={}",
            windows,
            join(&out.probs),
            out.class.name(),
            out.confidence,
            fmt_state(out.state),
        );
        if out.state != prev {
            eprintln!(
                "  window {windows}: {} -> {}",
                fmt_state(prev),
                fmt_state(out.state)
            );
            if matches!(out.state, State::Alert { .. }) {
                alerts += 1;
            }
        }
        prev = out.state;
        windows += 1;
    }

    let leftover = windower.fill_level();
    eprintln!(
        "replay done: {windows} windows, {alerts} alert(s) raised, final state {}, {leftover} trailing samples dropped",
        fmt_state(prev)
    );
    Ok(())
}

/// Render an alert state for the human-readable log.
fn fmt_state(s: State) -> String {
    match s {
        State::Normal => "NORMAL".to_string(),
        State::Alert { class } => format!("ALERT:{}", class.name()),
    }
}

/// Parse an arbitrarily long stream CSV (`x,y,z` per row) into a flat sample vector.
fn parse_stream(path: &str) -> Result<Vec<Sample>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut samples = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut cols = line.split(',');
        let mut s = [0i16; N_AXES];
        for axis in s.iter_mut() {
            let tok = cols
                .next()
                .ok_or_else(|| format!("line {}: expected {N_AXES} columns", n + 1))?;
            *axis = tok
                .trim()
                .parse::<i16>()
                .map_err(|e| format!("line {}: {e}", n + 1))?;
        }
        samples.push(s);
    }
    Ok(samples)
}

/// Parse a window CSV (`x,y,z` per row, [`WINDOW_LEN`] rows) into a sample array.
fn parse_window(path: &str) -> Result<[Sample; WINDOW_LEN], String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;

    let mut window = [[0i16; N_AXES]; WINDOW_LEN];
    let mut rows = 0usize;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if rows >= WINDOW_LEN {
            return Err(format!("more than {WINDOW_LEN} sample rows"));
        }
        let mut cols = line.split(',');
        for axis in window[rows].iter_mut() {
            let tok = cols
                .next()
                .ok_or_else(|| format!("line {}: expected {N_AXES} columns", n + 1))?;
            *axis = tok
                .trim()
                .parse::<i16>()
                .map_err(|e| format!("line {}: {e}", n + 1))?;
        }
        rows += 1;
    }
    if rows != WINDOW_LEN {
        return Err(format!("expected {WINDOW_LEN} sample rows, got {rows}"));
    }
    Ok(window)
}

/// Space-separated, 6 decimals — matches the Python tools' output format.
fn join(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.6}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn banner() {
    eprintln!("edge-pm host simulator (host-first replay harness)");
    eprintln!(
        "  window:   {WINDOW_LEN} samples x {N_AXES} axes  ->  {FEATURE_LEN} features  ->  {N_CLASSES} classes"
    );
    eprint!("  classes:  ");
    for i in 0..N_CLASSES {
        eprint!("{i}={} ", Class::from_index(i).unwrap().name());
    }
    eprintln!();
    eprintln!();
    eprintln!("  host-sim features <window.csv>      Stage 2 — print the 9 features");
    eprintln!("  host-sim infer <model.bin> <csv>    Stages 2+3 — print class probabilities");
    eprintln!("  host-sim replay <model.bin> <csv>   Stages 1-4 — windowing + alert state machine");
    eprintln!();
    eprintln!(
        "  Alert: a fault class above {ALERT_CONFIDENCE:.2} latches ALERT; it clears after 3 consecutive"
    );
    eprintln!("  normal windows (hysteresis). Override the threshold with --alert-confidence <f>.");
}
