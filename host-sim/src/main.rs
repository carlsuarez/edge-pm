//! edge-pm host simulator — the host-first development harness.
//!
//! Runs the platform-agnostic [`pmcore`] pipeline on the laptop over recorded or
//! synthetic data, so feature extraction, the model forward pass, and the decision state
//! machine are all validated before any board (emulated in Renode, then real) runs them.
//! This is the same code path the firmware uses; only the sample source differs.
//!
//! Milestone B: given a window CSV (one `x,y,z` sample per row, [`WINDOW_LEN`] rows), it
//! runs the real [`pmcore::features::extract`] and prints the 9 features — directly
//! comparable to `tools/verify_features.py` on the same file.
//!
//! Usage:
//!   `host-sim <window.csv>`   run feature extraction on a window
//!   `host-sim`                print the pipeline configuration

use std::process::ExitCode;

use pmcore::alert::ALERT_CONFIDENCE;
use pmcore::features::{extract, Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use pmcore::model::{Class, N_CLASSES};

fn main() -> ExitCode {
    match std::env::args().nth(1) {
        Some(arg) if arg != "-h" && arg != "--help" => match run_features(&arg) {
            Ok(feats) => {
                let line: Vec<String> = feats.iter().map(|v| format!("{v:.6}")).collect();
                println!("{}", line.join(" "));
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            banner();
            ExitCode::SUCCESS
        }
    }
}

/// Parse a window CSV and run the real `pmcore` feature extraction over it.
fn run_features(path: &str) -> Result<[f32; FEATURE_LEN], String> {
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

    let window: &[Sample; WINDOW_LEN] = &window;
    let mut out = [0.0f32; FEATURE_LEN];
    extract(window, &mut out);
    Ok(out)
}

fn banner() {
    eprintln!("edge-pm host simulator (host-first replay harness)");
    eprintln!(
        "  window:   {WINDOW_LEN} samples x {N_AXES} axes  ->  {FEATURE_LEN} features  ->  {N_CLASSES} classes"
    );
    eprint!("  classes:  ");
    for i in 0..N_CLASSES {
        let c = Class::from_index(i).unwrap();
        eprint!("{i}={} ", c.name());
    }
    eprintln!();
    eprintln!("  alert:    confidence > {ALERT_CONFIDENCE:.2} in any fault class");
    eprintln!();
    eprintln!("Stage 2 (features) is wired: `host-sim <window.csv>` prints the feature vector.");
    eprintln!("Model inference + decision logic follow in Milestones C-D.");
}
