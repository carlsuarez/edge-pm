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
//! ```
//!
//! A window CSV is one `x,y,z` sample per row, [`WINDOW_LEN`] rows. The `features` and
//! `infer` outputs are directly comparable to `tools/verify_features.py` and
//! `tools/export_model.py` on the same input — the Milestone B and C gates.

use std::process::ExitCode;

use pmcore::features::{extract, Sample, FEATURE_LEN, N_AXES, WINDOW_LEN};
use pmcore::model::{Class, Model, N_CLASSES};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["features", window] => cmd_features(window),
        ["infer", model, window] => cmd_infer(model, window),
        [] | ["-h"] | ["--help"] => {
            banner();
            return ExitCode::SUCCESS;
        }
        _ => Err("usage: host-sim [features <window.csv> | infer <model.bin> <window.csv>]".into()),
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
    let model = Model::load(&bytes).map_err(|e| format!("{model_path}: {e}"))?;

    let window = parse_window(window_path)?;
    let mut feats = [0.0f32; FEATURE_LEN];
    extract(&window, &mut feats);

    let mut scratch = vec![0.0f32; model.config().arena_floats()];
    let mut probs = [0.0f32; N_CLASSES];
    model
        .forward(&window, &feats, &mut scratch, &mut probs)
        .map_err(|e| e.to_string())?;

    let top = (0..N_CLASSES)
        .max_by(|&a, &b| probs[a].total_cmp(&probs[b]))
        .unwrap();
    println!("{}", join(&probs));
    eprintln!(
        "class={} confidence={:.4}",
        Class::from_index(top).unwrap().name(),
        probs[top]
    );
    Ok(())
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
    eprintln!("  host-sim features <window.csv>     Stage 2 — print the 9 features");
    eprintln!("  host-sim infer <model.bin> <csv>   Stages 2+3 — print class probabilities");
    eprintln!();
    eprintln!("Decision logic (alert state machine) follows in Milestone D.");
}
