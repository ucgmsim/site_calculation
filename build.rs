use csv::Reader;
use std::fs;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
struct BA18Record {
    frequency: f64,
    f3: f64,
    f4: f64,
    f5: f64,
    c8: Option<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct CB14Record {
    frequency: f64,
    c11: f64,
    k1: f64,
    k2: f64,
}

const BA18_F_KAPPA_TRANSITION: f64 = 24.0;
const BA18_IR_REF_FREQUENCY: f64 = 5.0;

/// Relative tolerance when matching a target frequency to a coefficient row.
///
/// The BA18 table is sampled on a log-frequency grid of 0.01 decades, i.e.
/// consecutive rows are ~2.33% apart, so the closest row to any target is
/// within half a step (~1.17%). A miss wider than that means the table's
/// frequency grid is no longer the one these lookups were written against.
const FREQUENCY_MATCH_RTOL: f64 = 0.0117;

/// Index of the coefficient row closest to `target_frequency`.
///
/// # Panics
///
/// Panics if no row lies within [`FREQUENCY_MATCH_RTOL`] of the target. These
/// lookups are the only link between the code and the row ordering of
/// `data/ba18_coefficients.csv`, so a regenerated table with a different
/// frequency grid must fail loudly rather than silently read the wrong
/// coefficients.
fn frequency_index(frequencies: &[f64], target_frequency: f64) -> usize {
    let index = frequencies
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (*left - target_frequency)
                .abs()
                .total_cmp(&(*right - target_frequency).abs())
        })
        .map(|(index, _)| index)
        .expect("BA18 coefficient table is empty");

    let frequency = frequencies[index];
    assert!(
        (frequency - target_frequency).abs() / target_frequency <= FREQUENCY_MATCH_RTOL,
        "BA18 coefficient table has no frequency within {:.2}% of {target_frequency} Hz \
         (closest is {frequency} Hz at index {index}); the coefficient table's \
         frequency grid has changed and the model lookups need revisiting",
        FREQUENCY_MATCH_RTOL * 100.0
    );
    index
}

fn ba18_model_coefficients() {
    let csv_path = "data/ba18_coefficients.csv";
    println!("cargo:rerun-if-changed={}", csv_path);

    let rdr = Reader::from_path(csv_path);
    let mut frequencies = Vec::new();
    let mut f3 = Vec::new();
    let mut f4 = Vec::new();
    let mut f5 = Vec::new();
    let mut c8 = Vec::new();
    for result in rdr
        .expect("Could not open ba18_coefficients.csv for reading")
        .deserialize()
    {
        let record: BA18Record = result.expect("Could not read ba18 coefficient table");
        frequencies.push(record.frequency);
        f3.push(record.f3);
        f4.push(record.f4);
        f5.push(record.f5);
        // BA18 only defines c8 up to the kappa transition frequency (24 Hz);
        // above it the linear site term is kappa-extrapolated from c8 at that
        // frequency and c8[idx] is never read. If the extrapolation branch is read beyond 24 Hz then this NaN will ensure the record is corrupted and hence becomes visible to the caller.
        c8.push(record.c8.unwrap_or(f64::NAN))
    }

    // `{:?}` renders a NaN as the bare token `NaN`, which is not a valid f64
    // literal, so the c8 array (whose gaps above 24 Hz are NaN sentinels) is
    // formatted explicitly.
    let c8_literals = c8
        .iter()
        .map(|value| {
            if value.is_nan() {
                "f64::NAN".to_string()
            } else {
                format!("{value:?}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    let ref_c8_idx = frequency_index(&frequencies, BA18_F_KAPPA_TRANSITION);
    let ir_ref_c8_idx = frequency_index(&frequencies, BA18_IR_REF_FREQUENCY);

    let generated_code = format!(
        r#"
pub static FREQUENCIES: [f64; {}] = {:?};
pub static F3: [f64; {}] = {:?};
pub static F4: [f64; {}] = {:?};
pub static F5: [f64; {}] = {:?};
pub static C8: [f64; {}] = [{}];

pub struct BA18Constants {{
    pub v1: f64,                 // Linear reference velocity (1000 m/s)
    pub v_ref: f64,              // Nonlinear reference velocity (760 m/s)
    pub f_kappa_transition: f64, // Frequency for kappa extrapolation (24 Hz)
}}

pub const CONSTANTS: BA18Constants = BA18Constants {{
    v1: 1000.0,
    v_ref: 760.0,
    f_kappa_transition: {:?},
}};
pub static REF_C8_IDX: usize = {};
pub static IR_REF_C8_IDX: usize = {};
"#,
        frequencies.len(),
        frequencies,
        f3.len(),
        f3,
        f4.len(),
        f4,
        f5.len(),
        f5,
        c8.len(),
        c8_literals,
        // NOTE: This is deliberately not the same as the frequency found in the table.
        BA18_F_KAPPA_TRANSITION,
        ref_c8_idx,
        ir_ref_c8_idx
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir).join("bayless_abrahamson_2018_coefficients.rs");
    fs::write(dest_path, generated_code).unwrap();
}

fn cb14_model_coefficients() {
    let csv_path = "data/cb14_coefficients.csv";
    println!("cargo:rerun-if-changed={}", csv_path);

    let rdr = Reader::from_path(csv_path);
    let mut frequencies = Vec::new();
    let mut c11 = Vec::new();
    let mut k1 = Vec::new();
    let mut k2 = Vec::new();
    for result in rdr
        .expect("Could not open cb14_coefficients.csv for reading")
        .deserialize()
    {
        let record: CB14Record = result.expect("Could not read cb14 coefficient table");
        frequencies.push(record.frequency);
        c11.push(record.c11);
        k1.push(record.k1);
        k2.push(record.k2);
    }
    let generated_code = format!(
        r#"
pub static FREQUENCIES: [f64; {}] = {:?};
pub static C11: [f64; {}] = {:?};
pub static K1: [f64; {}] = {:?};
pub static K2: [f64; {}] = {:?};
"#,
        frequencies.len(),
        frequencies,
        c11.len(),
        c11,
        k1.len(),
        k1,
        k2.len(),
        k2,
    );

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest_path = Path::new(&out_dir).join("campbell_bozorgnia_2014_coefficients.rs");
    fs::write(dest_path, generated_code).unwrap();
}

fn main() {
    ba18_model_coefficients();
    cb14_model_coefficients();
}
