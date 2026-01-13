use csv::Reader;
use std::fs;
use std::path::Path;

#[derive(Debug, serde::Deserialize)]
struct Record {
    frequency: f64,
    f3: f64,
    f4: f64,
    f5: f64,
    c8: Option<f64>,
}

fn main() {
    // 1. Path to your source CSV
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
        let record: Record = result.expect("Could not read ba18 coefficient table");
        frequencies.push(record.frequency);
        f3.push(record.f3);
        f4.push(record.f4);
        f5.push(record.f5);
        c8.push(record.c8.unwrap_or(0.0))
    }

    // 3. Generate the Rust code
    let generated_code = format!(
        r#"
pub const FREQUENCIES: [f64; {}] = {:?};
pub const F3: [f64; {}] = {:?};
pub const F4: [f64; {}] = {:?};
pub const F5: [f64; {}] = {:?};
pub const C8: [f64; {}] = {:?};
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
        c8
    );

    // 4. Write to the output directory
    let out_dir = "src-rust";
    let dest_path = Path::new(&out_dir).join("bayless_abrahamson_2018_coefficients.rs");
    fs::write(dest_path, generated_code).unwrap();
}
