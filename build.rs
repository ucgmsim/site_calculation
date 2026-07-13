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
        c8.push(record.c8.unwrap_or(0.0))
    }

    // 3. Generate the Rust code
    let generated_code = format!(
        r#"
pub static FREQUENCIES: [f64; {}] = {:?};
pub static F3: [f64; {}] = {:?};
pub static F4: [f64; {}] = {:?};
pub static F5: [f64; {}] = {:?};
pub static C8: [f64; {}] = {:?};
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

    // 3. Generate the Rust code
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

    // 4. Write to the output directory
    let out_dir = "src-rust";
    let dest_path = Path::new(&out_dir).join("campbell_bozorgnia_2014_coefficients.rs");
    fs::write(dest_path, generated_code).unwrap();
}

fn main() {
    ba18_model_coefficients();
    cb14_model_coefficients();
}
