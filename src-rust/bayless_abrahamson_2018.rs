use crate::bayless_abrahamson_2018_coefficients::{C8, F3, F4, F5, FREQUENCIES};
use std::f64::consts::PI;

/// Constants used for high-frequency spectral decay (Kappa)
/// and reference velocity thresholds.
struct ModelConstants {
    pub vs_ref: f64,          // Reference shear-wave velocity (e.g., 1000 m/s)
    pub vs_model_ref: f64,    // Model-specific reference velocity (e.g., 760 m/s)
    pub max_freq_linear: f64, // Frequency threshold for transition to Kappa decay
    pub ref_c8_idx: usize,    // Linear site term coefficient at reference
    pub ir_ref_c8_idx: usize, // Intensity reference coefficient
}

const CONSTANTS: ModelConstants = ModelConstants {
    vs_ref: 1000.0,
    vs_model_ref: 760.0,
    max_freq_linear: 24.0,
    ref_c8_idx: 238,
    ir_ref_c8_idx: 170,
};

pub struct SiteProperties {
    pub vs30: f64,   // Time-averaged shear-wave velocity in upper 30m
    pub vs_sim: f64, // Vs30 in the simulation
    pub pga_high: f64,
}

fn calculate_min_non_linear_factor(intensity: f64, vs30: f64, vs_ref: f64) -> (f64, f64) {
    FREQUENCIES
        .iter()
        .enumerate()
        .map(|(i, &freq)| {
            let f2_val = calculate_f2_coefficient(vs30, vs_ref, F4[i], F5[i]);
            (freq, calculate_non_linear_term(intensity, f2_val, F3[i]))
        })
        .min_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap_or((0.0, 0.0))
}
/// Calculates the linear component of the site amplification.
/// Based on the ratio of local Vs30 to reference velocity.
fn calculate_linear_site_term(vs30: f64, c8: f64) -> f64 {
    (vs30 / CONSTANTS.vs_ref).min(1.0).powf(c8)
}

/// Calculates the high-frequency spectral decay (Kappa).
/// Models how site conditions attenuate high-frequency ground motion.
fn calculate_kappa(vs30: f64) -> f64 {
    let base_kappa = (vs30 / CONSTANTS.vs_model_ref).powf(-0.4);
    base_kappa / 3.5f64.exp()
}

/// Computes the linear site factor, applying high-frequency Kappa decay
/// if the frequency exceeds the model's maximum linear frequency.
fn get_linear_site_factor(vs30: f64, c8: f64, freq: f64) -> f64 {
    if freq < CONSTANTS.max_freq_linear {
        calculate_linear_site_term(vs30, c8)
    } else {
        let kappa = calculate_kappa(vs30);
        let linear_term_ref = calculate_linear_site_term(vs30, C8[CONSTANTS.ref_c8_idx]);
        let frequency_delta = freq - CONSTANTS.max_freq_linear;

        linear_term_ref * (-PI * kappa * frequency_delta).exp()
    }
}

/// Functional form for non-linear site response (soil softening).
fn calculate_non_linear_term(intensity: f64, f2: f64, f3: f64) -> f64 {
    ((intensity + f3) / f3).powf(f2)
}

/// Calculates the f2 coefficient used in non-linear site response.
/// Represents the change in non-linear response relative to Vs30.
fn calculate_f2_coefficient(vs30: f64, vs_ref: f64, f4: f64, f5: f64) -> f64 {
    let vs_constrained = vs30.min(vs_ref);
    let term1 = (f5 * (vs_constrained - 360.0)).exp();
    let term2 = (f5 * (vs_ref - 360.0)).exp();
    f4 * (term1 - term2)
}
/// The primary function to calculate the total site amplification factor.
/// This combines linear amplification and non-linear soil effects.
pub fn calculate_total_site_factor(site: &SiteProperties, idx: usize) -> f64 {
    // Context for the log
    //     "--- Debugging Total Site Factor (Idx: {}, Freq: {} Hz) ---",
    //     idx, FREQUENCIES[idx]
    // );

    //     "Site Inputs: vs30={:.2}, vs_sim={:.2}, pga_high={:.4}",
    //     site.vs30, site.vs_sim, site.pga_high
    // );

    let c8 = C8[idx];
    let freq = FREQUENCIES[idx];

    let linear_vs = get_linear_site_factor(site.vs30, c8, freq);
    let linear_ref = get_linear_site_factor(site.vs_sim, c8, freq);
    let linear_ratio = linear_vs / linear_ref;
    //     "Linear Amp: vs_term={:.4}, ref_term={:.4}, ratio={:.4}",
    //     linear_vs, linear_ref, linear_ratio
    // );

    let ir_ref_c8 = C8[CONSTANTS.ir_ref_c8_idx];
    let ir_vs = calculate_linear_site_term(CONSTANTS.vs_model_ref, ir_ref_c8);
    let ir_ref = calculate_linear_site_term(site.vs_sim, ir_ref_c8);
    let induced_intensity = site.pga_high * (ir_vs / ir_ref).powf(0.846);
    //     "Induced Intensity: ir_vs={:.4}, ir_ref={:.4}, result={:.4}",
    //     ir_vs, ir_ref, induced_intensity
    // );

    let (freq_min, non_linear_min) =
        calculate_min_non_linear_factor(induced_intensity, site.vs30, CONSTANTS.vs_model_ref);
    //     "Non-Linear Bounds: freq_min={:.2}, nl_min={:.4}",
    //     freq_min, non_linear_min
    // );

    let f2_val = calculate_f2_coefficient(site.vs30, CONSTANTS.vs_model_ref, F4[idx], F5[idx]);
    let current_nonlinear = calculate_non_linear_term(induced_intensity, f2_val, F3[idx]);
    //     "F2 Coeff: {:.4}, Current NL Term: {:.4}",
    //     f2_val, current_nonlinear
    // );

    let adjusted_nonlinear = if freq < freq_min {
        current_nonlinear
    } else {
        non_linear_min
    };

    let total_factor = linear_ratio * adjusted_nonlinear;

    total_factor
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::error::Error;

    #[test]
    fn test_linear_site_term() {
        assert_abs_diff_eq!(
            calculate_linear_site_term(650.0, C8[0]),
            1.3773640,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_linear_site_term_saturation() {
        assert_abs_diff_eq!(
            calculate_linear_site_term(650.0, C8[239]),
            1.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_kappa() {
        assert_abs_diff_eq!(calculate_kappa(650.0), 0.032146, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_site_factor() {
        let idx = 50; // 0.316Hz
        let c8 = C8[idx];
        let f = FREQUENCIES[idx];
        let vs30 = 650.0;
        assert_abs_diff_eq!(
            get_linear_site_factor(vs30, c8, f),
            1.574327,
            epsilon = 1e-6
        )
    }

    #[test]
    fn test_linear_site_factor_decay() {
        let idx = 246; // 28.840Hz
        let c8 = C8[idx];
        let f = FREQUENCIES[idx];
        let vs30 = 650.0;
        assert_abs_diff_eq!(
            get_linear_site_factor(vs30, c8, f),
            0.494244,
            epsilon = 1e-6
        )
    }

    #[test]
    fn test_calculate_f2_coefficient() {
        // fn calculate_f2_coefficient(vs30: f64, vs_ref: f64, f4: f64, f5: f64) -> f64 {
        let vs30 = 650.0;
        let vs_ref = CONSTANTS.vs_model_ref;
        let idx = 159; // 3.890Hz
        let f4 = F4[idx];
        let f5 = F5[idx];
        let f2 = calculate_f2_coefficient(vs30, vs_ref, f4, f5);
        assert_abs_diff_eq!(f2, -0.034994, epsilon = 1e-6)
    }

    #[test]
    fn test_calculate_non_linear_term() {
        // fn calculate_non_linear_term(intensity: f64, f2: f64, f3: f64) -> f64 {
        //     ((intensity + f3) / f3).powf(f2)
        // }
        let ir = 0.387394736043994;
        let vs30 = 650.0;
        let vs_ref = CONSTANTS.vs_model_ref;
        let idx = 0; // 3.890Hz
        let f3 = F3[idx];
        let f4 = F4[idx];
        let f5 = F5[idx];
        let f2 = calculate_f2_coefficient(vs30, vs_ref, f4, f5);
        assert_abs_diff_eq!(calculate_non_linear_term(ir, f2, f3), 1.0, epsilon = 1e-6);
    }
    #[test]
    fn test_calculate_min_non_linear_factor() {
        let ir = 0.387394736043994;
        let vs30 = 650.0;
        let (f_min, non_linear_min) =
            calculate_min_non_linear_factor(ir, vs30, CONSTANTS.vs_model_ref);
        assert_abs_diff_eq!(f_min, 8.709635, epsilon = 1e-6);
        assert_abs_diff_eq!(non_linear_min, 0.907103674068, epsilon = 1e-12);
    }

    #[test]
    fn test_computed_site_factors() {
        let site_properties = SiteProperties {
            vs30: 650.0,
            vs_sim: 500.0,
            pga_high: 0.46,
        };
        // calculate_total_site_factor(site: &SiteProperties, idx: usize) -> f64 {
        let sf = calculate_total_site_factor(&site_properties, 0);
        let sf_high = calculate_total_site_factor(&site_properties, 67);
        assert_abs_diff_eq!(sf, 0.822837, epsilon = 1e-5);
        assert_abs_diff_eq!(sf_high, 0.74334, epsilon = 1e-5);
    }

    #[test]
    fn test_computed_site_factors_from_csv() -> Result<(), Box<dyn Error>> {
        // 1. Initialise the CSV reader
        // Replace with your actual file path
        let mut rdr = csv::Reader::from_path(
            "/home/jake/src/site_calculation/src-rust/data/site_factors_data.csv",
        )?;

        for result in rdr.records() {
            let record = result?;

            // 2. Manually extract and parse the first three columns
            let vs30: f64 = record.get(0).ok_or("Missing vs30")?.parse()?;
            let vs_sim: f64 = record.get(1).ok_or("Missing vs_sim")?.parse()?;
            let pga_high: f64 = record.get(2).ok_or("Missing pga_high")?.parse()?;

            let site_properties = SiteProperties {
                vs30,
                vs_sim,
                pga_high,
            };

            // 3. Iterate over the FREQUENCIES array and check corresponding columns
            // Columns for site factors start at index 3
            for (i, &freq) in FREQUENCIES.iter().enumerate() {
                let expected_sf_str = record.get(3 + i).ok_or_else(|| {
                    format!(
                        "Missing site factor for frequency {}Hz at column {}",
                        freq,
                        3 + i
                    )
                })?;

                let expected_sf: f64 = expected_sf_str.parse()?;

                // 4. Compute and Assert
                let actual_sf = calculate_total_site_factor(&site_properties, i);
                assert_abs_diff_eq!(actual_sf, expected_sf, epsilon = 5e-3);
            }
        }

        Ok(())
    }
}
