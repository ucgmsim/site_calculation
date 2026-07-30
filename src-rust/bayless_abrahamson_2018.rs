//! Site amplification factor based on the effective amplitude spectra ground motion model (Bayless and Abrahamson 2018) with modifications (Kuncar F. et al 2025).
//!
//! This module implements the site amplification factor based on the BA18 model
//! with nonlinearity and kappa extrapolation. It does not implement Z1.0 scaling.
//!
//! # Reference
//! Bayless, J., & Abrahamson, N. A. (2019). Summary of the BA18
//! ground‐motion model for Fourier amplitude spectra for crustal
//! earthquakes in California. Bulletin of the Seismological Society
//! of America, 109(5), 2088-2105.
//!
//! Kuncar F, Bradley BA, de la Torre CA, Rodriguez-Marek A, Zhu C,
//! Lee RL. Methods to account for shallow site effects in hybrid
//! broadband ground-motion simulations. Earthquake Spectra.
//! 2025;41(2):1272-1313.

use crate::bayless_abrahamson_2018_coefficients::{C8, F3, F4, F5, FREQUENCIES};
use crate::site::SiteProperties;
use ndarray::prelude::*;
use std::f64::consts::PI;
use std::sync::LazyLock;

/// Constants for the BA18 (Bayless & Abrahamson, 2018) model.
struct BA18Constants {
    pub v1: f64,                 // Linear reference velocity (1000 m/s)
    pub v_ref: f64,              // Nonlinear reference velocity (760 m/s)
    pub f_kappa_transition: f64, // Frequency for kappa extrapolation (24 Hz)
    pub ir_ref_frequency: f64,   // Frequency the Ir reference site factor is taken at (5 Hz)
}

const CONSTANTS: BA18Constants = BA18Constants {
    v1: 1000.0,
    v_ref: 760.0,
    f_kappa_transition: 24.0,
    ir_ref_frequency: 5.0,
};

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
fn frequency_index(target_frequency: f64) -> usize {
    let index = FREQUENCIES
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            (*left - target_frequency)
                .abs()
                .total_cmp(&(*right - target_frequency).abs())
        })
        .map(|(index, _)| index)
        .expect("BA18 coefficient table is empty");

    let frequency = FREQUENCIES[index];
    assert!(
        (frequency - target_frequency).abs() / target_frequency <= FREQUENCY_MATCH_RTOL,
        "BA18 coefficient table has no frequency within {:.2}% of {target_frequency} Hz \
         (closest is {frequency} Hz at index {index}); the coefficient table's \
         frequency grid has changed and the model lookups need revisiting",
        FREQUENCY_MATCH_RTOL * 100.0
    );
    index
}

static REF_C8_IDX: LazyLock<usize> =
    LazyLock::new(|| frequency_index(CONSTANTS.f_kappa_transition));
static IR_REF_C8_IDX: LazyLock<usize> =
    LazyLock::new(|| frequency_index(CONSTANTS.ir_ref_frequency));

/// Finds the minimum nonlinear site factor across the spectrum to enforce
/// model constraints on soil softening.
fn calc_min_f_nl(ir: f64, vs30: f64, v_ref: f64) -> (f64, f64) {
    FREQUENCIES
        .iter()
        .enumerate()
        .map(|(i, &freq)| {
            let f2 = calc_f2(vs30, v_ref, F4[i], F5[i]);
            (freq, calc_f_nl_exponent(ir, f2, F3[i]))
        })
        .min_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap_or((1.0, 1.0))
}

/// Calculates the linear site term factor: exp(f_sl) = (Vs30/V1)^c8.
/// Based on Equation 10b.
fn calc_f_sl_exponent(vs30: f64, c8: f64) -> f64 {
    (vs30 / CONSTANTS.v1).min(1.0).powf(c8)
}

/// Calculates high-frequency spectral decay (Kappa) for extrapolation
/// above 24 Hz.
fn calc_kappa(vs30: f64) -> f64 {
    let base_kappa = (vs30 / CONSTANTS.v_ref).powf(-0.4);
    base_kappa / 3.5f64.exp()
}

/// Computes the linear site factor, applying Kappa-based extrapolation
/// if freq > 24 Hz.
fn calc_linear_site_factor_with_kappa(vs30: f64, c8: f64, kappa: f64, freq: f64) -> f64 {
    if freq < CONSTANTS.f_kappa_transition {
        calc_f_sl_exponent(vs30, c8)
    } else {
        let linear_term_ref = calc_f_sl_exponent(vs30, C8[*REF_C8_IDX]);
        let df = freq - CONSTANTS.f_kappa_transition;

        linear_term_ref * (-PI * kappa * df).exp()
    }
}

/// Calculates the nonlinear site term factor: exp(f_nl) = ((ir + f3) / f3)^f2.
/// Based on Equation 10c.
fn calc_f_nl_exponent(ir: f64, f2: f64, f3: f64) -> f64 {
    ((ir + f3) / f3).powf(f2)
}

/// Calculates the f2 coefficient for nonlinear response scaling.
/// Based on Equation 10d.
fn calc_f2(vs30: f64, v_ref: f64, f4: f64, f5: f64) -> f64 {
    let vs_min = vs30.min(v_ref);
    let term1 = (f5 * (vs_min - 360.0)).exp();
    let term2 = (f5 * (v_ref - 360.0)).exp();
    f4 * (term1 - term2)
}

/// Primary function to calculate the total site amplification factor f_s.
/// Combines linear site response (f_sl) and nonlinear soil effects (f_nl).
pub fn calc_f_s(
    site: &SiteProperties,
    ir: f64,
    f_min: f64,
    f_nl_min: f64,
    kappa: f64,
    kappa_sim: f64,
    idx: usize,
) -> f64 {
    let c8 = C8[idx];
    let freq = FREQUENCIES[idx];

    // Linear amplification ratio relative to simulation velocity
    let f_sl_vs = calc_linear_site_factor_with_kappa(site.vs30, c8, kappa, freq);
    let f_sl_sim = calc_linear_site_factor_with_kappa(site.vs30_sim, c8, kappa_sim, freq);
    let linear_ratio = f_sl_vs / f_sl_sim;

    // Apply the floor to the nonlinear term
    let f_nl = if freq < f_min {
        // Get non-linear term f_nl
        let f2 = calc_f2(site.vs30, CONSTANTS.v_ref, F4[idx], F5[idx]);
        calc_f_nl_exponent(ir, f2, F3[idx])
    } else {
        f_nl_min
    };

    linear_ratio * f_nl
}

fn calc_nl_ir_parameters(site: &SiteProperties) -> (f64, f64, f64) {
    // Calculate induced intensity (Ir) based on Equation 10e.
    // The frequency grid these coefficient lookups assume is checked in
    // `frequency_index`, which panics (in release builds too) on a mismatch.
    let c8_5hz = C8[*IR_REF_C8_IDX];
    let ir_sim = calc_f_sl_exponent(site.vs30_sim, c8_5hz);
    // This IR calculation is derived in the e-Supp to Kuncar et al. 2025
    // Equation B.2 of
    // https://journals.sagepub.com/doi/suppl/10.1177/87552930241301059/suppl_file/sj-pdf-1-eqs-10.1177_87552930241301059.pdf
    let ir_ref_vs = calc_f_sl_exponent(CONSTANTS.v_ref, C8[*IR_REF_C8_IDX]);
    let ir = site.pga * (ir_ref_vs / ir_sim).powf(0.846);
    let (f_min, f_nl_min) = calc_min_f_nl(ir, site.vs30, CONSTANTS.v_ref);
    (ir, f_min, f_nl_min)
}

fn bayless_abrahamson_2018_eas_one(site: &SiteProperties, out: ArrayViewMut1<f64>) {
    // This is calculated once because it is independent of frequency.
    let (ir, f_min, f_nl_min) = calc_nl_ir_parameters(site);
    let kappa = calc_kappa(site.vs30);
    let kappa_sim = calc_kappa(site.vs30_sim);
    azip!((index idx, out in out) {
        *out = calc_f_s(site, ir, f_min, f_nl_min, kappa, kappa_sim, idx)
    });
}

pub fn bayless_abrahamson_2018_eas(sites: &[SiteProperties]) -> Array2<f64> {
    let n_stations = sites.len();
    let n_frequencies = FREQUENCIES.len();
    let mut out = Array2::default((n_stations, n_frequencies));
    azip!((site in sites, out in out.axis_iter_mut(Axis(0))) {
        bayless_abrahamson_2018_eas_one(site, out)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::error::Error;
    use std::path::PathBuf;

    #[test]
    fn test_f_sl_exponent() {
        assert_abs_diff_eq!(calc_f_sl_exponent(650.0, C8[0]), 1.3773640, epsilon = 1e-6);
    }

    #[test]
    fn test_f_sl_exponent_saturation() {
        // At or above v1 the linear term saturates: (vs30/v1).min(1.0) == 1.0,
        // so the factor is 1.0 for any c8.
        assert_abs_diff_eq!(calc_f_sl_exponent(CONSTANTS.v1, C8[0]), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(calc_f_sl_exponent(1500.0, C8[0]), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_c8_undefined_above_kappa_transition_is_nan() {
        // BA18 does not define c8 above the kappa transition frequency. This
        // checks that trying to access them gives us a NaN.
        assert!(C8[*REF_C8_IDX].is_finite());
        assert!(C8[*REF_C8_IDX + 1].is_nan());
        assert!(FREQUENCIES[*REF_C8_IDX + 1] > CONSTANTS.f_kappa_transition);
    }

    #[test]
    fn test_frequency_index_lookups() {
        assert_abs_diff_eq!(FREQUENCIES[*REF_C8_IDX], 23.988321, epsilon = 1e-6);
        assert_abs_diff_eq!(FREQUENCIES[*IR_REF_C8_IDX], 5.011872, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "no frequency within")]
    fn test_frequency_index_rejects_off_grid_target() {
        // 200 Hz is well outside the table, so the closest row is nowhere near
        // half a grid step away and the lookup must refuse it.
        frequency_index(200.0);
    }

    #[test]
    fn test_kappa() {
        assert_abs_diff_eq!(calc_kappa(650.0), 0.032146, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_site_factor() {
        let idx = 50; // 0.316Hz
        let c8 = C8[idx];
        let f = FREQUENCIES[idx];
        let vs30 = 650.0;
        let kappa = calc_kappa(vs30);
        assert_abs_diff_eq!(
            calc_linear_site_factor_with_kappa(vs30, c8, kappa, f),
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
        let kappa = calc_kappa(vs30);
        assert_abs_diff_eq!(
            calc_linear_site_factor_with_kappa(vs30, c8, kappa, f),
            0.494244,
            epsilon = 1e-6
        )
    }

    #[test]
    fn test_calc_f2() {
        let vs30 = 650.0;
        let v_ref = CONSTANTS.v_ref;
        let idx = 159; // 3.890Hz
        let f4 = F4[idx];
        let f5 = F5[idx];
        let f2 = calc_f2(vs30, v_ref, f4, f5);
        assert_abs_diff_eq!(f2, -0.034994, epsilon = 1e-6)
    }

    #[test]
    fn test_f_nl_exponent() {
        let ir = 0.387394736043994;
        let vs30 = 650.0;
        let v_ref = CONSTANTS.v_ref;
        let idx = 0; // 0.1Hz
        let f3 = F3[idx];
        let f4 = F4[idx];
        let f5 = F5[idx];
        let f2 = calc_f2(vs30, v_ref, f4, f5);
        assert_abs_diff_eq!(calc_f_nl_exponent(ir, f2, f3), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_calc_min_f_nl() {
        let ir = 0.387394736043994;
        let vs30 = 650.0;
        let (f_min, f_nl_min) = calc_min_f_nl(ir, vs30, CONSTANTS.v_ref);
        assert_abs_diff_eq!(f_min, 8.709635, epsilon = 1e-6);
        assert_abs_diff_eq!(f_nl_min, 0.907103674068, epsilon = 1e-12);
    }

    #[test]
    fn test_computed_site_factors() {
        let site_properties = SiteProperties {
            vs30: 650.0,
            vs30_sim: 500.0,
            pga: 0.46,
        };
        let (ir, f_min, f_nl_min) = calc_nl_ir_parameters(&site_properties);
        let kappa = calc_kappa(site_properties.vs30);
        let kappa_sim = calc_kappa(site_properties.vs30_sim);
        let sf = calc_f_s(&site_properties, ir, f_min, f_nl_min, kappa, kappa_sim, 0);
        let sf_high = calc_f_s(&site_properties, ir, f_min, f_nl_min, kappa, kappa_sim, 67);
        assert_abs_diff_eq!(sf, 0.822837, epsilon = 1e-5);
        assert_abs_diff_eq!(sf_high, 0.74334, epsilon = 1e-5);
    }

    #[test]
    fn test_computed_site_factors_from_csv() -> Result<(), Box<dyn Error>> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("resources");
        path.push("site_factors_data.csv");

        let mut rdr = csv::Reader::from_path(path)?;

        for result in rdr.records() {
            let record = result?;

            let vs30: f64 = record.get(0).ok_or("Missing vs30")?.parse()?;
            let vs_host: f64 = record.get(1).ok_or("Missing vs_sim")?.parse()?;
            let pga: f64 = record.get(2).ok_or("Missing pga_high")?.parse()?;

            let site_properties = SiteProperties {
                vs30,
                vs30_sim: vs_host,
                pga,
            };
            let mut expected_site_factors: Vec<f64> = Vec::with_capacity(FREQUENCIES.len());

            for (i, &freq) in FREQUENCIES.iter().enumerate() {
                let expected_sf_str = record.get(3 + i).ok_or_else(|| {
                    format!(
                        "Missing site factor for frequency {}Hz at column {}",
                        freq,
                        3 + i
                    )
                })?;

                let expected_sf = expected_sf_str.parse()?;
                expected_site_factors.push(expected_sf)
            }
            let mut calculated_site_factors_array = Array1::default(FREQUENCIES.len());
            bayless_abrahamson_2018_eas_one(
                &site_properties,
                calculated_site_factors_array.view_mut(),
            );
            let expected_site_factors_array = Array1::from_vec(expected_site_factors);
            assert_abs_diff_eq!(
                calculated_site_factors_array,
                expected_site_factors_array,
                epsilon = 5e-3
            )
        }

        Ok(())
    }
}
