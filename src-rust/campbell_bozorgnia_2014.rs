//! Site amplification factor based on the spectral acceleration ground motion model (Campbell and Borzorgnia 2014) with modifications (Kuncar F. et al 2025).
//!
//! This module implements the site amplification factor based on the CB14 model
//! with nonlinearity considered in the site but not the simulation.
//!
//! Comments on the page numbers are in the form (n1; n2) where n1 is
//! the physical page number in the printed journal, and n2 is the
//! page number in the PDF.
//!
//! # Reference
//! [0]: Campbell KW, Bozorgnia Y. NGA-West2 Ground Motion Model for
//! the Average Horizontal Components of PGA, PGV, and 5% Damped
//! Linear Acceleration Response Spectra. Earthquake Spectra.
//! 2014;30(3):1087-1115. doi:10.1193/062913EQS175M
//!
//! [1]: Kuncar F, Bradley BA, de la Torre CA, Rodriguez-Marek A, Zhu C,
//! Lee RL. Methods to account for shallow site effects in hybrid
//! broadband ground-motion simulations. Earthquake Spectra.
//! 2025;41(2):1272-1313.
use crate::campbell_bozorgnia_2014_coefficients::{C11, FREQUENCIES, K1, K2};
use crate::site::SiteProperties;
use ndarray::prelude::*;

/// Constants for the CB14 (Campbell & Bozorgnia, 2014) model.
struct CB14Constants {
    /// Period-independent model coefficient 'c', defined on page (1095; 9) and as a footnote to Table 1.
    pub scon_c: f64,
    /// Period-independent model coefficient 'n', defined on page (1095; 9) and as a footnote to Table 1.
    pub scon_n: f64,
    /// Reference rock velocity Vs30 = 1,100 m/s used to define A1100 (1104; 18).
    pub vs_high: f64,
    /// Index for 1000Hz (approx. PGA) rock adjustment. This is the
    /// same as 100Hz row in Table 1, but uses the PGA value in the
    /// footnote to Table 1 instead of the 100Hz value.
    pub idx_1000hz: usize,
}

const CONSTANTS: CB14Constants = CB14Constants {
    scon_c: 1.88,
    scon_n: 1.18,
    vs_high: 1100.0,
    idx_1000hz: FREQUENCIES.len() - 1,
};

/// Implements the linear and nonlinear site response term for Vs30 <= k1.
/// Corresponds to the first branch of Equation (18) for f_site,G.
fn fs_low(vs30: f64, a1100: f64, c11: f64, k1: f64, k2: f64) -> f64 {
    let scon_c = CONSTANTS.scon_c;
    let scon_n = CONSTANTS.scon_n;

    // ln(Vs30 / k1) term
    let log_vs30_minus_k1 = (vs30 / k1).ln();

    // ln(A1100 + c * (Vs30 / k1)^n)
    let nl_term_1 = (a1100 + scon_c * (vs30 / k1).powf(scon_n)).ln();

    // ln(A1100 + c)
    let nl_term_2 = (a1100 + scon_c).ln();

    // Result: c11 * ln(Vs30/k1) + k2 * [ln(A1100 + c*(Vs30/k1)^n) - ln(A1100 + c)]
    c11 * log_vs30_minus_k1 + k2 * (nl_term_1 - nl_term_2)
}

/// Implements the linear site response term for Vs30 > k1.
/// Corresponds to the second branch of Equation (18) for f_site,G.
fn fs_mid_high(vs30: f64, c11: f64, k1: f64, k2: f64) -> f64 {
    let scon_n = CONSTANTS.scon_n;
    // Result: (c11 + k2 * n) * ln(Vs30 / k1)
    (c11 + k2 * scon_n) * ((vs30 / k1).ln())
}

/// Helper to select the appropriate branch of the site response model based on Vs30.
/// Selection logic defined in Equation (18).
fn compute_fs_value(vs30: f64, a1100: f64, c11: f64, k1: f64, k2: f64) -> f64 {
    if vs30 < k1 {
        fs_low(vs30, a1100, c11, k1, k2)
    } else {
        fs_mid_high(vs30, c11, k1, k2)
    }
}

/// Computes the site amplification factor relative to the simulation reference.
fn campbell_bozorgnia_2014_one(site: &SiteProperties, out: ArrayViewMut1<f64>) {
    let c11_1000hz = C11[CONSTANTS.idx_1000hz];
    let k1_1000hz = K1[CONSTANTS.idx_1000hz];
    let k2_1000hz = K2[CONSTANTS.idx_1000hz];

    // Modification from (Kuncar et al. 2025): Consider only linear site amplification in rock PGA adjustment.
    // See page (1285; 14) of [1].
    let fs_vhigh = compute_fs_value(CONSTANTS.vs_high, 0.0, c11_1000hz, k1_1000hz, k2_1000hz);
    let fs_vsim = compute_fs_value(site.vs30_sim, 0.0, c11_1000hz, k1_1000hz, k2_1000hz);
    let a1100 = site.pga * (fs_vhigh - fs_vsim).exp();
    azip!((&c11 in &C11, &k1 in &K1, &k2 in &K2, out in out) {
        // f_site for the actual site conditions
        let fs_site = compute_fs_value(site.vs30, a1100, c11, k1, k2);
        // f_site for the simulation reference (linear rock baseline)
        let fs_sim = compute_fs_value(site.vs30_sim, 0.0, c11, k1, k2);
        // Amplification = exp(f_site_actual - f_site_sim)
        *out = (fs_site - fs_sim).exp();
    });
}

pub fn campbell_bozorgnia_2014(sites: &[SiteProperties]) -> Array2<f64> {
    let n_stations = sites.len();
    let n_frequencies = FREQUENCIES.len();
    let mut out = Array2::default((n_stations, n_frequencies));
    azip!((site in sites, out in out.axis_iter_mut(Axis(0))) {
        campbell_bozorgnia_2014_one(site, out)
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_amplification_approx_eq;
    use std::error::Error;
    use std::path::PathBuf;

    #[test]
    fn test_computed_site_factors_from_csv() -> Result<(), Box<dyn Error>> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests");
        path.push("resources");
        path.push("cb14_site_factors_data.csv");

        let mut rdr = csv::Reader::from_path(path)?;

        for result in rdr.records() {
            let record = result?;

            let vs30: f64 = record.get(0).ok_or("Missing vs30")?.parse()?;
            let vs30_sim: f64 = record.get(1).ok_or("Missing vs_sim")?.parse()?;
            let pga: f64 = record.get(2).ok_or("Missing pga_high")?.parse()?;

            let site_properties = SiteProperties {
                vs30,
                vs30_sim,
                pga,
            };
            let mut expected_site_factors: Vec<f64> = Vec::with_capacity(FREQUENCIES.len() - 1);

            for (i, &freq) in FREQUENCIES.iter().take(FREQUENCIES.len() - 1).enumerate() {
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
            campbell_bozorgnia_2014_one(&site_properties, calculated_site_factors_array.view_mut());
            let expected_site_factors_array = Array1::from_vec(expected_site_factors);
            let calculated_slice = calculated_site_factors_array.slice(s![..-1]);
            let ctxt = format!("Site properties = {:?}", site_properties);
            assert_amplification_approx_eq(
                calculated_slice,
                expected_site_factors_array.view(),
                &FREQUENCIES,
                5e-5,
                &ctxt,
            );
        }

        Ok(())
    }
}
