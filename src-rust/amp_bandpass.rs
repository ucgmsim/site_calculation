use std::ops::Div;

use ndarray::azip;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

fn amp_bandpass_one(
    ampf: ArrayViewMut1<f64>,
    log_fftfreq: ArrayView1<f64>,
    log_fmin: f64,
    log_fmidbot: f64,
    log_fhightop: f64,
    log_fmax: f64,
) {
    azip!(
        (amp in ampf, log_freq in log_fftfreq) {
            if (log_fmin..log_fmidbot).contains(log_freq) {
                let fmin_diff = (log_freq - log_fmin) / (log_fmidbot - log_fmin);
                *amp = (1.0 - fmin_diff) * (*amp) + fmin_diff;
            }
            else if (log_fhightop..log_fmax).contains(log_freq) {
                let fmax_diff = (log_freq - log_fmin) / (log_fmidbot - log_fmin);
                *amp = (1.0 - fmax_diff) * (*amp) + fmax_diff;
            }
            else if !(log_fmidbot..log_fhightop).contains(log_freq) {
                *amp = 1.0;
            }
        }
    );
}

fn fftfreq(n: usize, dt: f64) -> Array1<f64> {
    Array1::from_shape_fn(n.div(2) + 1, |n| n as f64) / (dt * (n as f64))
}

fn interp(
    xp: ArrayView1<f64>,
    x_out: ArrayView1<f64>,
    arr: ArrayView1<f64>,
    out: ArrayViewMut1<f64>,
) {
    // Assumes already sorted.
    // Pointer to the last good interpolation node.
    let mut i = 0;
    let xp_slice = xp.as_slice().unwrap();
    azip!((&x in x_out, y in out) {
        if x > xp_slice[i + 1] {
            i += xp_slice[i..].binary_search_by(|y| y.total_cmp(&x)).unwrap_or_else(|j| j);
        }

        let xp_0 = xp_slice[i];
        let xp_1 = xp_slice[i + 1];
        let dxp = xp_1 - xp_0;
        let dx = x - xp_0;
        let yp_0 = arr[i];
        let yp_1 = arr[i + 1];

        *y = yp_0 + (dx / dxp) * (yp_1 - yp_0);
    });
}

fn interpolate_frequencies(
    freqs_in: ArrayView1<f64>,
    freqs_out: ArrayView1<f64>,
    ampf0: ArrayView2<f64>,
) -> Array2<f64> {
    let (n_stations, _) = ampf0.dim();
    let mut out = Array2::default((n_stations, freqs_out.dim()));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(ampf0.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(out_amp, in_amp)| {
            interp(freqs_in, freqs_out, in_amp, out_amp);
        });
    out
}

fn amp_bandpass(
    mut ampv: ArrayViewMut2<f64>,
    fftfreq: ArrayView1<f64>,
    fmin: f64,
    fmidbot: f64,
    fhightop: f64,
    fmax: f64,
) {
    let log_fmin = fmin.ln();
    let log_fmidbot = fmidbot.ln();
    let log_fhightop = fhightop.ln();
    let log_fmax = fmax.ln();
    let log_fftfreq = fftfreq.ln();
    ampv.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|ampv| {
            amp_bandpass_one(
                ampv,
                log_fftfreq.view(),
                log_fmin,
                log_fmidbot,
                log_fhightop,
                log_fmax,
            )
        });
}
