#[cfg(test)]
pub fn assert_amplification_approx_eq(
    actual: ndarray::ArrayView1<f64>,
    expected: ndarray::ArrayView1<f64>,
    freqs: &[f64],
    epsilon: f64,
    context: &str,
) {
    use approx::abs_diff_eq;

    if !abs_diff_eq!(actual, expected, epsilon = epsilon) {
        let mut table =
            String::from("\n| Index | Freq (Hz) | Expected | Actual | Diff | Status |\n");
        table.push_str("|-------|-----------|----------|--------|------|--------|\n");

        for i in 0..expected.len() {
            let freq = freqs.get(i).unwrap_or(&-1.0);
            let exp = expected[i];
            let act = actual[i];
            let diff = (exp - act).abs();
            let status = if diff <= epsilon {
                "  ✅  "
            } else {
                "  ❌  "
            };

            table.push_str(&format!(
                "| {:<5} | {:<9.2} | {:<8.4} | {:<6.4} | {:<4.4} | {:<6} |\n",
                i, freq, exp, act, diff, status
            ));
        }

        panic!(
            "\nNumerical Comparison Failed: {}\n{}\nEpsilon: {}\n",
            context, table, epsilon
        );
    }
}
