pub mod bayless_abrahamson_2018;
mod bayless_abrahamson_2018_coefficients;
pub mod campbell_bozorgnia_2014;
mod campbell_bozorgnia_2014_coefficients;
pub mod site;
mod tests;

use pyo3::prelude::*;

#[pymodule]
mod _utils {
    use crate::bayless_abrahamson_2018;
    use crate::bayless_abrahamson_2018_coefficients;
    use crate::campbell_bozorgnia_2014;
    use crate::campbell_bozorgnia_2014_coefficients;
    use crate::site::SiteProperties;

    use itertools::izip;
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, ToPyArray};
    use pyo3::prelude::*;

    fn collect_site_properties(
        vs30_py: PyReadonlyArray1<f64>,
        vs30_sim_py: PyReadonlyArray1<f64>,
        pga_py: PyReadonlyArray1<f64>,
    ) -> Vec<SiteProperties> {
        let vs30 = vs30_py.as_slice().unwrap();
        let vs30_sim = vs30_sim_py.as_slice().unwrap();
        let pga = pga_py.as_slice().unwrap();
        izip!(vs30, vs30_sim, pga)
            .map(|(&vs30_one, &vs30_sim_one, &pga_one)| SiteProperties {
                vs30: vs30_one,
                vs30_sim: vs30_sim_one,
                pga: pga_one,
            })
            .collect()
    }

    #[pyfunction]
    fn _bayless_abrahamson_2018_eas<'py>(
        py: Python<'py>,
        vs30_py: PyReadonlyArray1<f64>,
        vs30_sim_py: PyReadonlyArray1<f64>,
        pga_py: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let sites = collect_site_properties(vs30_py, vs30_sim_py, pga_py);
        bayless_abrahamson_2018::bayless_abrahamson_2018_eas(sites.as_slice()).into_pyarray(py)
    }

    #[pyfunction]
    fn _bayless_abrahamson_2018_frequencies<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        // NOTE: This is different from `into_pyarray` in the following sense:
        // 1. Into pyarray :: Creates a numpy array with the backing
        // buffer in rust. If you change the values in that space then
        // it also changes the values in the rust world! This is ok
        // because it also *moves* the value so rust will guarantee
        // you don't use it after that.
        // 2. To pyarray :: Creates a numpy array allocated into
        // Python's heap using Numpy API. It does not move the data,
        // it instead copies the data.
        //
        // We can't move a static global, so we instead copy the array into numpy's memory via the API.
        bayless_abrahamson_2018_coefficients::FREQUENCIES.to_pyarray(py)
    }

    #[pyfunction]
    fn _campbell_bozorgnia_2014_frequencies<'py>(py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        campbell_bozorgnia_2014_coefficients::FREQUENCIES.to_pyarray(py)
    }

    #[pyfunction]
    fn _campbell_bozorgnia_2014<'py>(
        py: Python<'py>,
        vs30_py: PyReadonlyArray1<f64>,
        vs30_sim_py: PyReadonlyArray1<f64>,
        pga_py: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let sites = collect_site_properties(vs30_py, vs30_sim_py, pga_py);
        campbell_bozorgnia_2014::campbell_bozorgnia_2014(sites.as_slice()).into_pyarray(py)
    }
}
