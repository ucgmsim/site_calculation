"""Type stubs for the ``site_calculation._utils`` Rust extension."""

import numpy as np

type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

def _bayless_abrahamson_2018_eas(
    vs30_py: FloatArray1D,
    vs30_sim_py: FloatArray1D,
    pga_py: FloatArray1D,
) -> FloatArray2D: ...
def _bayless_abrahamson_2018_frequencies() -> FloatArray1D: ...
def _campbell_bozorgnia_2014_frequencies() -> FloatArray1D: ...
def _campbell_bozorgnia_2014(
    vs30_py: FloatArray1D,
    vs30_sim_py: FloatArray1D,
    pga_py: FloatArray1D,
) -> FloatArray2D: ...
