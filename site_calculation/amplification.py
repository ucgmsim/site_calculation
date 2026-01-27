import numpy as np
import pandas as pd

from site_calculation import _utils  # type: ignore[unresolved-import]

AmplificationArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]

REQUIRED_COLUMNS = {"vs30", "vs30_sim", "pga"}

CAMPBELL_BOZORGNIA_2014_FREQUENCIES: FrequencyArray = (
    _utils._campbell_bozorgnia_2014_frequencies()
)
BAYLESS_ABRAHAMSON_2018_FREQUENCIES: FrequencyArray = (
    _utils._bayless_abrahamson_2018_frequencies()
)


def campbell_bozorgnia_2014(sites: pd.DataFrame) -> AmplificationArray:
    """Site amplification factor based on the Campbell and Bozorgnia
    2014 spectral acceleration ground motion model [0] with
    modifications[1].

    Parameters
    ----------
    sites : pd.DataFrame
        DataFrame of sites to calculate amplification factors for.
        Must contain all columns in `REQUIRED_COLUMNS`, at least
        ``vs30`` (vs30 of the site), ``vs30_sim`` (vs30 of the
        simulation, typically 500 m/s), and ``pga`` (peak ground
        acceleration from the simulation).

    Returns
    -------
    AmplificationArray
        An array of shape ``(n_stations, n_frequencies)`` where
        ``n_frequencies`` is the number of frequencies supported by
        the model.

    Notes
    -----
    Implementation details are in ``src-rust/campbell_bozorgnia_2014.rs``

    See Also
    --------
    CAMPBELL_BOZORGNIA_2014_FREQUENCIES : Frequencies of the amplification values.

    References
    ----------
    [0] Campbell KW, Bozorgnia Y. NGA-West2 Ground Motion Model for
    the Average Horizontal Components of PGA, PGV, and 5% Damped
    Linear Acceleration Response Spectra. Earthquake Spectra.
    2014;30(3):1087-1115. doi:10.1193/062913EQS175M
    [1] Kuncar F, Bradley BA, de la Torre CA, Rodriguez-Marek A, Zhu C,
    Lee RL. Methods to account for shallow site effects in hybrid
    broadband ground-motion simulations. Earthquake Spectra.
    2025;41(2):1272-1313.
    """
    column_diff = REQUIRED_COLUMNS - set(sites.columns)

    if column_diff:
        missing_columns = sorted(column_diff)
        raise ValueError(f"Required columns missing: {', '.join(missing_columns)}.")

    return _utils._campbell_bozorgnia_2014(
        sites["vs30"].values, sites["vs30_sim"].values, sites["pga"].values
    )


def bayless_abrahamson_2018(sites: pd.DataFrame) -> AmplificationArray:
    """Site amplification factor based on the Bayless and Abrahamson
    2018 effective amplitude spectra ground motion model [0] with
    modifications[1].

    Parameters
    ----------
    sites : pd.DataFrame
        DataFrame of sites to calculate amplification factors for.
        Must contain all columns in `REQUIRED_COLUMNS`, at least
        ``vs30`` (vs30 of the site), ``vs30_sim`` (vs30 of the
        simulation, typically 500 m/s), and ``pga`` (peak ground
        acceleration from the simulation).

    Returns
    -------
    AmplificationArray
        An array of shape ``(n_stations, n_frequencies)`` where
        ``n_frequencies`` is the number of frequencies supported by
        the model.

    Notes
    -----
    Implementation details are in ``src-rust/bayless_abrahamson_2018.rs``

    See Also
    --------
    BAYLESS_ABRAHAMSON_2018_FREQUENCIES : Frequencies of the amplification values.

    References
    ----------
    [0] Bayless, J., & Abrahamson, N. A. (2019). Summary of the BA18
    ground‐motion model for Fourier amplitude spectra for crustal
    earthquakes in California. Bulletin of the Seismological Society
    of America, 109(5), 2088-2105.
    [1] Kuncar F, Bradley BA, de la Torre CA, Rodriguez-Marek A, Zhu C,
    Lee RL. Methods to account for shallow site effects in hybrid
    broadband ground-motion simulations. Earthquake Spectra.
    2025;41(2):1272-1313.
    """
    column_diff = REQUIRED_COLUMNS - set(sites.columns)

    if column_diff:
        missing_columns = sorted(column_diff)
        raise ValueError(f"Required columns missing: {', '.join(missing_columns)}.")

    return _utils._bayless_abrahamson_2018_eas(
        sites["vs30"].values, sites["vs30_sim"].values, sites["pga"].values
    )
