import multiprocessing

import numpy as np
import pandas as pd
import pyfftw
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import scipy as sp

from site_calculation import _utils  # type: ignore[unresolved-import]

AmplificationArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
WaveformArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FrequencyArray = np.ndarray[tuple[int,], np.dtype[np.float64]]
Filter = np.ndarray[tuple[int,], np.dtype[np.float64]]

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


def amplify_waveform(
    waveform: WaveformArray,
    amplification_factor: AmplificationArray,
    cores: int = multiprocessing.cpu_count(),
    taper: bool = True,
) -> np.ndarray:
    """Apply amplification factor to waveforms.

    Parameters
    ----------
    waveform : np.ndarray
        The input waveform.
    amplification_factor : np.ndarray
        The frequency amplification factors. If `waveform` has
        length `2^i`, then `amplification_factor` should have length `2^(ceil(i) -
        1)`.
    cores : int, optional
        The number of cores to use for FFT. Defaults to all cores
        available on the system as reported by
        `muliprocessing.cpu_count()`.
    taper : bool, optional
        If true, taper the waveform to avoid spectral leakage. Default
        True.

    Returns
    -------
    np.ndarray
        The input waveform (de)amplified at frequencies according to
        the values of `amplification_factor`.
    """
    # Saved for later to reset after inverse Fourier.
    old_fftw_num_threads = pyfftw.config.NUM_THREADS
    pyfftw.config.NUM_THREADS = cores

    nt = waveform.shape[-1]
    waveform_dtype = waveform.dtype

    # Taper 5% on the right using the Hanning method
    ntap = int(nt * 0.05)

    if ntap > 0 and taper:
        # Create a Hanning window for the taper, ensuring it's float32
        hanning_window = np.hanning(ntap * 2 + 1)[ntap + 1 :].astype(waveform_dtype)
        # Create a copy of the original waveform so-as not to modify it in-place.
        waveform = waveform.copy()
        waveform[..., nt - ntap :] *= hanning_window

    n_fft = 2 * amplification_factor.shape[-1]

    # NOTE: The old code had the following resizing behaviour
    # timeseries = np.resize(timeseries, ft_len)
    # timeseries[nt:] = 0
    # this is actually unnecessary as setting `n=n_fft` will automatically do the same thing
    # See: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
    # and the PYFFTW equivalent:
    # https://pyfftw.readthedocs.io/en/latest/source/pyfftw/interfaces/numpy_fft.html#pyfftw.interfaces.numpy_fft.rfft

    fourier = pyfftw_fft.rfft(waveform, n=n_fft, axis=-1)

    fourier[..., :-1] *= amplification_factor.astype(waveform.dtype)

    result_full = pyfftw_fft.irfft(fourier, n=n_fft, axis=-1)
    pyfftw.config.NUM_THREADS = old_fftw_num_threads
    # Trim to original length
    return result_full[..., :nt]


def interpolate_frequencies(
    model_frequencies: FrequencyArray,
    output_frequencies: FrequencyArray,
    amplification_array: AmplificationArray,
) -> AmplificationArray:
    """Interpolate frequencies into another frequency space.

    Parameters
    ----------
    model_frequencies : FrequencyArray
        The input model frequencies.
    output_frequencies : FrequencyArray
        The output frequencies, e.g. FFT output frequencies.
    amplification_array : AmplificationArray
        The amplification array. Interpolation occurs over the last
        axis.

    Returns
    -------
    AmplificationArray
        The amplification array interpolated from the model
        frequencies into the output frequencies. The frequencies are
        interpolated in log-space.
    """
    interpolator = sp.interpolate.make_interp_spline(
        np.log(model_frequencies), amplification_array, k=1, axis=-1
    )
    return interpolator(np.log(output_frequencies))


def amp_lowpass(
    fftfreq: FrequencyArray, ampf: AmplificationArray, fmin: float, fmidbot: float
) -> None:
    """Lowpass filter site amplification.

    Modifies ``ampf`` array in place setting all amplification below
    ``fmin`` to be ``1.0``, and does not change amplification after
    ``fmidbot``. Between ``fmin`` and ``fmidbot`` the amplification
    array ``ampf`` is introduced logarithmically. This is part of the
    original broadband ground-motion approach described in _[0] and
    modified according to _[1].

    Parameters
    ----------
    fftfreq : FrequencyArray
        Array of frequencies to filter.
    ampf : AmplificationArray
        Array of amplification values.
    fmin : float
        Minimum frequency for amplification.
    fmidbot : float
        Maximum frequency for filter.

    References
    ----------
    [0] Graves, R. W., & Pitarka, A. (2010). Broadband ground-motion
    simulation using a hybrid approach. Bulletin of the Seismological
    Society of America, 100(5A), 2095-2123.
    [1] Lee, R. L., Bradley, B. A., Stafford, P. J., Graves, R. W., &
    Rodriguez-Marek, A. (2022). Hybrid broadband ground-motion
    simulation validation of small magnitude active shallow crustal
    earthquakes in New Zealand. Earthquake Spectra, 38(4), 2548-2579.

    See Also
    --------
    amp_highpass : Highpass filter site amplification.
    """
    ampf[:, fftfreq < fmin] = 1.0
    log_fmin_diff = (np.log(fftfreq) - np.log(fmin)) / (np.log(fmidbot) - np.log(fmin))
    low_frequency_taper_mask = (fftfreq >= fmin) & (fftfreq < fmidbot)
    np.multiply(ampf, log_fmin_diff, out=ampf, where=low_frequency_taper_mask)
    np.add(ampf, 1 - log_fmin_diff, out=ampf, where=low_frequency_taper_mask)


def amp_highpass(
    fftfreq: FrequencyArray, ampf: AmplificationArray, fhightop: float, fmax: float
) -> None:
    """Lowpass filter site amplification.

    Modifies ``ampf`` array in place setting all amplification above
    ``fmax`` to be ``1.0``, and does not change amplification below
    ``fhightop``. Between ``fhightop`` and ``fmax`` the amplification
    array ``ampf`` is reduced logarithmically. This is part of the
    original broadband ground-motion approach described in _[0] and
    modified according to _[1].

    Parameters
    ----------
    fftfreq : FrequencyArray
        Array of frequencies to filter.
    ampf : AmplificationArray
        Array of amplification values.
    fhightop : float
        Minimum frequency for filter.
    fmax : float
        Maximum frequency for amplification.

    References
    ----------
    [0] Graves, R. W., & Pitarka, A. (2010). Broadband ground-motion
    simulation using a hybrid approach. Bulletin of the Seismological
    Society of America, 100(5A), 2095-2123.
    [1] Lee, R. L., Bradley, B. A., Stafford, P. J., Graves, R. W., &
    Rodriguez-Marek, A. (2022). Hybrid broadband ground-motion
    simulation validation of small magnitude active shallow crustal
    earthquakes in New Zealand. Earthquake Spectra, 38(4), 2548-2579.

    See Also
    --------
    amp_lowpass : Lowpass filter site amplification.
    """
    ampf[:, fftfreq >= fmax] = 1.0
    high_fmin_diff = (np.log(fftfreq) - np.log(fhightop)) / (
        np.log(fmax) - np.log(fhightop)
    )
    high_frequency_taper_mask = (fhightop <= fftfreq) & (fftfreq < fmax)
    np.multiply(ampf, 1 - high_fmin_diff, out=ampf, where=high_frequency_taper_mask)
    np.add(ampf, high_fmin_diff, out=ampf, where=high_frequency_taper_mask)
