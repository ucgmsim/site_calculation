"""Amplification models for simulated sites."""

import contextlib
import multiprocessing
import typing
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyfftw
import pyfftw.config as _pyfftw_config
import pyfftw.interfaces.numpy_fft as pyfftw_fft
import scipy as sp

from site_calculation import _utils  # type: ignore[unresolved-import]

if TYPE_CHECKING:
    pyfftw_config = typing.cast(Any, _pyfftw_config)
else:
    pyfftw_config = _pyfftw_config

AmplificationArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
WaveformArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FrequencyArray = np.ndarray[tuple[int,], np.dtype[np.float64]]
ValueArray = np.ndarray[tuple[int,], np.dtype[np.float64]]
Filter = ValueArray


REQUIRED_COLUMNS = {"vs30", "vs30_sim", "pga"}

CAMPBELL_BOZORGNIA_2014_FREQUENCIES: FrequencyArray = (
    _utils._campbell_bozorgnia_2014_frequencies()
)
BAYLESS_ABRAHAMSON_2018_FREQUENCIES: FrequencyArray = (
    _utils._bayless_abrahamson_2018_frequencies()
)


def campbell_bozorgnia_2014(
    vs30: ValueArray, vs30_sim: ValueArray, pga: ValueArray
) -> AmplificationArray:
    """Site amplification factor based on the Campbell and Bozorgnia
    2014 spectral acceleration ground motion model [0] with
    modifications[1].

    Parameters
    ----------
    vs30 : ValueArray
        Vs30 values for station, the target for site amplification.
    vs30_sim : ValueArray
        Vs30 values in simulation.
    pga : ValueArray
        Peak ground acceleration measured in un-amplified array.

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

    try:
        return _utils._campbell_bozorgnia_2014(vs30, vs30_sim, pga)
    except TypeError as e:
        e.add_note("All arrays must have float64 dtype.")
        raise


def bayless_abrahamson_2018(
    vs30: ValueArray, vs30_sim: ValueArray, pga: ValueArray
) -> AmplificationArray:
    """Site amplification factor based on the Bayless and Abrahamson
    2018 effective amplitude spectra ground motion model [0] with
    modifications[1].

    Parameters
    ----------
    vs30 : ValueArray
        Vs30 values for station, the target for site amplification.
    vs30_sim : ValueArray
        Vs30 values in simulation.
    pga : ValueArray
        Peak ground acceleration measured in un-amplified array.

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

    return _utils._bayless_abrahamson_2018_eas(vs30, vs30_sim, pga)


@contextlib.contextmanager
def _pyfftw_cores(cores: int) -> Generator[None, None, None]:
    old_cores = pyfftw_config.NUM_THREADS
    pyfftw_config.NUM_THREADS = cores
    try:
        yield
    finally:
        pyfftw_config.NUM_THREADS = old_cores


def taper(waveform: WaveformArray, taper_percent: float) -> None:
    """Taper the end of a waveform using the Hanning method.

    Parameters
    ----------
    waveform : WaveformArray
        The input waveform.
    taper_percent : float
        The taper percentage. The last ``taper_percent * nt`` values
        should taper to 0.0.

    See Also
    --------
    np.hanning : The hanning taper used.
    """
    nt = waveform.shape[-1]
    ntap = int(nt * taper_percent)
    if ntap > 0:
        # Create a Hanning window for the taper, ensuring it's float32
        hanning_window = np.hanning(ntap * 2 + 1)[ntap + 1 :].astype(waveform.dtype)
        # Create a copy of the original waveform so-as not to modify it in-place.
        waveform[..., nt - ntap :] *= hanning_window


def amplify_waveform(
    waveform: WaveformArray,
    amplification_factor: AmplificationArray,
    n_fft: int,
    cores: int = multiprocessing.cpu_count(),
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
    n_fft : int
        The FFT length to pad out to.
    cores : int, optional
        The number of cores to use for FFT. Defaults to all cores
        available on the system as reported by
        `muliprocessing.cpu_count()`.

    Returns
    -------
    np.ndarray
        The input waveform (de)amplified at frequencies according to
        the values of `amplification_factor`.
    """
    nt = waveform.shape[-1]

    with _pyfftw_cores(cores):
        fourier = pyfftw_fft.rfft(waveform, n=n_fft, axis=-1)

        fourier *= amplification_factor.astype(waveform.dtype)

        result_full = pyfftw_fft.irfft(fourier, n=n_fft, axis=-1)

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
