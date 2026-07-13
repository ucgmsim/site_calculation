"""Amplification models for simulated sites."""

import typing
from typing import TYPE_CHECKING, Any

import numpy as np
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
) -> AmplificationArray:  # pragma: no cover
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
        return _utils._campbell_bozorgnia_2014(
            np.ascontiguousarray(vs30),
            np.ascontiguousarray(vs30_sim),
            np.ascontiguousarray(pga),
        )
    except TypeError as e:
        e.add_note("All arrays must have float64 dtype.")
        raise


def bayless_abrahamson_2018(
    vs30: ValueArray, vs30_sim: ValueArray, pga: ValueArray
) -> AmplificationArray:  # pragma: no cover
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

    try:
        return _utils._bayless_abrahamson_2018_eas(
            np.ascontiguousarray(vs30),
            np.ascontiguousarray(vs30_sim),
            np.ascontiguousarray(pga),
        )
    except TypeError as e:
        e.add_note("All arrays must have float64 dtype.")
        raise


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
        # Create a Hanning window for the taper, ensuring it's the same dtype as the waveform
        hanning_window = np.hanning(ntap * 2 + 1)[ntap + 1 :].astype(waveform.dtype)
        waveform[..., nt - ntap :] *= hanning_window


def amplify_waveform(
    waveform: WaveformArray,
    amplification_factor: AmplificationArray,
    n_fft: int,
) -> np.ndarray:
    """Apply amplification factor to waveforms.

    Parameters
    ----------
    waveform : np.ndarray
        The input waveform.
    amplification_factor : np.ndarray
        The frequency amplification factors, sampled at the FFT output
        frequencies ``np.fft.rfftfreq(n_fft, dt)``. Must have
        ``n_fft // 2 + 1`` values along the last axis.
    n_fft : int
        The FFT length to pad out to.

    Returns
    -------
    np.ndarray
        The input waveform amplified at frequencies according to
        the values of `amplification_factor`.
    """
    nt = waveform.shape[-1]
    if amplification_factor.shape[-1] != n_fft // 2 + 1:
        raise ValueError(
            "amplification_factor must have n_fft // 2 + 1 frequency values."
        )
    if waveform.ndim > 1 and waveform.shape[0] != amplification_factor.shape[0]:
        raise ValueError(
            "The number of stations in waveform and amplification_factor must match."
        )

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
        interpolated in log-space. Output frequencies outside the
        model's frequency range (including the DC frequency) are
        clamped to the nearest model frequency.
    """
    if amplification_array.shape[-1] != len(model_frequencies):
        raise ValueError(
            f"The last dimension of amplification_array ({amplification_array.shape[-1]}) "
            f"must match the number of model_frequencies ({len(model_frequencies)})."
        )
    log_model_frequencies = np.log(model_frequencies)
    interpolator = sp.interpolate.make_interp_spline(
        log_model_frequencies, amplification_array, k=1, axis=-1
    )
    # log(0) = -inf at the DC frequency; clamping maps it (and any
    # frequency beyond the model's range) to the nearest endpoint
    # rather than extrapolating.
    with np.errstate(divide="ignore"):
        log_output_frequencies = np.log(output_frequencies)
    return interpolator(
        np.clip(
            log_output_frequencies,
            log_model_frequencies[0],
            log_model_frequencies[-1],
        )
    )


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
    if fmin < 1e-6:
        raise ValueError("Lowpass requires fmin > 0.")
    if fmidbot <= fmin:
        raise ValueError("Lowpass requires fmidbot > fmin.")
    if fftfreq.size != ampf.shape[-1]:
        raise ValueError("fftfreq and ampf must have the same number of frequencies.")
    ampf[:, fftfreq < fmin] = 1.0
    low_frequency_taper_mask = (fftfreq >= fmin) & (fftfreq < fmidbot)
    log_fmin_diff = (np.log(fftfreq[low_frequency_taper_mask]) - np.log(fmin)) / (
        np.log(fmidbot) - np.log(fmin)
    )
    ampf[:, low_frequency_taper_mask] = 1.0 + log_fmin_diff * (
        ampf[:, low_frequency_taper_mask] - 1.0
    )


def amp_highpass(
    fftfreq: FrequencyArray, ampf: AmplificationArray, fhightop: float, fmax: float
) -> None:
    """Highpass filter site amplification.

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
    if fhightop < 1e-6:
        raise ValueError("Highpass requires fhightop > 0.")
    if fmax <= fhightop:
        raise ValueError("Highpass requires fmax > fhightop.")
    if fftfreq.size != ampf.shape[-1]:
        raise ValueError("fftfreq and ampf must have the same number of frequencies.")
    ampf[:, fftfreq >= fmax] = 1.0
    high_frequency_taper_mask = (fhightop <= fftfreq) & (fftfreq < fmax)
    high_fmin_diff = (np.log(fftfreq[high_frequency_taper_mask]) - np.log(fhightop)) / (
        np.log(fmax) - np.log(fhightop)
    )
    ampf[:, high_frequency_taper_mask] = (
        ampf[:, high_frequency_taper_mask] * (1.0 - high_fmin_diff) + high_fmin_diff
    )
