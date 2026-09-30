"""Tests for site_calculation.amplification."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from site_calculation.amplification import (
    BAYLESS_ABRAHAMSON_2018_FREQUENCIES,
    CAMPBELL_BOZORGNIA_2014_FREQUENCIES,
    Band,
    _validate_inputs,
    amp_highpass,
    amp_lowpass,
    amplify_waveform,
    bwfilter,
    interpolate_frequencies,
    taper,
)


class TestValidateInputs:
    """Tests for _validate_inputs."""

    def test_non_positive_vs30_raises(self) -> None:
        """A non-positive vs30 value raises ValueError."""
        vs30 = np.array([0.0, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.1])
        with pytest.raises(
            ValueError, match="vs30 and vs30_sim must be strictly positive"
        ):
            _validate_inputs(vs30, vs30_sim, pga)

    def test_non_positive_vs30_sim_raises(self) -> None:
        """A non-positive vs30_sim value raises ValueError."""
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([-1.0, 500.0])
        pga = np.array([0.1, 0.1])
        with pytest.raises(
            ValueError, match="vs30 and vs30_sim must be strictly positive"
        ):
            _validate_inputs(vs30, vs30_sim, pga)

    def test_negative_pga_raises(self) -> None:
        """A negative pga value raises ValueError."""
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, -0.05])
        with pytest.raises(ValueError, match="pga must be non-negative"):
            _validate_inputs(vs30, vs30_sim, pga)

    def test_valid_inputs_passes(self) -> None:
        """Valid inputs do not raise."""
        vs30 = np.array([400.0, 760.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.05])
        _validate_inputs(vs30, vs30_sim, pga)  # no error


class TestTaper:
    """Tests for taper."""

    def test_zero_percent_does_nothing(self) -> None:
        """A zero taper quantile leaves the waveform unchanged."""
        waveform = np.ones((3, 100), dtype=np.float32)
        original = waveform.copy()
        taper(waveform, 0.0)
        assert waveform == pytest.approx(original)

    def test_taper_applied_to_last_portion(self) -> None:
        """The taper is applied only to the trailing portion of the waveform."""
        waveform = np.ones((1, 100), dtype=np.float32)
        taper(waveform, 0.1)
        # Last 10 values should be tapered
        assert np.all(waveform[0, :90] == 1.0)
        assert np.all(waveform[0, 90:] < 1.0)
        assert waveform[0, -1] == 0.0

    def test_taper_multi_channel(self) -> None:
        """The taper is applied independently to each channel."""
        waveform = np.ones((3, 100), dtype=np.float32)
        taper(waveform, 0.1)
        for i in range(3):
            assert np.all(waveform[i, :90] == 1.0)
            assert np.all(waveform[i, 90:] < 1.0)
            assert waveform[i, -1] == 0.0

    def test_preserves_dtype(self) -> None:
        """The waveform dtype is preserved after tapering."""
        waveform = np.ones((1, 100), dtype=np.float64)
        taper(waveform, 0.1)  # ty: ignore[invalid-argument-type]
        assert waveform.dtype == np.float64


class TestAmplifyWaveform:
    """Tests for amplify_waveform."""

    def test_shape_mismatch_raises(self) -> None:
        """An amplification array with the wrong frequency count raises."""
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="n_fft // 2 \\+ 1"):
            amplify_waveform(waveform, amp, n_fft=256)

    def test_n_fft_non_positive_raises(self) -> None:
        """A zero n_fft raises ValueError."""
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 65), dtype=np.float64)
        with pytest.raises(ValueError, match="n_fft must be a positive integer"):
            amplify_waveform(waveform, amp, n_fft=0)

    def test_n_fft_negative_raises(self) -> None:
        """A negative n_fft raises ValueError."""
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 65), dtype=np.float64)
        with pytest.raises(ValueError, match="n_fft must be a positive integer"):
            amplify_waveform(waveform, amp, n_fft=-1)

    def test_n_fft_shorter_than_waveform_raises(self) -> None:
        """An n_fft shorter than the waveform length raises ValueError."""
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 51), dtype=np.float64)
        with pytest.raises(ValueError, match="n_fft must be >= waveform length"):
            amplify_waveform(waveform, amp, n_fft=99)

    def test_station_count_mismatch_raises(self) -> None:
        """A mismatched number of stations between waveform and amp raises."""
        waveform = np.ones((2, 100), dtype=np.float32)
        amp = np.ones((3, 65), dtype=np.float64)
        with pytest.raises(ValueError, match="number of stations"):
            amplify_waveform(waveform, amp, n_fft=128)

    def test_output_shape(self) -> None:
        """The output shape matches the input waveform shape."""
        waveform = np.ones((1, 100), dtype=np.float32)
        n_fft = 128
        amp = np.ones((1, n_fft // 2 + 1), dtype=np.float64)
        result = amplify_waveform(waveform, amp, n_fft)
        assert result.shape == waveform.shape

    def test_multi_channel(self) -> None:
        """The output shape matches the input waveform shape for multiple channels."""
        waveform = np.ones((3, 100), dtype=np.float32)
        n_fft = 128
        amp = np.ones((3, n_fft // 2 + 1), dtype=np.float64)
        result = amplify_waveform(waveform, amp, n_fft)
        assert result.shape == waveform.shape


class TestInterpolateFrequencies:
    """Tests for interpolate_frequencies."""

    def test_interpolation_at_model_frequencies(self) -> None:
        """Interpolating at the model frequencies returns the model values."""
        model_freqs = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        amp = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = interpolate_frequencies(model_freqs, model_freqs, amp)
        assert result == pytest.approx(amp)

    def test_interpolation_between_frequencies(self) -> None:
        """Interpolation between model frequencies is linear in log-frequency space."""
        model_freqs = np.array([1.0, 10.0])
        amp = np.array([[1.0, 10.0]])
        output_freqs = np.array([1.0, 3.16227766, 10.0])  # log-spaced
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        # In log space: log(1)=0, log(10)=2.3026
        # log(3.1623)=1.1513, which is halfway
        expected = np.array([[1.0, 5.5, 10.0]])
        assert result == pytest.approx(expected, abs=5e-6)

    def test_clamps_below_model_range(self) -> None:
        """Frequencies below the model range are clamped to the lowest model frequency."""
        model_freqs = np.array([1.0, 10.0])
        amp = np.array([[5.0, 10.0]])
        output_freqs = np.array([0.0, 0.5])
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        # Both clamped to 1.0 Hz -> amp = 5.0
        expected = np.array([[5.0, 5.0]])
        assert result == pytest.approx(expected)

    def test_clamps_above_model_range(self) -> None:
        """Frequencies above the model range are clamped to the highest model frequency."""
        model_freqs = np.array([1.0, 10.0])
        amp = np.array([[5.0, 10.0]])
        output_freqs = np.array([20.0, 50.0])
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        expected = np.array([[10.0, 10.0]])
        assert result == pytest.approx(expected)

    def test_multi_station(self) -> None:
        """Interpolation is applied independently per station."""
        model_freqs = np.array([1.0, 10.0])
        amp = np.array([[1.0, 10.0], [2.0, 20.0]])
        output_freqs = np.array([1.0, 5.0, 10.0])
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        assert result.shape == (2, 3)

    def test_last_dim_mismatch_raises(self) -> None:
        """A mismatch between amp's last dimension and model frequencies raises."""
        model_freqs = np.array([1.0, 10.0])
        amp = np.array([[1.0, 2.0, 3.0]])
        output_freqs = np.array([1.0, 5.0])
        with pytest.raises(ValueError, match="last dimension"):
            interpolate_frequencies(model_freqs, output_freqs, amp)


class TestAmpLowpass:
    """Tests for amp_lowpass."""

    def test_fmin_too_small_raises(self) -> None:
        """A non-positive fmin raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Lowpass requires fmin > 0"):
            amp_lowpass(fftfreq, ampf, fmin=0.0, fmidbot=5.0)

    def test_fmidbot_not_greater_than_fmin_raises(self) -> None:
        """An fmidbot equal to fmin raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Lowpass requires fmidbot > fmin"):
            amp_lowpass(fftfreq, ampf, fmin=5.0, fmidbot=5.0)

    def test_fmidbot_less_than_fmin_raises(self) -> None:
        """An fmidbot less than fmin raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Lowpass requires fmidbot > fmin"):
            amp_lowpass(fftfreq, ampf, fmin=5.0, fmidbot=1.0)

    def test_shape_mismatch_raises(self) -> None:
        """A mismatch between fftfreq and ampf frequency counts raises."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 5))
        with pytest.raises(ValueError, match="same number of frequencies"):
            amp_lowpass(fftfreq, ampf, fmin=0.1, fmidbot=5.0)

    def test_sets_below_fmin_to_one(self) -> None:
        """Frequencies below fmin are set to unity amplification."""
        fftfreq = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]])
        amp_lowpass(fftfreq, ampf, fmin=1.0, fmidbot=5.0)
        expected = np.ones_like(ampf[0, :2])
        assert ampf[0, :2] == pytest.approx(expected)

    def test_above_fmidbot_unchanged(self) -> None:
        """Frequencies above fmidbot are left unchanged."""
        fftfreq = np.array([0.1, 1.0, 5.0, 10.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0]])
        amp_lowpass(fftfreq, ampf, fmin=1.0, fmidbot=5.0)
        assert ampf[0, -1] == pytest.approx(2.0)

    def test_taper_region(self) -> None:
        """Frequencies between fmin and fmidbot are tapered between 1.0 and the original value."""
        fftfreq = np.array([0.1, 1.0, 2.0, 5.0, 10.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]])
        amp_lowpass(fftfreq, ampf, fmin=1.0, fmidbot=5.0)
        # fmin=1.0 -> amp=1.0, fmidbot=5.0 -> amp=2.0, 2.0 is in between
        assert ampf[0, 0] == pytest.approx(1.0)  # below fmin
        assert ampf[0, 1] == pytest.approx(1.0)  # at fmin
        assert 1.0 < ampf[0, 2] < 2.0  # in taper
        assert ampf[0, 3] == pytest.approx(2.0)  # at fmidbot
        assert ampf[0, 4] == pytest.approx(2.0)  # above fmidbot

    def test_multi_station(self) -> None:
        """The lowpass taper is applied independently per station."""
        fftfreq = np.array([0.1, 1.0, 5.0, 10.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
        amp_lowpass(fftfreq, ampf, fmin=1.0, fmidbot=5.0)
        assert ampf.shape == (2, 4)
        np.testing.assert_array_equal(ampf[:, 0], [1.0, 1.0])


class TestAmpHighpass:
    """Tests for amp_highpass."""

    def test_fhightop_too_small_raises(self) -> None:
        """A non-positive fhightop raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Highpass requires fhightop > 0"):
            amp_highpass(fftfreq, ampf, fhightop=0.0, fmax=10.0)

    def test_fmax_not_greater_than_fhightop_raises(self) -> None:
        """An fmax equal to fhightop raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Highpass requires fmax > fhightop"):
            amp_highpass(fftfreq, ampf, fhightop=5.0, fmax=5.0)

    def test_fmax_less_than_fhightop_raises(self) -> None:
        """An fmax less than fhightop raises ValueError."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 3))
        with pytest.raises(ValueError, match="Highpass requires fmax > fhightop"):
            amp_highpass(fftfreq, ampf, fhightop=10.0, fmax=5.0)

    def test_shape_mismatch_raises(self) -> None:
        """A mismatch between fftfreq and ampf frequency counts raises."""
        fftfreq = np.array([0.1, 1.0, 10.0])
        ampf = np.ones((1, 5))
        with pytest.raises(ValueError, match="same number of frequencies"):
            amp_highpass(fftfreq, ampf, fhightop=0.1, fmax=10.0)

    def test_sets_above_fmax_to_one(self) -> None:
        """Frequencies above fmax are set to unity amplification."""
        fftfreq = np.array([0.1, 1.0, 5.0, 10.0, 20.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0, 2.0]])
        amp_highpass(fftfreq, ampf, fhightop=5.0, fmax=10.0)
        assert ampf[0, -1] == pytest.approx(1.0)

    def test_below_fhightop_unchanged(self) -> None:
        """Frequencies below fhightop are left unchanged."""
        fftfreq = np.array([0.1, 1.0, 5.0, 10.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0]])
        amp_highpass(fftfreq, ampf, fhightop=5.0, fmax=10.0)
        assert ampf[0, 0] == 2.0

    def test_taper_region(self) -> None:
        """Frequencies between fhightop and fmax are tapered between the original value and 1.0."""
        fftfreq = np.array([0.1, 1.0, 5.0, 7.0, 10.0, 20.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
        amp_highpass(fftfreq, ampf, fhightop=5.0, fmax=10.0)
        assert ampf[0, 0] == pytest.approx(2.0)  # below fhightop
        assert ampf[0, 2] == pytest.approx(2.0)  # at fhightop
        assert 1.0 < ampf[0, 3] < 2.0  # in taper
        assert ampf[0, 4] == pytest.approx(1.0)  # at fmax
        assert ampf[0, 5] == pytest.approx(1.0)  # above fmax

    def test_multi_station(self) -> None:
        """The highpass taper is applied independently per station."""
        fftfreq = np.array([0.1, 1.0, 5.0, 10.0, 20.0])
        ampf = np.array([[2.0, 2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0, 3.0]])
        amp_highpass(fftfreq, ampf, fhightop=5.0, fmax=10.0)
        assert ampf.shape == (2, 5)
        expected = np.ones_like(ampf[:, -1])
        assert ampf[:, -1] == pytest.approx(expected)


class TestModuleConstants:
    """Verify the module-level frequency arrays are populated.

    This test ensures that the rust library is being linked correctly from the python side.
    """

    def test_campbell_frequencies_populated(self) -> None:
        """The Campbell-Bozorgnia 2014 frequency array is non-empty."""
        assert len(CAMPBELL_BOZORGNIA_2014_FREQUENCIES) > 0

    def test_bayless_frequencies_populated(self) -> None:
        """The Bayless-Abrahamson 2018 frequency array is non-empty."""
        assert len(BAYLESS_ABRAHAMSON_2018_FREQUENCIES) > 0


# Filter behaviour depends only on frequencies relative to the
# sampling rate, so frequencies below are drawn as fractions of the
# sampling frequency. They are kept well below the Nyquist frequency
# so that the bilinear transform's frequency warping is negligible.
dts = st.floats(1e-3, 0.1)
taper_fractions = st.floats(0.005, 0.03)
pass_band_ratios = st.floats(1 / 16, 1 / 4)
stop_band_ratios = st.floats(4.0, 16.0)


def _steady_state_gain(
    frequency: float, dt: float, taper_frequency: float, band: Band
) -> float:
    """Amplitude of ``bwfilter`` applied to a unit sinusoid.

    The amplitude is fitted over the middle half of the signal to
    avoid transients at the edges of the filtered waveform.
    """
    nt = 32768
    t = np.arange(nt) * dt
    filtered = bwfilter(np.sin(2 * np.pi * frequency * t), dt, taper_frequency, band)
    middle = slice(nt // 4, 3 * nt // 4)
    basis = np.stack(
        [
            np.sin(2 * np.pi * frequency * t[middle]),
            np.cos(2 * np.pi * frequency * t[middle]),
        ],
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(basis, filtered[middle])
    return float(np.hypot(*coefficients))


class TestBwfilter:
    """Tests for bwfilter."""

    @settings(deadline=None)
    @given(
        dt=dts,
        taper_fraction=taper_fractions,
        band=st.sampled_from([Band.LOWPASS, Band.HIGHPASS]),
    )
    def test_taper_frequency_attenuation(
        self, dt: float, taper_fraction: float, band: Band
    ) -> None:
        """The amplitude at the taper frequency is 1/sqrt(2)."""
        taper_frequency = taper_fraction / dt
        gain = _steady_state_gain(taper_frequency, dt, taper_frequency, band)
        assert gain == pytest.approx(1 / np.sqrt(2), rel=0.01)

    @settings(deadline=None)
    @given(dt=dts, taper_fraction=taper_fractions, ratio=pass_band_ratios)
    def test_lowpass_passes_low_frequencies(
        self, dt: float, taper_fraction: float, ratio: float
    ) -> None:
        """Lowpass leaves frequencies below the taper frequency intact."""
        taper_frequency = taper_fraction / dt
        gain = _steady_state_gain(
            ratio * taper_frequency, dt, taper_frequency, Band.LOWPASS
        )
        assert gain == pytest.approx(1.0, abs=0.01)

    @settings(deadline=None)
    @given(dt=dts, taper_fraction=taper_fractions, ratio=stop_band_ratios)
    def test_lowpass_removes_high_frequencies(
        self, dt: float, taper_fraction: float, ratio: float
    ) -> None:
        """Lowpass removes frequencies above the taper frequency."""
        taper_frequency = taper_fraction / dt
        gain = _steady_state_gain(
            ratio * taper_frequency, dt, taper_frequency, Band.LOWPASS
        )
        assert gain < 0.01

    @settings(deadline=None)
    @given(dt=dts, taper_fraction=taper_fractions, ratio=stop_band_ratios)
    def test_highpass_passes_high_frequencies(
        self, dt: float, taper_fraction: float, ratio: float
    ) -> None:
        """Highpass leaves frequencies above the taper frequency intact."""
        taper_frequency = taper_fraction / dt
        gain = _steady_state_gain(
            ratio * taper_frequency, dt, taper_frequency, Band.HIGHPASS
        )
        assert gain == pytest.approx(1.0, abs=0.01)

    @settings(deadline=None)
    @given(dt=dts, taper_fraction=taper_fractions, ratio=pass_band_ratios)
    def test_highpass_removes_low_frequencies(
        self, dt: float, taper_fraction: float, ratio: float
    ) -> None:
        """Highpass removes frequencies below the taper frequency."""
        taper_frequency = taper_fraction / dt
        gain = _steady_state_gain(
            ratio * taper_frequency, dt, taper_frequency, Band.HIGHPASS
        )
        assert gain < 0.01
