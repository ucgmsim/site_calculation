"""Tests for site_calculation.amplification."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from site_calculation.amplification import (
    BAYLESS_ABRAHAMSON_2018_FREQUENCIES,
    CAMPBELL_BOZORGNIA_2014_FREQUENCIES,
    _validate_inputs,
    amp_highpass,
    amp_lowpass,
    amplify_waveform,
    bayless_abrahamson_2018,
    campbell_bozorgnia_2014,
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

    def test_vs30_below_one_does_not_raise(self) -> None:
        # Only values <= 0 are invalid; small positive values must pass.
        vs30 = np.array([0.5, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.1])
        _validate_inputs(vs30, vs30_sim, pga)  # no error

    def test_vs30_sim_below_one_does_not_raise(self) -> None:
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([0.5, 500.0])
        pga = np.array([0.1, 0.1])
        _validate_inputs(vs30, vs30_sim, pga)  # no error

    def test_vs30_sim_zero_raises(self) -> None:
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([0.0, 500.0])
        pga = np.array([0.1, 0.1])
        with pytest.raises(
            ValueError, match="vs30 and vs30_sim must be strictly positive"
        ):
            _validate_inputs(vs30, vs30_sim, pga)

    def test_pga_zero_does_not_raise(self) -> None:
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.0, 0.1])
        _validate_inputs(vs30, vs30_sim, pga)  # no error

    def test_vs30_error_message_exact(self) -> None:
        vs30 = np.array([0.0])
        vs30_sim = np.array([500.0])
        pga = np.array([0.1])
        with pytest.raises(ValueError) as exc_info:
            _validate_inputs(vs30, vs30_sim, pga)
        assert str(exc_info.value) == "vs30 and vs30_sim must be strictly positive."

    def test_pga_error_message_exact(self) -> None:
        vs30 = np.array([400.0])
        vs30_sim = np.array([500.0])
        pga = np.array([-0.1])
        with pytest.raises(ValueError) as exc_info:
            _validate_inputs(vs30, vs30_sim, pga)
        assert str(exc_info.value) == "pga must be non-negative."


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

    def test_ntap_of_exactly_one_still_tapers(self) -> None:
        # nt * taper_quantile == 1.0 exactly (both binary-exact floats).
        waveform = np.ones((1, 8), dtype=np.float32)
        taper(waveform, 0.125)
        assert waveform[0, -1] == 0.0
        assert np.all(waveform[0, :-1] == 1.0)

    def test_multiplies_rather_than_overwrites_amplitude(self) -> None:
        waveform = np.full((1, 100), 2.0, dtype=np.float32)
        taper(waveform, 0.1)
        ntap = 10
        window = np.hanning(ntap * 2 + 1)[ntap + 1 :].astype(np.float32)
        expected = 2.0 * window
        assert waveform[0, 90:] == pytest.approx(expected, rel=1e-5)

    def test_quantile_above_one_raises(self) -> None:
        """A taper_quantile greater than 1 raises a clear ValueError."""
        waveform = np.ones((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="taper_quantile"):
            taper(waveform, 1.5)

    def test_negative_quantile_raises(self) -> None:
        """A negative taper_quantile raises a clear ValueError instead of silently doing nothing."""
        waveform = np.ones((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="taper_quantile"):
            taper(waveform, -0.1)


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

    def test_1d_waveform_with_2d_amp_raises(self) -> None:
        waveform: np.ndarray = np.ones(100, dtype=np.float32)
        amp = np.ones((1, 65), dtype=np.float64)
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

    def test_1d_waveform_skips_station_check(self) -> None:
        # A 1D waveform has no station axis, so amplification_factor's
        # leading dimension is the frequency axis, not a station count,
        # and must not be compared against waveform.shape[0].
        waveform = np.ones(100, dtype=np.float32)
        amp = np.ones(65, dtype=np.float64)
        # Deliberately outside the 2D type hint: 1D input is a supported code path.
        result = amplify_waveform(waveform, amp, n_fft=128)  # ty: ignore[invalid-argument-type]
        assert result.shape == waveform.shape

    def test_n_fft_equal_to_waveform_length_does_not_raise(self) -> None:
        waveform = np.ones((1, 100), dtype=np.float32)
        n_fft = 100
        amp = np.ones((1, n_fft // 2 + 1), dtype=np.float64)
        result = amplify_waveform(waveform, amp, n_fft)
        assert result.shape == waveform.shape

    def test_minimal_length_waveform_does_not_raise(self) -> None:
        waveform = np.ones((1, 1), dtype=np.float32)
        amp = np.ones((1, 1), dtype=np.float64)
        result = amplify_waveform(waveform, amp, n_fft=1)
        assert result.shape == waveform.shape

    def test_odd_n_fft_uses_floor_division_for_frequency_count(self) -> None:
        # n_fft // 2 + 1 (int) must be compared, not n_fft / 2 + 1 (float).
        waveform = np.ones((1, 50), dtype=np.float32)
        n_fft = 101
        amp = np.ones((1, n_fft // 2 + 1), dtype=np.float64)
        result = amplify_waveform(waveform, amp, n_fft)
        assert result.shape == waveform.shape

    def test_amplification_multiplies_frequency_content(self) -> None:
        rng = np.random.default_rng(0)
        waveform = rng.standard_normal((1, 64)).astype(np.float32)
        n_fft = 64
        amp = np.full((1, n_fft // 2 + 1), 2.0)
        result = amplify_waveform(waveform, amp, n_fft)
        assert result == pytest.approx(2.0 * waveform, rel=1e-4)

    def test_n_fft_error_message_exact(self) -> None:
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 65), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            amplify_waveform(waveform, amp, n_fft=0)
        assert str(exc_info.value) == "n_fft must be a positive integer."

    def test_n_fft_length_error_message_exact(self) -> None:
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 51), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            amplify_waveform(waveform, amp, n_fft=99)
        assert str(exc_info.value) == "n_fft must be >= waveform length."

    def test_shape_mismatch_error_message_exact(self) -> None:
        waveform = np.ones((1, 100), dtype=np.float32)
        amp = np.ones((1, 10), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            amplify_waveform(waveform, amp, n_fft=256)
        assert (
            str(exc_info.value)
            == "amplification_factor must have n_fft // 2 + 1 frequency values."
        )

    def test_station_count_mismatch_error_message_exact(self) -> None:
        waveform = np.ones((2, 100), dtype=np.float32)
        amp = np.ones((3, 65), dtype=np.float64)
        with pytest.raises(ValueError) as exc_info:
            amplify_waveform(waveform, amp, n_fft=128)
        assert (
            str(exc_info.value)
            == "The number of stations in waveform and amplification_factor must match."
        )

    def test_unit_amplification_is_identity(self) -> None:
        """A unit amplification factor returns the original waveform in float32."""
        rng = np.random.default_rng(1)
        waveform = rng.standard_normal((2, 1000)).astype(np.float32)
        n_fft = 1024
        amp = np.ones((2, n_fft // 2 + 1))
        result = amplify_waveform(waveform, amp, n_fft)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, waveform, rtol=0, atol=1e-5)


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

    def test_matches_np_interp_on_real_model_frequencies(self) -> None:
        """Matches a per-station np.interp in log-frequency, including DC and out-of-range frequencies."""
        model_freqs = CAMPBELL_BOZORGNIA_2014_FREQUENCIES
        rng = np.random.default_rng(0)
        amp = rng.uniform(0.2, 5.0, (4, len(model_freqs)))
        output_freqs = np.concatenate(
            [
                np.fft.rfftfreq(4096, 0.005),
                [model_freqs[0] / 10, model_freqs[-1] * 10, 1e300],
                model_freqs,
            ]
        )
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        with np.errstate(divide="ignore"):
            log_output = np.log(output_freqs)
        expected = np.array(
            [np.interp(log_output, np.log(model_freqs), row) for row in amp]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
        # Model frequencies reproduce the model values (to rounding).
        np.testing.assert_allclose(
            result[:, -len(model_freqs) :], amp, rtol=1e-15, atol=0
        )

    @settings(deadline=None)
    @given(
        log_model_frequencies=hnp.arrays(
            np.float64,
            st.integers(2, 20),
            elements=st.floats(-5, 5),
            unique=True,
        ).map(np.sort),
        data=st.data(),
    )
    def test_property_matches_np_interp(
        self, log_model_frequencies: np.ndarray, data: st.DataObject
    ) -> None:
        """Log-frequency linear interpolation with end clamping agrees with np.interp."""
        model_freqs = np.exp(log_model_frequencies)
        if np.any(np.diff(np.log(model_freqs)) <= 0):
            return  # exp/log round-trip collapsed two nearby frequencies
        n_stations = data.draw(st.integers(1, 3))
        amp = data.draw(
            hnp.arrays(
                np.float64,
                (n_stations, len(model_freqs)),
                elements=st.floats(-1e3, 1e3),
            )
        )
        output_freqs = data.draw(
            hnp.arrays(
                np.float64,
                st.integers(1, 30),
                elements=st.one_of(
                    st.just(0.0),
                    st.floats(1e-4, 1e4),
                    st.sampled_from(model_freqs.tolist()),
                ),
            )
        )
        result = interpolate_frequencies(model_freqs, output_freqs, amp)
        with np.errstate(divide="ignore"):
            log_output = np.log(output_freqs)
        expected = np.array(
            [np.interp(log_output, np.log(model_freqs), row) for row in amp]
        )
        assert result.shape == (n_stations, len(output_freqs))
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-9)

    def test_unsorted_model_frequencies_raises(self) -> None:
        """Model frequencies must be strictly increasing."""
        model_freqs = np.array([1.0, 10.0, 5.0])
        amp = np.ones((1, 3))
        with pytest.raises(ValueError, match="strictly increasing"):
            interpolate_frequencies(model_freqs, np.array([2.0]), amp)

    def test_single_model_frequency_raises(self) -> None:
        """At least two model frequencies are required to interpolate."""
        with pytest.raises(ValueError, match="At least two"):
            interpolate_frequencies(np.array([1.0]), np.array([2.0]), np.ones((1, 1)))

    def test_non_finite_amplification_raises(self) -> None:
        """Non-finite amplification values are rejected."""
        amp = np.array([[1.0, np.nan]])
        with pytest.raises(ValueError, match="finite"):
            interpolate_frequencies(np.array([1.0, 10.0]), np.array([2.0]), amp)

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


class TestCampbellBozorgnia2014:
    """Tests for campbell_bozorgnia_2014."""

    def test_output_shape_and_finiteness(self) -> None:
        """A valid call returns finite values shaped by frequencies."""
        vs30 = np.array([400.0, 760.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.05])
        result = campbell_bozorgnia_2014(vs30, vs30_sim, pga)
        assert result.shape == (2, len(CAMPBELL_BOZORGNIA_2014_FREQUENCIES))
        assert np.all(np.isfinite(result))

    def test_validates_inputs(self) -> None:
        """Invalid inputs raise ValueError via _validate_inputs."""
        vs30 = np.array([0.0, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.1])
        with pytest.raises(
            ValueError, match="vs30 and vs30_sim must be strictly positive"
        ):
            campbell_bozorgnia_2014(vs30, vs30_sim, pga)

    def test_non_float64_dtype_raises_typeerror_with_note(self) -> None:
        """Non-float64 dtype inputs raise a TypeError with an explanatory note."""
        vs30 = np.array([400], dtype=np.int64)
        vs30_sim = np.array([500], dtype=np.int64)
        pga = np.array([0.1])
        with pytest.raises(TypeError) as exc_info:
            campbell_bozorgnia_2014(vs30, vs30_sim, pga)  # ty: ignore[invalid-argument-type]
        assert any(
            "All arrays must have float64 dtype." in note
            for note in exc_info.value.__notes__
        )


class TestBaylessAbrahamson2018:
    """Tests for bayless_abrahamson_2018."""

    def test_output_shape_and_finiteness(self) -> None:
        """A valid call returns finite values shaped by frequencies."""
        vs30 = np.array([400.0, 760.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, 0.05])
        result = bayless_abrahamson_2018(vs30, vs30_sim, pga)
        assert result.shape == (2, len(BAYLESS_ABRAHAMSON_2018_FREQUENCIES))
        assert np.all(np.isfinite(result))

    def test_validates_inputs(self) -> None:
        """Invalid inputs raise ValueError via _validate_inputs."""
        vs30 = np.array([400.0, 400.0])
        vs30_sim = np.array([500.0, 500.0])
        pga = np.array([0.1, -0.05])
        with pytest.raises(ValueError, match="pga must be non-negative"):
            bayless_abrahamson_2018(vs30, vs30_sim, pga)

    def test_non_float64_dtype_raises_typeerror_with_note(self) -> None:
        """Non-float64 dtype inputs raise a TypeError with an explanatory note."""
        vs30 = np.array([400], dtype=np.int64)
        vs30_sim = np.array([500], dtype=np.int64)
        pga = np.array([0.1])
        with pytest.raises(TypeError) as exc_info:
            bayless_abrahamson_2018(vs30, vs30_sim, pga)  # ty: ignore[invalid-argument-type]
        assert any(
            "All arrays must have float64 dtype." in note
            for note in exc_info.value.__notes__
        )


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
