# Site Calculation

This repo implements the basics of the site calculation package. This incorporates both the Bayless 2018 model *and* the Campbell and Borzognia 2014 model. It is intended to replace the qcore modules for the same task, as well as being a future home for site calculations including things like backarc modules, kappa0 models, etc in the future. This repo incorporates the lessons of previous designs of this code including:

1. **Copious documentation**. Prior implementations of site amplification models and the amplification process were not well documented. I have tried very hard to pin the equations in the rust code `src-rust/bayless_abrahamson_2018.rs` and `src-rust/campbell_bozorgnia_2014.rs` to equations in the paper and document changes in the models per Felipe's thesis.
2. **Rust over numba for complex high-performance code**. The benefits of rust code include: speed, correctness, existing library of high-performance code. Numba is great for small blocks, but for code we intend to maintain long-term rust's usability beats Numba's conveniences.
3. **Single-threaded as fast as possible**. Implementing multi-threaded calculations within a library is annoying for downstream application users because (a) the multi-threading model may not be compatible with your model forcing you to turn it off (e.g. NZGMDB), and (b) it can muck up the environment variables of the caller. Instead the code is designed to be run on a single-thread as fast as possible using optimised rust and numpy code. This enables the caller to multi-threading how they like without worrying about extra threads spawned by this library. 

## Example
```python
import numpy as np
from site_calculation.amplification import (
    bayless_abrahamson_2018,
    interpolate_frequencies,
    amp_lowpass,
    amp_highpass,
    amplify_waveform,
    BAYLESS_ABRAHAMSON_2018_FREQUENCIES,
)

# Compute amplification factors for 2 stations
vs30 = np.array([400.0, 760.0])
vs30_sim = np.array([500.0, 500.0])
pga = np.array([0.1, 0.05])
amp = bayless_abrahamson_2018(vs30, vs30_sim, pga)  # (2, n_freqs)

# Interpolate to FFT frequencies
dt = 0.005
n_fft = 2048
fft_freqs = np.fft.rfftfreq(n_fft, dt)
amp_interp = interpolate_frequencies(
    BAYLESS_ABRAHAMSON_2018_FREQUENCIES, fft_freqs, amp
)

# Apply lowpass / highpass filters
amp_lowpass(fft_freqs, amp_interp, fmin=0.1, fmidbot=0.5)
amp_highpass(fft_freqs, amp_interp, fhightop=15.0, fmax=25.0)
# amp_interp is now our filter!


# Amplify a waveform
waveform = np.random.randn(2, 2000).astype(np.float32)
result = amplify_waveform(waveform, amp_interp, n_fft)
```

## Repo structure

```
site_calculation/          # Python package
    __init__.py
    amplification.py       # Amplification models & signal processing
    _utils.pyi             # Type stubs for the rust extension module
src-rust/                  # Rust extension (PyO3)
    lib.rs
    campbell_bozorgnia_2014.rs
    bayless_abrahamson_2018.rs
tests/                     # Python test suite
    resources/             # Test data (CSV files)
    test_amplification.py
data/                      # Reference / validation data
Cargo.toml                 # Rust build configuration
pyproject.toml             # Python build & tool configuration
build.rs                   # Rust build script
```

