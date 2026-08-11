import numpy as np

from sparseecho.temporal import apply_fir_memory, deconvolve_fir_memory


def test_continuous_fir_inverse_across_view_boundary():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(512, 2)) + 1j * rng.normal(size=(512, 2))
    taps = (0.90, 0.10)
    y = apply_fir_memory(x, taps)
    recovered = deconvolve_fir_memory(y, taps)
    np.testing.assert_allclose(recovered, x, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(recovered[256], x[256], atol=1e-10, rtol=1e-10)
