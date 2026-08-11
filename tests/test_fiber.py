import numpy as np

from sparseecho.temporal import doppler_fiber_coefficients, doppler_shell_energies
from sparseecho.transforms import centered_gray_rank, fwht


def test_exact_gray_doppler_factorization():
    bits = 8
    span = 0.4
    q = np.arange(1 << bits, dtype=np.uint32)
    tau = centered_gray_rank(q, bits)
    omega = 2.0 * np.pi * span / ((1 << bits) - 1)
    direct = fwht(np.exp(1j * omega * tau), normalize=True)
    analytic = doppler_fiber_coefficients(span, bits)
    np.testing.assert_allclose(direct, analytic, atol=1e-12, rtol=1e-12)
    assert abs(np.sum(doppler_shell_energies(span, bits)) - 1.0) < 1e-12
