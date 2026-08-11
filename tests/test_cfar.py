import numpy as np

from sparseecho.recovery import GrayFiberViewDecoder
from sparseecho.transforms import gray_order


def test_cfar_null_does_not_saturate_proposal_budget():
    rng = np.random.default_rng(12)
    decoder = GrayFiberViewDecoder(local_bits=6, max_components=16, phase_grid=9, query_order=gray_order(6))
    counts = []
    for _ in range(20):
        noise = (rng.normal(size=(64, 4)) + 1j * rng.normal(size=(64, 4))) / np.sqrt(2.0)
        counts.append(len(decoder.decode(noise).components))
    assert max(counts) < 16
