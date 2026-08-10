from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sparseecho import ApertureBudget
from sparseecho.temporal import doppler_fiber_coefficients, doppler_shell_energies
from sparseecho.transforms import centered_gray_rank, fwht


def main() -> int:
    r = 8
    u = np.arange(1 << r, dtype=np.uint32)
    tau = centered_gray_rank(u, r)
    time_spectrum = fwht(tau, normalize=True)

    direct_errors = {}
    tails = {}
    for span in [0.2, 0.4, 0.8]:
        omega = 2 * np.pi * span / ((1 << r) - 1)
        physical = np.exp(1j * omega * tau)
        direct = fwht(physical, normalize=True)
        exact = doppler_fiber_coefficients(span, r)
        direct_errors[str(span)] = float(np.linalg.norm(direct - exact))
        shell = doppler_shell_energies(span, r)
        tails[str(span)] = {
            "shell_energy": shell.tolist(),
            "tail_above_order_3": float(np.sum(shell[4:])),
        }

    budgets = {}
    for leakage in [1e-3, 1e-4, 1e-5]:
        budget = ApertureBudget(local_bits=r, modeled_shell_order=3, leakage_budget=leakage)
        budgets[str(leakage)] = budget.max_phase_span_cycles()

    payload = {
        "profile": "sfpti-1.0-theory-acceptance",
        "software_version": "1.0",
        "gray_time_nonzero_walsh_coefficients": int(np.count_nonzero(np.abs(time_spectrum) > 1e-12)),
        "expected_minimum_support": r,
        "exact_doppler_factorization_l2_error": direct_errors,
        "shell_tails": tails,
        "max_phase_span_cycles_by_tail_budget": budgets,
    }
    output = Path("results/theory_acceptance.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
