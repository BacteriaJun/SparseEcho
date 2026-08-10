from sparseecho import ApertureBudget

budget = ApertureBudget(local_bits=8, modeled_shell_order=3, leakage_budget=1e-5)
print("maximum residual phase span:", budget.max_phase_span_cycles(), "cycles/view")
print("at 120 Hz residual Doppler:", budget.max_view_seconds(120), "seconds/view")
