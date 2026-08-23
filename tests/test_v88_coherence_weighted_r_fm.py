from __future__ import annotations

import numpy as np
import pandas as pd

from src.v88_coherence_weighted_r_fm import coherence_correction, select_policy


def test_coherence_correction_requires_sign_and_magnitude_agreement() -> None:
    older = np.asarray([0.2, -0.2, 0.1, 0.0])
    recent = np.asarray([0.1, -0.4, -0.2, 0.2])
    linear, agree, weight = coherence_correction(
        older, recent, "coherence_linear"
    )
    assert agree.tolist() == [True, True, False, False]
    assert np.allclose(weight, [0.5, 0.5, 0.5, 0.0])
    assert np.allclose(linear, [0.075, -0.15, 0.0, 0.0])
    squared, _, _ = coherence_correction(
        older, recent, "coherence_squared"
    )
    assert np.all(np.abs(squared) <= np.abs(linear))


def test_selector_prefers_robust_source_passer() -> None:
    rows = []
    for policy, gain, month, worst in (
        ("coherence_sqrt", 2.0, 0.50, -1.0),
        ("coherence_linear", 1.0, 0.75, 0.2),
        ("coherence_squared", 0.5, 1.0, 0.1),
    ):
        for axis in ("full_2022", "late_2023"):
            rows.append(
                {
                    "policy": policy,
                    "axis": axis,
                    "gain": gain,
                    "positive_month_fraction": month,
                    "worst_month_gain": worst,
                    "minimum_domain_gain": 0.0,
                    "applied_domain_gain": gain,
                }
            )
    chosen, ranking = select_policy(pd.DataFrame(rows))
    assert chosen == "coherence_linear"
    assert bool(ranking.set_index("policy").loc[chosen, "source_gate_passed"])
