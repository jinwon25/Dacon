from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.archive.v211_pitcher_transport_workload_h1 import (
    apply_pitcher_transport,
    pitcher_eb_gate,
    restrictions,
)


def test_pitcher_eb_gate_uses_shrunk_prior_utility() -> None:
    gate, metadata = pitcher_eb_gate(
        np.array([1, 1, 2, 2]),
        np.array([0.2, 0.2, -0.4, -0.4]),
        np.array([1, 2, 3]),
        alpha=0.1,
    )
    assert gate.tolist() == [True, False, False]
    assert metadata["query_unknown_fraction"] == pytest.approx(1.0 / 3.0)


def test_transport_modes_preserve_non_jy_rows() -> None:
    axis = {"domain3": np.array(["R_CORE", "R_CORE", "F"])}
    frame = pd.DataFrame(
        {"num_runners_on": [1, 0, 1], "li": [1.0, 1.0, 1.0]}
    )
    output, active = apply_pitcher_transport(
        np.array([0.4, 0.4, 0.4]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.6, 0.6, 0.6]),
        axis,
        frame,
        np.array([True, True, True]),
        "only_positive",
    )
    assert active.tolist() == [True, False, False]
    assert np.allclose(output, [0.6, 0.4, 0.4])


def test_unknown_transport_mode_is_rejected() -> None:
    axis = {"domain3": np.array(["R_CORE"])}
    frame = pd.DataFrame({"num_runners_on": [1], "li": [1.0]})
    with pytest.raises(ValueError, match="unknown transport mode"):
        apply_pitcher_transport(
            np.array([0.4]), np.array([0.5]), np.array([0.6]), axis, frame,
            np.array([True]), "month_tuned",
        )


def test_v211_gate_is_prior_origin_and_test_batch_free() -> None:
    audit = restrictions()
    assert audit["strictly_prior_origin_pitcher_gate"]
    assert audit["fixed_eb_alphas_and_modes"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
