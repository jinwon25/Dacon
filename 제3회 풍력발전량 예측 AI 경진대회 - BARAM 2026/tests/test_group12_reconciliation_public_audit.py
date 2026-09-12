from __future__ import annotations

import numpy as np

from experiments.group12_reconciliation_public_audit import transfer_breakdown


def test_two_group_public_delta_uses_three_halves_multiplier() -> None:
    result = transfer_breakdown(
        {
            "score": 0.0006,
            "one_minus_nmae": 0.00015,
            "ficr": 0.00105,
        },
        {
            "score": -0.0004,
            "one_minus_nmae": 0.0001,
            "ficr": -0.0009,
        },
    )
    assert np.isclose(result["public_pair_delta"]["score"], -0.0006)
    assert result["near_mirror_score_reversal"]
    assert result["sign_agreement"]["one_minus_nmae"]
    assert not result["sign_agreement"]["ficr"]
