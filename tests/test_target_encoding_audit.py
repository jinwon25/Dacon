import numpy as np
import pandas as pd

from src.target_encoding_audit import audit


def test_target_encoding_audit_detects_order_and_future_effects():
    n = 8
    frame = pd.DataFrame({
        "balls_before": [0] * n,
        "strikes_before": [0] * n,
        "pitcher_hand": [1] * n,
        "batter_hand": [1] * n,
        "control_success": [0, 1, 1, 0, 1, 0, 1, 0],
    })
    result = audit(frame, max_rows=n)
    values = result.set_index("audit")["value"]
    assert values["row_permutation_sensitivity_max_abs"] > 0
    assert values["future_row_effect_of_current_target_flip"] > 0
    assert np.isfinite(values["full_frame_global_prior_self_future_shift"])
