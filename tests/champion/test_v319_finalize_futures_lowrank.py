import numpy as np
import pandas as pd

from src.champion.v50_low_rank_pitcher_context import fit_source_matrix


def test_final_lookup_matrix_shape_contract() -> None:
    rows = pd.DataFrame({
        "pitcher_id": np.repeat([1, 2], 24),
        "balls_before": np.tile(np.repeat(np.arange(4), 6), 2),
        "strikes_before": np.tile(np.repeat(np.arange(3), 2), 8),
        "batter_hand": np.tile([1, 2], 24),
    })
    target = np.tile([0.0, 1.0], 24)
    model = fit_source_matrix(
        rows, target, np.full(len(rows), 0.5),
        smoothing_grid=(300.0,), rank_grid=(2,),
    )
    assert model["reconstructions"][(300.0, 2)].shape == (2, 24)
