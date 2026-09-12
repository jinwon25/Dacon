from __future__ import annotations

import pandas as pd

from experiments.kma_um_meta_risk_gate import build_kma_risk_features


def test_build_kma_risk_features_adds_model_disagreement(tmp_path) -> None:
    key = {
        "forecast_kst_dtm": ["2024-01-01 01:00", "2024-01-01 02:00"],
        "data_available_kst_dtm": ["2023-12-31 13:00"] * 2,
    }
    context = pd.DataFrame(
        {
            **key,
            "kma_um_ctx_u10_r0": [3.0, 4.0],
            "kma_um_ctx_v10_r0": [4.0, 5.0],
            "kma_um_ctx_speed10_r0": [5.0, 6.403124],
            "kma_um_ctx_u850_r0": [5.0, 6.0],
            "kma_um_ctx_v850_r0": [6.0, 7.0],
            "kma_um_ctx_speed850_r0": [7.81025, 9.21954],
            "kma_um_ctx_u700_r0": [7.0, 8.0],
            "kma_um_ctx_v700_r0": [8.0, 9.0],
            "kma_um_ctx_speed700_r0": [10.6301, 12.0416],
            "kma_um_ctx_shear10_850_r0": [1.0, 2.0],
            "kma_um_ctx_run_change10": [0.5, 0.7],
        }
    )
    context_path = tmp_path / "context.csv"
    context.to_csv(context_path, index=False)
    rows = []
    for target in key["forecast_kst_dtm"]:
        for grid in (1, 2):
            rows.append(
                {
                    "forecast_kst_dtm": target,
                    "data_available_kst_dtm": key["data_available_kst_dtm"][0],
                    "grid_id": grid,
                    "heightAboveGround_10_10u": 1.0,
                    "heightAboveGround_10_10v": 2.0,
                    "isobaricInhPa_850_u": 2.0,
                    "isobaricInhPa_850_v": 3.0,
                    "isobaricInhPa_700_u": 4.0,
                    "isobaricInhPa_700_v": 5.0,
                }
            )
    gfs_path = tmp_path / "gfs.csv"
    pd.DataFrame(rows).to_csv(gfs_path, index=False)
    ldaps_path = tmp_path / "ldaps.csv"
    pd.DataFrame(rows)[
        [
            "forecast_kst_dtm",
            "data_available_kst_dtm",
            "grid_id",
            "heightAboveGround_10_10u",
            "heightAboveGround_10_10v",
        ]
    ].to_csv(ldaps_path, index=False)
    result = build_kma_risk_features(context_path, gfs_path, ldaps_path)
    assert result["kma_um_risk_ldaps_du10"].tolist() == [2.0, 3.0]
    assert result["kma_um_risk_gfs_dv850"].tolist() == [3.0, 4.0]
    assert "kma_um_ctx_shear10_850_r0" in result
    assert list(result.attrs["issue_times"]) == [
        pd.Timestamp("2023-12-31 13:00")
    ] * 2
