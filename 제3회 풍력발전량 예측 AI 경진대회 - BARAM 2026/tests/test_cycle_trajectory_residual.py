from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts_final" / "structural_20260718" / "structural_report.json"


def test_finesweep_reference_parity() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert abs(report["baseline_parity"]["test_finesweep_changed_ratio_vs_reference"] - 0.11929223744292237) < 1e-6
    assert abs(report["baseline_parity"]["test_finesweep_mae_kwh_vs_reference"] - 13.749979248804843) < 0.01


def test_h2_rolling_fine_is_not_reference_cache() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    evaluation = json.loads(
        (REPORT.parent / "agent_evaluation.json").read_text(encoding="utf-8")
    )
    locked = np.load(ROOT / "artifacts_final" / "structural_20260718" / "locked_predictions.npz")
    reference = np.load(ROOT / "artifacts_final" / "meta_gate" / "meta_gate_cache.npz")
    timestamps = pd.to_datetime(locked["index_ns"])
    h2 = timestamps >= pd.Timestamp("2024-07-01")
    assert report["baseline_parity"]["reference_cache_surface"].find("p=.55/a=.25") >= 0
    assert np.mean(np.abs(locked["base"][h2] - reference["valid_candidate"][h2])) > 1e-3
    monthly_scores = [item["score"] for item in report["h1_to_h2_locked"]["monthly_deltas"].values()]
    assert evaluation["worst_month_score_delta"] == min(monthly_scores)
