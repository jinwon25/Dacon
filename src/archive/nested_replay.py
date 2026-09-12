"""Produce honest decomposition tables when strict nested retraining times out.

This file never labels legacy outer-early-stopped predictions as nested. It
records the frozen replay numbers, inner-selection ledger, and an explicit
blocked primary nested row so promotion cannot proceed on contaminated OOF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(project: Path) -> pd.DataFrame:
    rows = []
    for year in (2021, 2022, 2023, 2024):
        cache_path = project / f"artifacts/followup/oof/wave0_incumbent_validate_{year}.npz"
        with np.load(cache_path, allow_pickle=False) as z:
            y, a = z["target"].astype(float), z["incumbent"].astype(float)
            inner = year - 1
            inner_path = project / f"artifacts/followup/oof/wave0_incumbent_validate_{inner}.npz"
            inner_rounds = np.nan
            if inner_path.exists():
                with np.load(inner_path, allow_pickle=False) as inner_z: inner_rounds = int(inner_z["lgb_best_iteration"][0])
            rows.append({"baseline": "v2_frozen_replay", "arm": "A_original_incumbent", "outer_validation_season": year, "n_rows": len(y), "brier": float(np.mean((a-y)**2)), "delta_vs_A": 0.0, "inner_selected_lgb_iterations": inner_rounds, "outer_target_used_for_selection": True, "status": "legacy cache; diagnostic only"})
            rec_path = project / f"artifacts/followup/models/recency_h1p0_validate_{year}.npz"
            if rec_path.exists():
                with np.load(rec_path, allow_pickle=False) as r:
                    b = np.clip(0.35 * r["weighted_lgb"] + 0.65 * r["weighted_rf"], 1e-4, 1 - 1e-4)
                rows.append({"baseline": "v2_frozen_replay", "arm": "B_R_recency_substitution", "outer_validation_season": year, "n_rows": len(y), "brier": float(np.mean((b-y)**2)), "delta_vs_A": float(np.mean((b-y)**2)-np.mean((a-y)**2)), "inner_selected_lgb_iterations": inner_rounds, "outer_target_used_for_selection": True, "status": "legacy recency cache; diagnostic only"})
            tm_path = project / f"artifacts/followup/models/lgb_trackman_pitcher_v1_validate_{year}.npz"
            if tm_path.exists():
                with np.load(tm_path, allow_pickle=False) as tm:
                    t = tm["prediction"].astype(float)
                c = np.clip(0.95*a + 0.05*t, 1e-4, 1 - 1e-4)
                rows.append({"baseline": "v2_frozen_replay", "arm": "C_Trackman_5pct_only", "outer_validation_season": year, "n_rows": len(y), "brier": float(np.mean((c-y)**2)), "delta_vs_A": float(np.mean((c-y)**2)-np.mean((a-y)**2)), "inner_selected_lgb_iterations": inner_rounds, "outer_target_used_for_selection": True, "status": "legacy Trackman cache; diagnostic only"})
            hybrid_path = project / "research/reports/hybrid_candidate_results.csv"
            if hybrid_path.exists():
                hybrid = pd.read_csv(hybrid_path)
                match = hybrid.loc[hybrid["outer_validation_season"].eq(year)]
                if not match.empty:
                    d_brier = float(match.iloc[0]["brier"])
                    rows.append({"baseline": "v2_frozen_replay", "arm": "D_full_v2", "outer_validation_season": year, "n_rows": len(y), "brier": d_brier, "delta_vs_A": d_brier - float(np.mean((a-y)**2)), "inner_selected_lgb_iterations": inner_rounds, "outer_target_used_for_selection": True, "status": "legacy hybrid report; diagnostic only"})
            rows.append({"baseline": "v2_nested", "arm": "A/B/C/D_primary", "outer_validation_season": year, "n_rows": len(y), "brier": np.nan, "delta_vs_A": np.nan, "inner_selected_lgb_iterations": inner_rounds, "outer_target_used_for_selection": False, "status": "BLOCKED: strict fixed-fit retraining exceeded runtime window; no nested OOF artifact"})
    out = pd.DataFrame(rows)
    out.to_csv(project / "research/reports/champion_v2_nested_results.csv", index=False)
    out.loc[out["arm"].isin(["A_original_incumbent", "B_R_recency_substitution", "C_Trackman_5pct_only", "D_full_v2"]), ["baseline", "arm", "outer_validation_season", "brier", "delta_vs_A", "status"]].to_csv(project / "research/reports/v2_component_ablation.csv", index=False)
    (project / "artifacts/followup/v2_nested_provenance.json").write_text('{"primary_status":"BLOCKED","reason":"strict fixed-fit nested retraining exceeded the local runtime window; outer-target-dependent caches remain diagnostic only","outer_target_used_for_selection":false}', encoding="utf-8")
    print(out.to_string(index=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__": main()
