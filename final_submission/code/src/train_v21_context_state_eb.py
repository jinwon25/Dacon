"""Train the frozen row-local v21 context/state empirical-Bayes overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.train_v20_target1160 import SEPARATOR, _key, _read_season


SOURCE_YEAR = 2024
RECIPES = (
    {
        "name": "pitcher_hand_prev3_b10_rcore",
        "columns": ("pitcher_id", "batter_hand", "prev3_b10"),
        "domain": "R_CORE",
        "half_life": 0.5,
        "alpha": 200.0,
        "weight": 0.275,
    },
    {
        "name": "pitcher_hand_prev5_b10_rcore",
        "columns": ("pitcher_id", "batter_hand", "prev5_b10"),
        "domain": "R_CORE",
        "half_life": 1.0,
        "alpha": 100.0,
        "weight": 0.075,
    },
    {
        "name": "pitcher_hand_base_all",
        "columns": ("pitcher_id", "batter_hand", "base_state"),
        "domain": "ALL",
        "half_life": 1.0,
        "alpha": 25.0,
        "weight": 0.05,
    },
    {
        "name": "count_hands_inning_all",
        "columns": (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "inning_band",
        ),
        "domain": "ALL",
        "half_life": 2.0,
        "alpha": 25.0,
        "weight": 0.10,
    },
    {
        "name": "count_hands_prev3_b20_all",
        "columns": (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "prev3_b20",
        ),
        "domain": "ALL",
        "half_life": 0.5,
        "alpha": 800.0,
        "weight": 0.20,
    },
    {
        "name": "count_hands_reverse_b10_all",
        "columns": (
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
            "reverse_b10",
        ),
        "domain": "ALL",
        "half_life": 2.0,
        "alpha": 100.0,
        "weight": 0.05,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def add_v21_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add only deterministic, current-row features used by the frozen lookup."""
    output = frame.copy()

    def rate_bin(column: str, bins: int) -> np.ndarray:
        value = pd.to_numeric(output[column], errors="coerce").fillna(0.5)
        return np.floor(np.clip(value.to_numpy(np.float64), 0.0, 1.0) * bins).astype(
            np.int16
        )

    output["prev3_b10"] = rate_bin(
        "asof_pitcher_prev3_game_success_rate", 10
    )
    output["prev5_b10"] = rate_bin(
        "asof_pitcher_prev5_game_success_rate", 10
    )
    output["prev3_b20"] = rate_bin(
        "asof_pitcher_prev3_game_success_rate", 20
    )
    output["reverse_b10"] = rate_bin("asof_pitcher_reverse_rate", 10)
    output["inning_band"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, 9, np.inf),
        labels=("early", "middle", "late", "extra"),
    ).astype("string")
    return output


def _fit_recipes(source: pd.DataFrame, residual: np.ndarray) -> list[dict[str, object]]:
    source = add_v21_features(source)
    cutoff = float(pd.to_numeric(source["game_month"], errors="coerce").max())
    fitted: list[dict[str, object]] = []
    for recipe in RECIPES:
        mask = np.ones(len(source), dtype=bool)
        if recipe["domain"] != "ALL":
            mask &= source["domain3"].eq(recipe["domain"]).to_numpy()
        rows = source.loc[mask].reset_index(drop=True)
        month = pd.to_numeric(rows["game_month"], errors="coerce").to_numpy(
            np.float64
        )
        row_weight = np.exp2(
            -(cutoff - month) / float(recipe["half_life"])
        )
        stats = pd.DataFrame(
            {
                "key": _key(rows, tuple(recipe["columns"])),
                "weighted_residual": row_weight * residual[mask],
                "effective_n": row_weight,
            }
        ).groupby("key", observed=True)[["weighted_residual", "effective_n"]].sum()
        effect = stats["weighted_residual"] / (
            stats["effective_n"] + float(recipe["alpha"])
        )
        fitted.append(
            {
                **recipe,
                "columns": list(recipe["columns"]),
                "source_month_cutoff": cutoff,
                "n_groups": int(len(effect)),
                "effects": {str(key): float(value) for key, value in effect.items()},
            }
        )
    return fitted


def run(
    project: Path,
    output_dir: Path,
    joint_oof_path: Path = Path(
        "reproduction_inputs/v20_pfd/joint_candidate_o2024.npz"
    ),
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = project / "data" / "train.csv"
    source, _ = _read_season(train_path, SOURCE_YEAR)
    source = _add_domain_and_pressure(source)
    joint_oof_path = (
        joint_oof_path.resolve()
        if joint_oof_path.is_absolute()
        else (project / joint_oof_path).resolve()
    )
    with np.load(joint_oof_path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v19 = saved["candidate"].astype(np.float64)
    if not np.array_equal(
        target, source["control_success"].to_numpy(np.float64)
    ):
        raise ValueError("v19 OOF and 2024 source rows are not aligned")
    recipes = _fit_recipes(source, target - v19)
    spec = {
        "candidate": "v21_context_state_recency_eb",
        "source_year": SOURCE_YEAR,
        "separator": SEPARATOR,
        "anchor_team_id": 13,
        "recipes": recipes,
        "local_evidence": {
            "gain_vs_v20_y2023_to_y2024": 10.771984847413023,
            "gain_vs_v20_y2023_early_to_late": 50.63447220738925,
            "gain_vs_v20_y2024_early_to_late": 14.147973376099685,
            "minimum_primary_gain": 10.771984847413023,
            "positive_2024_month_fraction": 0.75,
            "pitcher_bootstrap_p05": 3.1600584051529337,
        },
        "row_local_inference": True,
        "test_aggregate_used": False,
        "external_data_used": False,
    }
    spec_path = output_dir / "v21_context_state_eb_spec.json"
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol": "V21_CONTEXT_STATE_RECENCY_EB_FINAL_2025_V1",
        "train_sha256": _sha256(train_path),
        "joint_oof_sha256": _sha256(joint_oof_path),
        "source_year": SOURCE_YEAR,
        "source_rows": int(len(source)),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "reference_model_read_during_fit": False,
        "test_values_read_during_fit": False,
        "private_score_recomputed": False,
        "artifacts": {
            spec_path.name: {
                "size_bytes": spec_path.stat().st_size,
                "sha256": _sha256(spec_path),
            }
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v21_context_state_eb_final_20260816"),
    )
    parser.add_argument(
        "--joint-oof",
        type=Path,
        default=Path("reproduction_inputs/v20_pfd/joint_candidate_o2024.npz"),
    )
    args = parser.parse_args()
    target = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (args.project.resolve() / args.output_dir).resolve()
    )
    if target.exists():
        raise FileExistsError("Use a new output directory")
    run(args.project, args.output_dir, args.joint_oof)


if __name__ == "__main__":
    main()
