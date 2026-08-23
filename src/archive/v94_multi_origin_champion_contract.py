"""Build a fidelity-labelled multi-origin OOF contract for post-v84 research.

No historical artifact can honestly reproduce the complete final v84 ladder on
every old season.  This runner therefore exposes two explicit evidence tiers:

* a common strict-forward wave0 parent for full 2020--2024 seasons;
* exact v84-component parents where they are reconstructable: R_CORE/F in
  full 2022, all rows in late 2023, and all rows in full 2024.

Every bundle is aligned by original train row index and protected by train,
row-id, target and parent digests.  Downstream experiments may use the common
tier for mechanism stability and the v84 tier for champion marginal gain, but
must never silently relabel common-wave0 evidence as exact-v84 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import brier_score, brier_skill_score_unclipped
from src.archive.v92_temporal_player_command_eb import _compose_historical_parents


PROTOCOL = "V94_MULTI_ORIGIN_CHAMPION_CONTRACT_V1"
TARGET = "control_success"
COMMON_YEARS = (2020, 2021, 2022, 2023, 2024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest().upper()


def sha256_strings(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype("string").fillna("__MISSING__"):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest().upper()


def domain3(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R")
    anchor = frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    return np.where(~regular.to_numpy(), "F", np.where(anchor.to_numpy(), "R_ANCHOR", "R_CORE"))


def validate_axis(
    frame: pd.DataFrame,
    raw_index: np.ndarray,
    target: np.ndarray,
    parent: np.ndarray,
) -> None:
    index = np.asarray(raw_index, dtype=np.int64)
    y = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(parent, dtype=np.float64)
    if not (index.shape == y.shape == prediction.shape == (len(frame),)):
        raise ValueError("axis arrays are not aligned")
    if len(np.unique(index)) != len(index) or np.any(index < 0):
        raise ValueError("axis raw indices must be unique and nonnegative")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("axis target must be binary")
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("axis parent is not a valid probability")
    if not np.array_equal(y, frame[TARGET].to_numpy(np.float64)):
        raise ValueError("axis target/frame order mismatch")


def _diagnostics(frame: pd.DataFrame, parent: np.ndarray) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    residual = target - np.asarray(parent, dtype=np.float64)
    month = []
    for value in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(value).to_numpy()
        month.append(
            {
                "month": int(value),
                "rows": int(mask.sum()),
                "target_rate": float(target[mask].mean()),
                "prediction_mean": float(np.mean(parent[mask])),
                "mean_residual": float(np.mean(residual[mask])),
                "brier": brier_score(target[mask], parent[mask]),
            }
        )
    return {
        "rows": int(len(frame)),
        "target_rate": float(target.mean()),
        "prediction_mean": float(np.mean(parent)),
        "mean_residual": float(np.mean(residual)),
        "brier": brier_score(target, parent),
        "unclipped_bss_equivalent": brier_skill_score_unclipped(target, parent),
        "months": month,
    }


def _save_axis(
    output_dir: Path,
    name: str,
    frame: pd.DataFrame,
    raw_index: np.ndarray,
    parent: np.ndarray,
    *,
    fidelity: str,
    training_max_season: int,
    exact_mask: np.ndarray | None = None,
    common_parent: np.ndarray | None = None,
) -> dict[str, Any]:
    target = frame[TARGET].to_numpy(np.float64)
    validate_axis(frame, raw_index, target, parent)
    if exact_mask is None:
        exact = np.ones(len(frame), dtype=bool)
    else:
        exact = np.asarray(exact_mask, dtype=bool)
        if exact.shape != (len(frame),):
            raise ValueError("axis exact mask is not aligned")
    payload: dict[str, np.ndarray] = {
        "raw_index": np.asarray(raw_index, dtype=np.int64),
        "target": target,
        "parent": np.asarray(parent, dtype=np.float64),
        "exact_mask": exact,
        "season": frame["season"].to_numpy(np.int16),
        "game_month": frame["game_month"].to_numpy(np.int16),
        "domain3": domain3(frame).astype(str),
        "pitcher_id": frame["pitcher_id"].to_numpy(),
        "batter_id": frame["batter_id"].to_numpy(),
    }
    if common_parent is not None:
        common = np.asarray(common_parent, dtype=np.float64)
        if common.shape != (len(frame),):
            raise ValueError("axis common parent is not aligned")
        payload["common_parent"] = common
    path = output_dir / f"{name}.npz"
    np.savez_compressed(path, **payload)
    domains = payload["domain3"]
    exact_domains = [str(value) for value in np.unique(domains[exact])]
    return {
        "axis": name,
        "path": path.name,
        "file_sha256": sha256_file(path),
        "fidelity": fidelity,
        "evaluation_season": int(frame["season"].max()),
        "training_max_season": int(training_max_season),
        "rows": int(len(frame)),
        "exact_rows": int(exact.sum()),
        "exact_fraction": float(exact.mean()),
        "exact_domains": exact_domains,
        "raw_index_sha256": sha256_array(payload["raw_index"]),
        "row_id_sha256": sha256_strings(frame["row_id"]),
        "target_sha256": sha256_array(target),
        "parent_sha256": sha256_array(np.asarray(parent, dtype=np.float64)),
        "diagnostics": _diagnostics(frame, parent),
    }


def run(project: Path, support_dir: Path, external_root: Path, output_dir: Path) -> dict[str, Any]:
    project = project.resolve()
    support_dir = support_dir.resolve()
    external_root = external_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = project / "data" / "train.csv"
    raw = pd.read_csv(train_path, low_memory=False)
    raw_index = np.arange(len(raw), dtype=np.int64)
    axes: list[dict[str, Any]] = []
    common_cache: dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}

    for year in COMMON_YEARS:
        mask = raw["season"].eq(year).to_numpy()
        frame = raw.loc[mask].reset_index(drop=True)
        index = raw_index[mask]
        cache_path = project / "artifacts" / "followup" / "oof" / f"wave0_incumbent_validate_{year}.npz"
        with np.load(cache_path, allow_pickle=False) as saved:
            cache_index = saved["valid_idx"].astype(np.int64)
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        if not np.array_equal(index, cache_index):
            raise ValueError(f"wave0 raw-index mismatch: {year}")
        if not np.array_equal(target, frame[TARGET].to_numpy(np.float64)):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        common_cache[year] = (frame, index, parent)
        axes.append(
            _save_axis(
                output_dir, f"common_full_{year}", frame, index, parent,
                fidelity="strict_forward_common_wave0", training_max_season=year - 1,
            )
        )

    frames, parents, provenance = _compose_historical_parents(
        project, support_dir, external_root, raw
    )
    frame22 = frames["full_2022"].reset_index(drop=True)
    common22, index22, wave22 = common_cache[2022]
    if not np.array_equal(frame22["row_id"].astype(str), common22["row_id"].astype(str)):
        raise ValueError("v84/full-2022 row-id mismatch")
    exact22 = ~frame22["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
    axes.append(
        _save_axis(
            output_dir, "v84_full_2022", frame22, index22, parents["full_2022"],
            fidelity="exact_v84_on_R_CORE_and_F_only", training_max_season=2021,
            exact_mask=exact22, common_parent=wave22,
        )
    )

    full23, index23, wave23 = common_cache[2023]
    late23 = full23["game_month"].ge(8).to_numpy()
    frame23 = frames["late_2023"].reset_index(drop=True)
    if not np.array_equal(frame23["row_id"].astype(str), full23.loc[late23, "row_id"].astype(str).reset_index(drop=True)):
        raise ValueError("v84/late-2023 row-id mismatch")
    axes.append(
        _save_axis(
            output_dir, "v84_late_2023", frame23, index23[late23], parents["late_2023"],
            fidelity="exact_v84_all_domains", training_max_season=2023,
            common_parent=wave23[late23],
        )
    )

    frame24 = frames["full_2024"].reset_index(drop=True)
    common24, index24, wave24 = common_cache[2024]
    if not np.array_equal(frame24["row_id"].astype(str), common24["row_id"].astype(str)):
        raise ValueError("v84/full-2024 row-id mismatch")
    axes.append(
        _save_axis(
            output_dir, "v84_full_2024", frame24, index24, parents["full_2024"],
            fidelity="exact_v84_all_domains", training_max_season=2023,
            common_parent=wave24,
        )
    )

    manifest = {
        "protocol": PROTOCOL,
        "train_csv_sha256": sha256_file(train_path),
        "train_rows": int(len(raw)),
        "common_axes": [f"common_full_{year}" for year in COMMON_YEARS],
        "v84_axes": ["v84_full_2022", "v84_late_2023", "v84_full_2024"],
        "parent_provenance": provenance,
        "axes": axes,
        "contract": {
            "common_tier_use": "mechanism direction and temporal stability only",
            "v84_tier_use": "marginal gain against the deployed champion analogue",
            "partial_exact_rule": "score v84_full_2022 only where exact_mask is true",
            "forbidden": [
                "relabelling common-wave0 evidence as exact-v84",
                "using an audit target to select a recipe for the same axis",
                "test-row aggregate, order, frequency or distribution feature",
                "Public-score-derived route, weight or postprocessing",
            ],
        },
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_candidate_requirement": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    summary = {
        "protocol": PROTOCOL,
        "train_csv_sha256": manifest["train_csv_sha256"],
        "axis_count": int(len(axes)),
        "common_axis_count": len(COMMON_YEARS),
        "v84_axis_count": 3,
        "axes": [
            {
                key: axis[key]
                for key in ("axis", "fidelity", "rows", "exact_rows", "exact_fraction", "exact_domains", "file_sha256")
            }
            for axis in axes
        ],
        "decision": "contract ready; downstream model selection must preserve fidelity labels",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.support_dir, args.external_root, args.output_dir)


if __name__ == "__main__":
    main()
