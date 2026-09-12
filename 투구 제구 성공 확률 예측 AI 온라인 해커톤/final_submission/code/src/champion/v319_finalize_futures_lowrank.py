"""Fit the frozen v50 low-rank lookup for row-local 2025 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.champion.v50_low_rank_pitcher_context import fit_source_matrix


PROTOCOL = "V319_FINALIZE_FUTURES_LOWRANK_V1"
TARGET = "control_success"
SOURCE_YEARS = (2020, 2021, 2022, 2023, 2024)
SMOOTHING = 300.0
RANK = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def run(train_csv: Path, wave0_oof_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(
        train_csv,
        usecols=[
            "season", "pitcher_id", "balls_before", "strikes_before",
            "batter_hand", TARGET,
        ],
        low_memory=False,
    )
    arrays: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for year in SOURCE_YEARS:
        rows = train.loc[train["season"].eq(year)].reset_index(drop=True)
        oof_path = wave0_oof_dir / f"wave0_incumbent_validate_{year}.npz"
        with np.load(oof_path, allow_pickle=False) as saved:
            target = saved["target"].astype(np.float64)
            parent = saved["incumbent"].astype(np.float64)
        if not np.array_equal(target, rows[TARGET].to_numpy(np.float64)):
            raise ValueError(f"wave0 target/order mismatch: {year}")
        model = fit_source_matrix(
            rows, target, parent,
            smoothing_grid=(SMOOTHING,), rank_grid=(RANK,),
        )
        arrays[f"pitcher_ids_{year}"] = np.asarray(model["pitcher_ids"], dtype=np.int64)
        arrays[f"matrix_{year}"] = np.asarray(
            model["reconstructions"][(SMOOTHING, RANK)], dtype=np.float32
        )
        diagnostics[str(year)] = model["diagnostics"]
        sources[str(year)] = {"path": str(oof_path), "sha256": sha256(oof_path)}
    lookup_path = output_dir / "futures_lowrank_lookup.npz"
    np.savez_compressed(lookup_path, source_years=np.asarray(SOURCE_YEARS), **arrays)
    summary = {
        "protocol": PROTOCOL,
        "status": "finalized",
        "source_years": list(SOURCE_YEARS),
        "recipe": {"smoothing": SMOOTHING, "rank": RANK},
        "lookup_path": str(lookup_path),
        "lookup_sha256": sha256(lookup_path),
        "lookup_bytes": lookup_path.stat().st_size,
        "diagnostics": diagnostics,
        "source_oof": sources,
        "restrictions": {
            "official_train_only": True,
            "source_residuals_are_forward_oof": True,
            "test_data_used": False,
            "runtime_mapping_is_row_local": True,
            "recipe_retuned": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--wave0-oof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train_csv, args.wave0_oof_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
