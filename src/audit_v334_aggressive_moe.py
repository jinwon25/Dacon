"""Audit v334 route formulas and row independence against v320."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_frame(train_csv: Path, rows_per_route: int) -> pd.DataFrame:
    frame = pd.read_csv(train_csv, low_memory=False).drop(
        columns=["control_success"], errors="ignore"
    )
    regular = frame["game_type"].astype(str).eq("R")
    anchor = regular & (
        frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    )
    routes = [frame[frame["game_type"].astype(str).eq("F")], frame[anchor], frame[regular & ~anchor]]
    if any(part.empty for part in routes):
        # Some inference files need not contain every historical routing value.
        # Create contract fixtures from otherwise valid 2025 rows so every
        # disjoint branch is still executed without learning a test aggregate.
        fixture = frame.sample(
            n=min(rows_per_route, len(frame)), random_state=334
        ).reset_index(drop=True)
        futures = fixture.copy()
        futures["game_type"] = "F"
        anchor_rows = fixture.copy()
        anchor_rows["game_type"] = "R"
        anchor_rows["pitcher_team_id"] = 13
        core_rows = fixture.copy()
        core_rows["game_type"] = "R"
        core_rows.loc[core_rows["pitcher_team_id"].eq(13), "pitcher_team_id"] = 16
        core_rows.loc[core_rows["batter_team_id"].eq(13), "batter_team_id"] = 16
        sampled = [futures, anchor_rows, core_rows]
    else:
        sampled = [
            part.sample(n=min(rows_per_route, len(part)), random_state=334 + index)
            for index, part in enumerate(routes)
        ]
    output = pd.concat(sampled, ignore_index=True)
    # The standalone competition bundle is intentionally fail-closed for the
    # 2025 inference season.  We use official historical rows only as diverse
    # row-local covariate fixtures, never as a scored validation target.
    output["season"] = 2025
    return output


def predict_partitioned(module, frame: pd.DataFrame) -> np.ndarray:
    split = max(1, len(frame) // 2)
    return np.concatenate(
        [
            module.predict_dataframe(frame.iloc[:split].reset_index(drop=True)),
            module.predict_dataframe(frame.iloc[split:].reset_index(drop=True)),
        ]
    )


def run(
    v320_zip: Path,
    v334_zip: Path,
    train_csv: Path,
    output_json: Path,
    rows_per_route: int,
) -> dict[str, object]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="v334_audit_") as temporary:
        root = Path(temporary)
        old = root / "v320"
        new = root / "v334"
        with zipfile.ZipFile(v320_zip) as archive:
            archive.extractall(old)
        with zipfile.ZipFile(v334_zip) as archive:
            archive.extractall(new)
        v320 = load_module("audit_v320_runtime_for_v334", old / "script.py")
        v334 = load_module("audit_v334_runtime", new / "script.py")
        frame = balanced_frame(train_csv, rows_per_route)
        regular = frame["game_type"].astype(str).eq("R").to_numpy()
        anchor = regular & (
            frame["pitcher_team_id"].eq(13).to_numpy()
            | frame["batter_team_id"].eq(13).to_numpy()
        )
        rcore = regular & ~anchor
        futures = frame["game_type"].astype(str).eq("F").to_numpy()

        parent = v320.predict_dataframe(frame.copy())
        candidate = v334.predict_dataframe(frame.copy())
        lowrank = v334._predict_futures_lowrank(frame.copy())
        transition = v334._predict_player_transition(frame.copy())
        expected = parent.copy()
        expected[anchor] = np.clip(
            expected[anchor] + 0.50 * lowrank[anchor], 0.001, 0.999
        )
        expected[rcore] = np.clip(
            expected[rcore] + 0.25 * transition[rcore], 0.001, 0.999
        )

        order = frame.sample(frac=1.0, random_state=334).index.to_numpy()
        shuffled_prediction = v334.predict_dataframe(
            frame.iloc[order].reset_index(drop=True)
        )
        restored = np.empty_like(shuffled_prediction)
        restored[order] = shuffled_prediction
        partitioned = predict_partitioned(v334, frame)

        helper_rows = frame.iloc[: min(96, len(frame))].reset_index(drop=True)
        helper_full = v334._predict_player_transition(helper_rows)
        helper_singletons = np.asarray(
            [
                v334._predict_player_transition(
                    helper_rows.iloc[[index]].reset_index(drop=True)
                )[0]
                for index in range(len(helper_rows))
            ]
        )
        input_is_test = train_csv.name.lower() == "test.csv"
        result = {
            "protocol": "V334_AGGRESSIVE_MOE_RUNTIME_AUDIT_V1",
            "rows": int(len(frame)),
            "route_rows": {
                "F": int(futures.sum()),
                "R_ANCHOR": int(anchor.sum()),
                "R_CORE": int(rcore.sum()),
            },
            "f_parity_max_abs": float(np.max(np.abs(candidate[futures] - parent[futures]))),
            "r_formula_max_abs": float(np.max(np.abs(candidate[regular] - expected[regular]))),
            "shuffle_max_abs": float(np.max(np.abs(candidate - restored))),
            "partition_max_abs": float(np.max(np.abs(candidate - partitioned))),
            "transition_helper_singleton_max_abs": float(
                np.max(np.abs(helper_full - helper_singletons))
            ),
            "finite": bool(np.isfinite(candidate).all()),
            "in_range": bool(np.all((candidate >= 0.0) & (candidate <= 1.0))),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_csv_read": input_is_test,
            "test_use_scope": (
                "unlabelled row-local runtime fixtures only; no aggregate, "
                "target, tuning or candidate selection"
                if input_is_test else "not used"
            ),
            "test_aggregate_used": False,
            "test_used_for_candidate_selection": False,
        }
        result["status"] = "pass" if (
            result["f_parity_max_abs"] <= 1e-15
            and result["r_formula_max_abs"] <= 1e-15
            and result["shuffle_max_abs"] <= 1e-15
            and result["partition_max_abs"] <= 1e-15
            and result["transition_helper_singleton_max_abs"] <= 1e-15
            and result["finite"]
            and result["in_range"]
        ) else "fail"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v320-zip", type=Path, required=True)
    parser.add_argument("--v334-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rows-per-route", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(run(
        args.v320_zip, args.v334_zip, args.train_csv,
        args.output_json, args.rows_per_route,
    ), indent=2))


if __name__ == "__main__":
    main()
