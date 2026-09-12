"""Build the v244 fixed-route package with corrected ASOF anchor semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROTOCOL = "V286_BUILD_EXACT_ANCHOR_PACKAGE_V1"
SPECS = [
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "p_succ"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "p_rev"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", "p_mid"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", "p_ball"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", "p_stk"),
    ("batter_id", "asof_batter_n", "asof_batter_success_rate", "b_succ"),
    ("batter_id", "asof_batter_n", "asof_batter_middle_rate", "b_mid"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _pairs(
    frame: pd.DataFrame, key: str, values: list[str]
) -> dict[int, tuple[float, ...]]:
    return {
        int(index): tuple(float(value) for value in row)
        for index, row in frame.set_index(key)[values].iterrows()
    }


def replace_anchors(
    lookup: dict[str, Any], history: pd.DataFrame
) -> dict[str, Any]:
    """Change only the seven ASOF anchors in the frozen v244 lookup."""

    for id_column, n_column, rate_column, prefix in SPECS:
        values = history[[id_column, n_column, rate_column]].copy()
        values["cumulative_sum"] = (
            pd.to_numeric(values[n_column], errors="coerce")
            * pd.to_numeric(values[rate_column], errors="coerce").fillna(0.0)
        )
        latest_index = values.groupby(id_column, sort=False)[n_column].idxmax()
        latest = values.loc[
            latest_index, [id_column, n_column, "cumulative_sum"]
        ]
        lookup["anchors"][prefix] = _pairs(
            latest, id_column, [n_column, "cumulative_sum"]
        )
    lookup["features_version"] = 2
    lookup["anchor_semantics"] = "latest_prior_pre_pitch_state"
    lookup["multiscale_semantics"] = "same_exact_end_anchor"
    lookup["anchor_history_max_season"] = int(history["season"].max())
    return lookup


def _restore_v244_script(text: str) -> str:
    text = text.replace("\r\n", "\n")
    helper_start = text.find("def _predict_pseudo_deployment(")
    helper_end = text.find("def _window_adjustment(", helper_start)
    if helper_start >= 0:
        if helper_end < 0:
            raise RuntimeError("pseudo helper end marker missing")
        text = text[:helper_start] + text[helper_end:]

    start_marker = "    jy_probability = output.copy()\n"
    end_marker = "    return parent, h1, c3, active, output\n"
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    replacement = '''    jy_probability = output.copy()
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    rcore = regular & ~(
        frame["pitcher_team_id"].eq(13).to_numpy()
        | frame["batter_team_id"].eq(13).to_numpy()
    )
    pressure = rcore & (
        (frame["num_runners_on"].to_numpy(dtype=np.float64) > 0.0)
        | (frame["li"].to_numpy(dtype=np.float64) >= 1.5)
    )
    same_hand = frame["pitcher_hand"].astype(str).eq(
        frame["batter_hand"].astype(str)
    ).to_numpy()
    xgb_probability = _predict_fallback_xgb(frame)
    deployed = pressure & (jy_probability >= XGB_ACTIVE_THRESHOLD)
    boundary = (
        pressure
        & (jy_probability >= XGB_BOUNDARY_LOW)
        & (jy_probability < XGB_BOUNDARY_HIGH)
        & (np.abs(xgb_probability - jy_probability) <= XGB_BOUNDARY_AGREEMENT)
    )
    nonpressure_same = rcore & ~pressure & same_hand
    nonpressure_opposite = (
        rcore
        & ~pressure
        & ~same_hand
        & (jy_probability >= XGB_NONPRESSURE_OPPOSITE_THRESHOLD)
    )
    routes = (
        (deployed, XGB_ACTIVE_WEIGHT),
        (boundary, XGB_BOUNDARY_WEIGHT),
        (nonpressure_same, XGB_NONPRESSURE_SAME_WEIGHT),
        (nonpressure_opposite, XGB_NONPRESSURE_OPPOSITE_WEIGHT),
    )
    route_sum = np.column_stack([mask for mask, _weight in routes]).sum(axis=1)
    if np.any(route_sum > 1):
        raise RuntimeError("fallback routes overlap")
    for mask, weight in routes:
        output[mask] = np.clip(
            jy_probability[mask]
            + weight * (xgb_probability[mask] - jy_probability[mask]),
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
    return text[:start] + replacement + text[end:]


def run(
    source_package: Path,
    train_csv: Path,
    runtime_source: Path,
    v285_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(v285_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V285_EXACT_ANCHOR_FIXED_ROUTE_AUDIT_V1":
        raise ValueError("unexpected v285 summary protocol")
    if not audit.get("eligible_for_exploratory_packaging"):
        raise ValueError("v285 is not eligible for exploratory packaging")
    if float(audit["selected"]["weight"]) != 1.0:
        raise ValueError("v285 did not select full exact-anchor replacement")

    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    output_zip = output_dir / "submit_v286_exact_anchor.zip"
    source_v244_sha256 = None
    with tempfile.TemporaryDirectory(prefix="v286_exact_anchor_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(source_package) as archive:
            archive.extractall(stage)

        pseudo_metadata = stage / "model" / "pseudo_deployment" / "metadata.json"
        if pseudo_metadata.exists():
            source_v244_sha256 = json.loads(
                pseudo_metadata.read_text(encoding="utf-8")
            ).get("source_package_sha256")

        script = stage / "script.py"
        script.write_text(
            _restore_v244_script(script.read_text(encoding="utf-8")),
            encoding="utf-8",
        )

        fallback = stage / "model" / "fallback_xgb"
        lookup_path = fallback / "fallback_lookups.joblib"
        lookup = replace_anchors(joblib.load(lookup_path), train)
        joblib.dump(lookup, lookup_path, compress=3)
        shutil.copy2(runtime_source, fallback / "runtime.py")

        fallback_metadata_path = fallback / "metadata.json"
        fallback_metadata = json.loads(
            fallback_metadata_path.read_text(encoding="utf-8")
        )
        fallback_metadata.update(
            {
                "features_version": 2,
                "anchor_semantics": "latest_prior_pre_pitch_state",
                "multiscale_semantics": "same_exact_end_anchor",
                "only_frozen_anchor_state_changed": True,
            }
        )
        fallback_metadata_path.write_text(
            json.dumps(fallback_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        pseudo_asset = stage / "model" / "pseudo_deployment"
        if pseudo_asset.exists():
            shutil.rmtree(pseudo_asset)

        metadata = {
            "protocol": PROTOCOL,
            "source_package_sha256": _sha256(source_package),
            "source_v244_sha256": source_v244_sha256,
            "v285_summary_sha256": _sha256(v285_summary),
            "official_train_sha256": _sha256(train_csv),
            "selected_exact_anchor_dose": 1.0,
            "v244_routes_and_route_weights_frozen": True,
            "test_row_aggregation": False,
            "exploratory_challenger": True,
        }
        (stage / "model" / "exact_anchor_v286.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in stage.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    with zipfile.ZipFile(output_zip) as archive:
        bad_member = archive.testzip()
        entries = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"zip CRC failure: {bad_member}")
    result = {
        "protocol": PROTOCOL,
        "status": "package_built_pending_standalone_audit",
        "package": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "sha256": _sha256(output_zip),
        "file_count": len(entries),
        "crc_passed": True,
        "source_v244_sha256": source_v244_sha256,
        "selected_exact_anchor_dose": 1.0,
        "v244_routes_and_route_weights_frozen": True,
        "exploratory_challenger": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--v285-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_package,
        args.train_csv,
        args.runtime_source,
        args.v285_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
