"""Build v345 by adding the frozen Beta cell to the v343 standalone ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import joblib
import numpy as np
import pandas as pd

from src.archive.v321_strict_beta_binomial_complement import (
    CANDIDATE_NAMES,
    fit_previous_year_pool,
    terminal_snapshot,
)


PROTOCOL = "V345_BUILD_TRANSITION_WORKLOAD_BETA_PACKAGE_V1"
BUNDLE_MEMBER = "model/beta_cell/spec.joblib"
FUNCTION_ANCHOR = "def _predict_player_transition(frame: pd.DataFrame) -> np.ndarray:\n"

BETA_FUNCTIONS = '''def _beta_cell_mask(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    support = pd.to_numeric(
        frame["asof_pitcher_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    pitchmix_n = pd.to_numeric(
        frame["asof_pitcher_pitchmix_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    pitchmix = frame[
        [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
    ].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy(np.float64)
    mixed = (np.max(pitchmix, axis=1) < 0.50) & (pitchmix_n >= 100.0)
    developing = (support >= 100.0) & (support < 800.0)
    return regular & ~anchor & developing & mixed


def _beta_season_posterior(
    frame: pd.DataFrame,
    bundle: dict,
    prefix: str,
    prior: np.ndarray,
) -> np.ndarray:
    identifier = pd.to_numeric(
        frame[f"{prefix}_id"], errors="raise"
    ).astype("int64")
    opening_n = identifier.map(bundle[f"{prefix}_opening_n"]).fillna(0.0).to_numpy(np.float64)
    opening_success = identifier.map(
        bundle[f"{prefix}_opening_success"]
    ).fillna(0.0).to_numpy(np.float64)
    career_n = pd.to_numeric(
        frame[f"asof_{prefix}_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    career_rate = pd.to_numeric(
        frame[f"asof_{prefix}_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    season_n = np.maximum(career_n - opening_n, 0.0)
    season_success = np.clip(
        career_n * career_rate - opening_success, 0.0, season_n
    )
    concentration = float(bundle["concentration"])
    return (season_success + concentration * prior) / (season_n + concentration)


def _predict_beta_cell(frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "beta_cell" / "spec.joblib")
    prior = frame["game_type"].astype(str).map(
        bundle["prior_by_game_type"]
    ).fillna(float(bundle["global_prior"])).to_numpy(np.float64)
    pitcher = _beta_season_posterior(frame, bundle, "pitcher", prior)
    batter = _beta_season_posterior(frame, bundle, "batter", prior)
    career = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    career = np.where(np.isfinite(career), career, prior)
    recent = pd.to_numeric(
        frame["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    recent = np.where(np.isfinite(recent), recent, career)
    matrix = np.clip(
        np.column_stack((pitcher, batter, prior, career, recent)),
        0.001,
        0.999,
    )
    probability = matrix @ np.asarray(bundle["weights"], dtype=np.float64)
    return np.clip(probability, 0.001, 0.999)


'''

OLD_TAIL = '''    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    if np.any(anchor):
        if lowrank_probability_delta is None:
            lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[anchor] = np.clip(
            output[anchor] + 0.50 * lowrank_probability_delta[anchor],
            0.001,
            0.999,
        )
    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
            0.001,
            0.999,
        )
    workload_active = active & rcore
    if np.any(workload_active):
        workload_h1 = _predict_workload_h1(
            frame.loc[workload_active].reset_index(drop=True)
        )
        workload_proposal = np.clip(
            0.82 * effective_bridge[workload_active]
            + 0.18 * workload_h1
            + C3_WEIGHT * c3[workload_active],
            0.001,
            0.999,
        )
        workload_delta = workload_proposal - jy_probability[workload_active]
        output[workload_active] = np.clip(
            output[workload_active] + workload_delta,
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''

NEW_TAIL = '''    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    if np.any(anchor):
        if lowrank_probability_delta is None:
            lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[anchor] = np.clip(
            output[anchor] + 0.50 * lowrank_probability_delta[anchor],
            0.001,
            0.999,
        )
    v335_probability = output.copy()
    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
            0.001,
            0.999,
        )
    workload_active = active & rcore
    if np.any(workload_active):
        workload_h1 = _predict_workload_h1(
            frame.loc[workload_active].reset_index(drop=True)
        )
        workload_proposal = np.clip(
            0.82 * effective_bridge[workload_active]
            + 0.18 * workload_h1
            + C3_WEIGHT * c3[workload_active],
            0.001,
            0.999,
        )
        workload_delta = workload_proposal - jy_probability[workload_active]
        output[workload_active] = np.clip(
            output[workload_active] + workload_delta,
            0.001,
            0.999,
        )
    beta_active = _beta_cell_mask(frame)
    if np.any(beta_active):
        beta_probability = _predict_beta_cell(
            frame.loc[beta_active].reset_index(drop=True)
        )
        beta_delta = 0.10 * (
            beta_probability - v335_probability[beta_active]
        )
        output[beta_active] = np.clip(
            output[beta_active] + beta_delta,
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def patch_script(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("Beta function insertion anchor is not unique")
    if source.count(OLD_TAIL) != 1:
        raise ValueError("v343 route tail is not unique")
    output = source.replace(FUNCTION_ANCHOR, BETA_FUNCTIONS + FUNCTION_ANCHOR)
    output = output.replace(OLD_TAIL, NEW_TAIL)
    if output.count("def _predict_beta_cell") != 1:
        raise ValueError("Beta runtime insertion failed")
    return output


def bundle_zip_info() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        BUNDLE_MEMBER,
        date_time=(2026, 8, 31, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(
    train: pd.DataFrame,
    prediction_year: int = 2025,
) -> dict[str, Any]:
    fitted = fit_previous_year_pool(train, prediction_year)
    previous = train.loc[train["season"].eq(prediction_year - 1)]
    history = train.loc[train["season"].lt(prediction_year)]
    pitcher = terminal_snapshot(
        history,
        "pitcher_id",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
    )
    batter = terminal_snapshot(
        history,
        "batter_id",
        "asof_batter_n",
        "asof_batter_success_rate",
    )
    return {
        "protocol": "V345_BETA_CELL_RUNTIME_BUNDLE_V1",
        "prediction_year": int(prediction_year),
        "calibration_year": int(fitted["calibration_year"]),
        "concentration": float(fitted["concentration"]),
        "weights": np.asarray(fitted["weights"], dtype=np.float64),
        "candidate_names": list(CANDIDATE_NAMES),
        "global_prior": float(history["control_success"].mean()),
        "prior_by_game_type": {
            str(key): float(value)
            for key, value in previous.groupby("game_type", observed=True)[
                "control_success"
            ].mean().items()
        },
        "pitcher_opening_n": {key: value[0] for key, value in pitcher.items()},
        "pitcher_opening_success": {
            key: value[1] for key, value in pitcher.items()
        },
        "batter_opening_n": {key: value[0] for key, value in batter.items()},
        "batter_opening_success": {key: value[1] for key, value in batter.items()},
    }


def _bundle_season_posterior(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    prefix: str,
    prior: np.ndarray,
) -> np.ndarray:
    identifier = pd.to_numeric(
        frame[f"{prefix}_id"], errors="raise"
    ).astype("int64")
    opening_n = identifier.map(bundle[f"{prefix}_opening_n"]).fillna(0.0).to_numpy(np.float64)
    opening_success = identifier.map(
        bundle[f"{prefix}_opening_success"]
    ).fillna(0.0).to_numpy(np.float64)
    career_n = pd.to_numeric(
        frame[f"asof_{prefix}_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    career_rate = pd.to_numeric(
        frame[f"asof_{prefix}_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    season_n = np.maximum(career_n - opening_n, 0.0)
    season_success = np.clip(
        career_n * career_rate - opening_success, 0.0, season_n
    )
    concentration = float(bundle["concentration"])
    return (season_success + concentration * prior) / (season_n + concentration)


def predict_beta_bundle(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    prior = frame["game_type"].astype(str).map(
        bundle["prior_by_game_type"]
    ).fillna(float(bundle["global_prior"])).to_numpy(np.float64)
    pitcher = _bundle_season_posterior(frame, bundle, "pitcher", prior)
    batter = _bundle_season_posterior(frame, bundle, "batter", prior)
    career = pd.to_numeric(
        frame["asof_pitcher_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    career = np.where(np.isfinite(career), career, prior)
    recent = pd.to_numeric(
        frame["asof_pitcher_prev5_game_success_rate"], errors="coerce"
    ).to_numpy(np.float64)
    recent = np.where(np.isfinite(recent), recent, career)
    matrix = np.clip(
        np.column_stack((pitcher, batter, prior, career, recent)),
        0.001,
        0.999,
    )
    return np.clip(
        matrix @ np.asarray(bundle["weights"], dtype=np.float64),
        0.001,
        0.999,
    )


def historical_runtime_parity(
    train: pd.DataFrame,
    beta_axes: Path,
) -> dict[str, Any]:
    bundle = build_bundle(train, prediction_year=2024)
    frame = train.loc[train["season"].eq(2024)].reset_index(drop=True)
    runtime = predict_beta_bundle(frame, bundle)
    with np.load(beta_axes, allow_pickle=False) as saved:
        expected = saved["beta_full_2024"].astype(np.float64)
    if len(runtime) != len(expected):
        raise ValueError("historical Beta parity axis length mismatch")
    max_abs = float(np.max(np.abs(runtime - expected)))
    if max_abs > 1e-12:
        raise ValueError(f"historical Beta runtime parity failed: {max_abs}")
    return {
        "prediction_year": 2024,
        "rows": int(len(runtime)),
        "max_abs": max_abs,
        "status": "pass",
    }


def run(
    source_zip: Path,
    train_csv: Path,
    beta_axes: Path,
    audit_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V345_TRANSITION_WORKLOAD_BETA_CELL_V335_V1":
        raise ValueError("unexpected v345 audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("v345 candidate gate did not pass")
    train = pd.read_csv(train_csv, encoding="utf-8-sig", low_memory=False)
    beta_parity = historical_runtime_parity(train, beta_axes)
    bundle = build_bundle(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "beta_cell_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    output_zip = output_dir / "submit_v345.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        if BUNDLE_MEMBER in names:
            raise ValueError("source ZIP already contains a Beta bundle")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for item in source.infolist():
                payload = (
                    patched.encode("utf-8")
                    if item.filename == "script.py"
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
            target.writestr(bundle_zip_info(), bundle_path.read_bytes())
    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        members = built.namelist()
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": len(members),
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_zip_sha256": sha256(source_zip),
        "bundle_sha256": sha256(bundle_path),
        "historical_beta_runtime_parity": beta_parity,
        "bundle": {
            "calibration_year": bundle["calibration_year"],
            "concentration": bundle["concentration"],
            "weights": bundle["weights"].tolist(),
            "candidate_names": bundle["candidate_names"],
            "pitcher_snapshots": len(bundle["pitcher_opening_n"]),
            "batter_snapshots": len(bundle["batter_opening_n"]),
        },
        "recipe": audit["recipe"],
        "local_metrics_vs_v335": audit["metrics_vs_v335"],
        "restrictions": {
            **audit["restrictions"],
            "beta_pool_fitted_on_2024_only": True,
            "terminal_snapshots_fit_through_2024": True,
            "runtime_audit_pending": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--beta-axes", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_zip,
                args.train_csv,
                args.beta_axes,
                args.audit_summary,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
