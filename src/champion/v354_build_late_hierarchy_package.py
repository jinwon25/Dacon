"""Build v354 by adding a frozen late-season hierarchy to v353."""

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

from src.archive.v39_hierarchical_season_forecast import forecast_bank
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V354_BUILD_LATE_HIERARCHY_PACKAGE_V1"
BUNDLE_MEMBER = "model/late_hierarchy/spec.joblib"
FUNCTION_ANCHOR = "_TRACKMAN_PFD_FEATURES = [\n"

HIERARCHY_FUNCTIONS = '''def _late_hierarchy_domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _late_hierarchy_entity(
    frame: pd.DataFrame,
    bundle: dict,
    entity: str,
    domain: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    identifier = pd.to_numeric(frame[entity], errors="raise").astype("int64")
    history_n = identifier.map(bundle[f"{entity}_history_n"]).fillna(0.0).to_numpy(np.float64)
    history_s = identifier.map(bundle[f"{entity}_history_s"]).fillna(0.0).to_numpy(np.float64)
    prefix = "pitcher" if entity == "pitcher_id" else "batter"
    cumulative_n = pd.to_numeric(
        frame[f"asof_{prefix}_n"], errors="coerce"
    ).fillna(0.0).to_numpy(np.float64)
    cumulative_rate = pd.to_numeric(
        frame[f"asof_{prefix}_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    cumulative_s = np.zeros(len(frame), dtype=np.float64)
    valid = (cumulative_n > 0.0) & np.isfinite(cumulative_rate)
    cumulative_s[valid] = np.rint(cumulative_n[valid] * cumulative_rate[valid])
    season_n = np.maximum(cumulative_n - history_n, 0.0)
    season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
    keys = zip(identifier.to_numpy(np.int64), domain)
    latest_n = np.fromiter(
        (bundle[f"{entity}_latest_n"].get((int(key), str(route)), 0.0) for key, route in keys),
        dtype=np.float64,
        count=len(frame),
    )
    keys = zip(identifier.to_numpy(np.int64), domain)
    latest_s = np.fromiter(
        (bundle[f"{entity}_latest_s"].get((int(key), str(route)), 0.0) for key, route in keys),
        dtype=np.float64,
        count=len(frame),
    )
    latest_alpha = 80.0 if entity == "pitcher_id" else 120.0
    latest_rate = (latest_s + latest_alpha * prior) / (latest_n + latest_alpha)
    return (season_s + 80.0 * latest_rate) / (season_n + 80.0)


def _predict_late_hierarchy(frame: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(MODEL_DIR / "late_hierarchy" / "spec.joblib")
    domain = _late_hierarchy_domain(frame)
    prior = pd.Series(domain).map(bundle["prior_by_domain"]).fillna(
        float(bundle["latest_global"])
    ).to_numpy(np.float64)
    pitcher = _late_hierarchy_entity(
        frame, bundle, "pitcher_id", domain, prior
    )
    batter = _late_hierarchy_entity(
        frame, bundle, "batter_id", domain, prior
    )
    return np.clip(0.75 * pitcher + 0.25 * batter, 0.001, 0.999)


'''

OLD_TAIL = '''    trackman_active = anchor
    if np.any(trackman_active):
        trackman_correction = _predict_trackman_pfd(
            frame.loc[trackman_active].reset_index(drop=True)
        )
        output[trackman_active] = np.clip(
            output[trackman_active] + 0.40 * trackman_correction,
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''

NEW_TAIL = '''    trackman_active = anchor
    if np.any(trackman_active):
        trackman_correction = _predict_trackman_pfd(
            frame.loc[trackman_active].reset_index(drop=True)
        )
        output[trackman_active] = np.clip(
            output[trackman_active] + 0.40 * trackman_correction,
            0.001,
            0.999,
        )
    game_month = pd.to_numeric(
        frame["game_month"], errors="coerce"
    ).fillna(0).to_numpy(np.int16)
    hierarchy_active = anchor & (game_month >= 8)
    if np.any(hierarchy_active):
        hierarchy_probability = _predict_late_hierarchy(
            frame.loc[hierarchy_active].reset_index(drop=True)
        )
        output[hierarchy_active] = np.clip(
            0.80 * output[hierarchy_active] + 0.20 * hierarchy_probability,
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


def _domain(frame: pd.DataFrame) -> pd.Series:
    regular = frame["game_type"].astype(str).eq("R")
    anchor = regular & (
        frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    )
    return pd.Series(
        np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE")),
        index=frame.index,
        dtype="string",
    )


def _stat_dict(table: pd.DataFrame, column: str) -> dict[Any, float]:
    return {key: float(value) for key, value in table[column].items()}


def build_bundle(train: pd.DataFrame, prediction_year: int = 2025) -> dict[str, Any]:
    history = train.loc[train["season"].lt(prediction_year)].copy()
    if history.empty:
        raise ValueError("late hierarchy requires prior history")
    latest_year = int(history["season"].max())
    latest = history.loc[history["season"].eq(latest_year)].copy()
    history["domain3"] = _domain(history)
    latest["domain3"] = _domain(latest)
    bundle: dict[str, Any] = {
        "protocol": "V354_LATE_HIERARCHY_RUNTIME_BUNDLE_V1",
        "prediction_year": int(prediction_year),
        "latest_year": latest_year,
        "latest_global": float(latest["control_success"].mean()),
        "prior_by_domain": {
            str(key): float(value)
            for key, value in latest.groupby("domain3", observed=True)[
                "control_success"
            ].mean().items()
        },
        "signal": "hier::domain_latest_k80_p75",
        "start_month": 8,
        "weight": 0.20,
    }
    for entity in ("pitcher_id", "batter_id"):
        historical = history.groupby(entity, observed=True)["control_success"].agg(
            n="size", s="sum"
        )
        recent = latest.groupby([entity, "domain3"], observed=True)[
            "control_success"
        ].agg(n="size", s="sum")
        bundle[f"{entity}_history_n"] = _stat_dict(historical, "n")
        bundle[f"{entity}_history_s"] = _stat_dict(historical, "s")
        bundle[f"{entity}_latest_n"] = _stat_dict(recent, "n")
        bundle[f"{entity}_latest_s"] = _stat_dict(recent, "s")
    return bundle


def _mapped_pair_stat(
    identifier: np.ndarray,
    domain: np.ndarray,
    table: dict[tuple[int, str], float],
) -> np.ndarray:
    return np.fromiter(
        (table.get((int(key), str(route)), 0.0) for key, route in zip(identifier, domain)),
        dtype=np.float64,
        count=len(identifier),
    )


def _predict_entity_bundle(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    entity: str,
    domain: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    identifier = pd.to_numeric(frame[entity], errors="raise").astype("int64")
    history_n = identifier.map(bundle[f"{entity}_history_n"]).fillna(0.0).to_numpy(np.float64)
    history_s = identifier.map(bundle[f"{entity}_history_s"]).fillna(0.0).to_numpy(np.float64)
    prefix = "pitcher" if entity == "pitcher_id" else "batter"
    cumulative_n = pd.to_numeric(frame[f"asof_{prefix}_n"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    cumulative_rate = pd.to_numeric(frame[f"asof_{prefix}_success_rate"], errors="coerce").fillna(0.5).to_numpy(np.float64)
    cumulative_s = np.zeros(len(frame), dtype=np.float64)
    valid = (cumulative_n > 0.0) & np.isfinite(cumulative_rate)
    cumulative_s[valid] = np.rint(cumulative_n[valid] * cumulative_rate[valid])
    season_n = np.maximum(cumulative_n - history_n, 0.0)
    season_s = np.clip(cumulative_s - history_s, 0.0, season_n)
    ids = identifier.to_numpy(np.int64)
    latest_n = _mapped_pair_stat(ids, domain, bundle[f"{entity}_latest_n"])
    latest_s = _mapped_pair_stat(ids, domain, bundle[f"{entity}_latest_s"])
    latest_alpha = 80.0 if entity == "pitcher_id" else 120.0
    latest_rate = (latest_s + latest_alpha * prior) / (latest_n + latest_alpha)
    return (season_s + 80.0 * latest_rate) / (season_n + 80.0)


def predict_bundle(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    domain = _domain(frame).astype(str).to_numpy()
    prior = pd.Series(domain).map(bundle["prior_by_domain"]).fillna(
        float(bundle["latest_global"])
    ).to_numpy(np.float64)
    pitcher = _predict_entity_bundle(frame, bundle, "pitcher_id", domain, prior)
    batter = _predict_entity_bundle(frame, bundle, "batter_id", domain, prior)
    return np.clip(0.75 * pitcher + 0.25 * batter, 0.001, 0.999)


def historical_runtime_parity(train: pd.DataFrame) -> dict[str, Any]:
    prepared = _add_domain_and_pressure(train.copy())
    frame = prepared.loc[prepared["season"].eq(2024)].reset_index(drop=True)
    expected = forecast_bank(prepared, 2024)["hier::domain_latest_k80_p75"]
    runtime = predict_bundle(frame, build_bundle(train, prediction_year=2024))
    max_abs = float(np.max(np.abs(runtime - expected)))
    if max_abs > 1e-12:
        raise ValueError(f"historical late-hierarchy parity failed: {max_abs}")
    return {"prediction_year": 2024, "rows": len(frame), "max_abs": max_abs, "status": "pass"}


def patch_script(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("TrackMan feature anchor is not unique")
    if source.count(OLD_TAIL) != 1:
        raise ValueError("v353 route tail is not unique")
    output = source.replace(FUNCTION_ANCHOR, HIERARCHY_FUNCTIONS + FUNCTION_ANCHOR)
    output = output.replace(OLD_TAIL, NEW_TAIL)
    if output.count("def _predict_late_hierarchy") != 1:
        raise ValueError("late hierarchy runtime insertion failed")
    return output


def bundle_zip_info() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(BUNDLE_MEMBER, date_time=(2026, 9, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def run(
    source_zip: Path,
    train_csv: Path,
    audit_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V354_LATE_HIERARCHY_REBASE_V345_V1":
        raise ValueError("unexpected v354 audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("v354 candidate gate did not pass")
    train = pd.read_csv(train_csv, low_memory=False)
    parity = historical_runtime_parity(train)
    bundle = build_bundle(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "late_hierarchy_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    output_zip = output_dir / "submit_v354.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        if BUNDLE_MEMBER in names:
            raise ValueError("source ZIP already contains a late hierarchy bundle")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for item in source.infolist():
                payload = patched.encode("utf-8") if item.filename == "script.py" else source.read(item.filename)
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
        "historical_runtime_parity": parity,
        "bundle": {
            "latest_year": bundle["latest_year"],
            "pitchers": len(bundle["pitcher_id_history_n"]),
            "batters": len(bundle["batter_id_history_n"]),
            "pitcher_domain_cells": len(bundle["pitcher_id_latest_n"]),
            "batter_domain_cells": len(bundle["batter_id_latest_n"]),
        },
        "recipe": audit["recipe"],
        "local_metrics": audit["metrics"],
        "restrictions": {**audit["restrictions"], "runtime_audit_pending": True},
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
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_zip, args.train_csv, args.audit_summary, args.output_dir), ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
