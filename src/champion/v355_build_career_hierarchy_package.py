"""Build v355 by replacing v354's latest prior with source-selected career prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

import joblib
import numpy as np
import pandas as pd

from src.archive.v39_hierarchical_season_forecast import forecast_bank
from src.champion.v354_build_late_hierarchy_package import (
    BUNDLE_MEMBER,
    _domain,
    _stat_dict,
    predict_bundle,
    sha256,
)
from src.temporal_stable_conditional import _add_domain_and_pressure


PROTOCOL = "V355_BUILD_CAREER_HIERARCHY_PACKAGE_V1"


def build_career_bundle(
    train: pd.DataFrame, prediction_year: int = 2025
) -> dict[str, Any]:
    history = train.loc[train["season"].lt(prediction_year)].copy()
    if history.empty:
        raise ValueError("career hierarchy requires prior history")
    latest_year = int(history["season"].max())
    latest = history.loc[history["season"].eq(latest_year)].copy()
    history["domain3"] = _domain(history)
    latest["domain3"] = _domain(latest)
    bundle: dict[str, Any] = {
        "protocol": "V355_CAREER_HIERARCHY_RUNTIME_BUNDLE_V1",
        "prediction_year": int(prediction_year),
        "latest_year": latest_year,
        "latest_global": float(latest["control_success"].mean()),
        "prior_by_domain": {
            str(key): float(value)
            for key, value in latest.groupby("domain3", observed=True)[
                "control_success"
            ].mean().items()
        },
        "signal": "hier::domain_career_k80_p75",
        "start_month": 8,
        "weight": 0.20,
    }
    for entity in ("pitcher_id", "batter_id"):
        historical = history.groupby(entity, observed=True)["control_success"].agg(
            n="size", s="sum"
        )
        career = history.groupby([entity, "domain3"], observed=True)[
            "control_success"
        ].agg(n="size", s="sum")
        bundle[f"{entity}_history_n"] = _stat_dict(historical, "n")
        bundle[f"{entity}_history_s"] = _stat_dict(historical, "s")
        # The inherited runtime uses prior strengths 80/120.  Career priors
        # use 160/240, exactly twice as large, so half-scaled sufficient
        # statistics reproduce (s + alpha*p)/(n + alpha) without code drift.
        bundle[f"{entity}_latest_n"] = {
            key: 0.5 * float(value) for key, value in career["n"].items()
        }
        bundle[f"{entity}_latest_s"] = {
            key: 0.5 * float(value) for key, value in career["s"].items()
        }
    return bundle


def historical_runtime_parity(train: pd.DataFrame) -> dict[str, Any]:
    prepared = _add_domain_and_pressure(train.copy())
    frame = prepared.loc[prepared["season"].eq(2024)].reset_index(drop=True)
    expected = forecast_bank(prepared, 2024)["hier::domain_career_k80_p75"]
    runtime = predict_bundle(frame, build_career_bundle(train, prediction_year=2024))
    max_abs = float(np.max(np.abs(runtime - expected)))
    if max_abs > 1e-12:
        raise ValueError(f"historical career hierarchy parity failed: {max_abs}")
    return {
        "prediction_year": 2024,
        "rows": int(len(frame)),
        "max_abs": max_abs,
        "status": "pass",
    }


def run(
    source_zip: Path,
    train_csv: Path,
    audit_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V355_LATE_HIERARCHY_SOURCE_SELECTION_V1":
        raise ValueError("unexpected v355 audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("v355 candidate gate did not pass")
    if audit["recipe"]["signal"] != "hier::domain_career_k80_p75":
        raise ValueError("v355 source selection did not choose the career hierarchy")
    train = pd.read_csv(train_csv, low_memory=False)
    parity = historical_runtime_parity(train)
    bundle = build_career_bundle(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "career_hierarchy_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    output_zip = output_dir / "submit_v355.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        if names.count(BUNDLE_MEMBER) != 1:
            raise ValueError("source ZIP must contain one v354 hierarchy bundle")
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for item in source.infolist():
                payload = (
                    bundle_path.read_bytes()
                    if item.filename == BUNDLE_MEMBER
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
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
        "source_selection": audit["selection"],
        "locked_full_2024": audit["locked_full_2024"],
        "gain_over_v354": audit["gain_over_v354"],
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
