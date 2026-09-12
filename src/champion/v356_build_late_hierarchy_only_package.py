"""Build submit_v356 by adding only the frozen late hierarchy to exact v345."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

import joblib

from src.champion.v354_build_late_hierarchy_package import (
    BUNDLE_MEMBER,
    HIERARCHY_FUNCTIONS,
    build_bundle,
    bundle_zip_info,
    historical_runtime_parity,
    sha256,
)


PROTOCOL = "V356_BUILD_LATE_HIERARCHY_ONLY_PACKAGE_V1"
EXPECTED_SOURCE_SHA256 = (
    "D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA"
)
FUNCTION_ANCHOR = "def _predict_player_transition(frame: pd.DataFrame) -> np.ndarray:\n"

OLD_TAIL = '''    beta_active = _beta_cell_mask(frame)
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

NEW_TAIL = '''    beta_active = _beta_cell_mask(frame)
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


def patch_script(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if "def _predict_trackman_pfd" in source:
        raise ValueError("source unexpectedly contains the rejected TrackMan branch")
    if "def _predict_late_hierarchy" in source or BUNDLE_MEMBER in source:
        raise ValueError("source already contains a late-hierarchy branch")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("v345 function anchor is not unique")
    if source.count(OLD_TAIL) != 1:
        raise ValueError("v345 route tail is not unique")
    output = source.replace(FUNCTION_ANCHOR, HIERARCHY_FUNCTIONS + FUNCTION_ANCHOR)
    output = output.replace(OLD_TAIL, NEW_TAIL)
    if output.count("def _predict_late_hierarchy") != 1:
        raise ValueError("late hierarchy runtime insertion failed")
    return output


def _validate_audit(audit: dict[str, Any]) -> None:
    if audit.get("protocol") != "V354_LATE_HIERARCHY_REBASE_V345_V1":
        raise ValueError("unexpected late-hierarchy audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("late-hierarchy candidate gate did not pass")
    metrics = audit.get("metrics", {})
    required_axes = ("full_2022", "late_2023", "full_2024")
    if any(float(metrics.get(axis, {}).get("gain", 0.0)) <= 0.0 for axis in required_axes):
        raise ValueError("late hierarchy is not positive on every required time axis")
    robust = audit.get("locked_robustness", {})
    required_bootstraps = ("pitcher", "crossed_pitcher_batter", "chronological_block")
    if any(float(robust.get(axis, {}).get("p05", 0.0)) <= 0.0 for axis in required_bootstraps):
        raise ValueError("late hierarchy failed a locked robustness lower bound")


def _validate_member_diff(source_zip: Path, output_zip: Path) -> dict[str, Any]:
    with zipfile.ZipFile(source_zip) as source, zipfile.ZipFile(output_zip) as output:
        source_names = source.namelist()
        output_names = output.namelist()
        if output_names != source_names + [BUNDLE_MEMBER]:
            raise ValueError("output ZIP member order or additions are invalid")
        changed = []
        for name in source_names:
            if source.read(name) != output.read(name):
                changed.append(name)
        if changed != ["script.py"]:
            raise ValueError(f"unexpected changed ZIP members: {changed}")
        if output.testzip() is not None:
            raise ValueError("output ZIP failed CRC validation")
    return {
        "changed_members": changed,
        "added_members": [BUNDLE_MEMBER],
        "member_order_preserved": True,
    }


def run(
    source_zip: Path,
    train_csv: Path,
    audit_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_hash = sha256(source_zip)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"source is not exact v345: {source_hash}")
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    _validate_audit(audit)
    import pandas as pd

    train = pd.read_csv(train_csv, low_memory=False)
    parity = historical_runtime_parity(train)
    bundle = build_bundle(train)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "late_hierarchy_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    output_zip = output_dir / "submit_v356.zip"
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
                payload = (
                    patched.encode("utf-8")
                    if item.filename == "script.py"
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
            target.writestr(bundle_zip_info(), bundle_path.read_bytes())
    member_diff = _validate_member_diff(source_zip, output_zip)
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "source_zip": str(source_zip),
        "source_zip_sha256": source_hash,
        "bundle_sha256": sha256(bundle_path),
        "historical_runtime_parity": parity,
        "member_diff": member_diff,
        "recipe": {
            "parent": "exact submit_v345.zip / Public 1182.94969702",
            "increment": "hier::domain_latest_k80_p75",
            "domain": "R_ANCHOR",
            "start_month": 8,
            "weight_toward_hierarchy": 0.20,
            "trackman_component_included": False,
        },
        "local_metrics": audit["metrics"],
        "locked_robustness": audit["locked_robustness"],
        "restrictions": {
            **audit["restrictions"],
            "exact_v345_parent": True,
            "trackman_component_included": False,
            "public_score_used_for_component_selection": False,
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
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.source_zip, args.train_csv, args.audit_summary, args.output_dir),
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
