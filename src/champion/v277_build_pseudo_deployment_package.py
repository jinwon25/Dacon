"""Build the standalone v271 pseudo-deployment calendar challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


PROTOCOL = "V277_BUILD_PSEUDO_DEPLOYMENT_PACKAGE_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script(text: str) -> str:
    text = text.replace("\r\n", "\n")
    marker = "def _window_adjustment(\n"
    helper = '''
def _predict_pseudo_deployment(frame: pd.DataFrame) -> np.ndarray:
    base_module = _load_module(
        "jy_fallback_xgb_for_pseudo", MODEL_DIR / "fallback_xgb" / "runtime.py"
    )
    module = _load_module(
        "pseudo_deployment", MODEL_DIR / "pseudo_deployment" / "runtime.py"
    )
    return module.predict(
        frame,
        MODEL_DIR / "pseudo_deployment",
        base_module,
        MODEL_DIR / "fallback_xgb",
    )

'''
    if marker not in text:
        raise RuntimeError("champion helper marker missing")
    text = text.replace(marker, helper + marker, 1)
    start = text.index("    jy_probability = output.copy()\n")
    end_marker = "    return parent, h1, c3, active, output\n"
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
    route_active = np.logical_or.reduce([mask for mask, _weight in routes])
    calendar = frame["game_month"].between(4, 9).to_numpy()
    pseudo_selected = route_active & calendar
    expert_fallback = xgb_probability.copy()
    if np.any(pseudo_selected):
        pseudo_probability = _predict_pseudo_deployment(frame)
        expert_fallback[pseudo_selected] = (
            0.50 * xgb_probability[pseudo_selected]
            + 0.50 * pseudo_probability[pseudo_selected]
        )
    for mask, weight in routes:
        output[mask] = np.clip(
            jy_probability[mask]
            + weight * (expert_fallback[mask] - jy_probability[mask]),
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
    return text[:start] + replacement + text[end:]


def run(
    source_package: Path,
    finalized_model_dir: Path,
    runtime_source: Path,
    v271_summary: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = json.loads(
        (finalized_model_dir / "summary.json").read_text(encoding="utf-8")
    )
    if final_summary.get("status") != "final_model_ready":
        raise ValueError("v276 finalized model is not ready")
    local_summary = json.loads(v271_summary.read_text(encoding="utf-8"))
    if local_summary.get("protocol") != "V271_PSEUDO_DEPLOYMENT_CALENDAR_AUDIT_V1":
        raise ValueError("unexpected v271 summary protocol")
    if not local_summary.get("eligible_for_exploratory_packaging"):
        raise ValueError("v271 is not eligible for exploratory packaging")

    output_zip = output_dir / "submit_v277_pseudo_deployment_calendar.zip"
    with tempfile.TemporaryDirectory(prefix="v277_pseudo_deployment_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(source_package) as archive:
            archive.extractall(stage)
        script = stage / "script.py"
        script.write_text(
            _patch_script(script.read_text(encoding="utf-8")), encoding="utf-8"
        )
        asset = stage / "model" / "pseudo_deployment"
        asset.mkdir(parents=True, exist_ok=True)
        for name in ("pseudo_deployment_xgb.json", "feature_columns.json"):
            source = finalized_model_dir / name
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, asset / name)
        shutil.copy2(runtime_source, asset / "runtime.py")
        metadata = {
            "protocol": PROTOCOL,
            "source_package_sha256": _sha256(source_package),
            "v276_summary_sha256": _sha256(finalized_model_dir / "summary.json"),
            "v271_summary_sha256": _sha256(v271_summary),
            "calendar_window": [4, 9],
            "pseudo_weight": 0.50,
            "test_row_aggregation": False,
            "exploratory_challenger": True,
        }
        (asset / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in stage.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    with zipfile.ZipFile(output_zip) as archive:
        bad = archive.testzip()
        entries = archive.namelist()
    if bad is not None:
        raise RuntimeError(f"zip CRC failure: {bad}")
    result = {
        "protocol": PROTOCOL,
        "status": "package_built_pending_standalone_audit",
        "package": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "sha256": _sha256(output_zip),
        "file_count": len(entries),
        "crc_passed": True,
        "calendar_window": [4, 9],
        "pseudo_weight": 0.50,
        "source_v244_unchanged": True,
        "exploratory_challenger": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--finalized-model-dir", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--v271-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_package,
        args.finalized_model_dir,
        args.runtime_source,
        args.v271_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
