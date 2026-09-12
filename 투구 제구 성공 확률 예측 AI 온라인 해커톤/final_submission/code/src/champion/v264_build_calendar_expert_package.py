"""Build the standalone v261 April--September command+batter candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


PROTOCOL = "V264_BUILD_CALENDAR_EXPERT_PACKAGE_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script(text: str) -> str:
    text = text.replace("\r\n", "\n")
    marker = "\ndef _window_adjustment(\n"
    helper = '''
def _predict_calendar_experts(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    base_module = _load_module(
        "jy_fallback_xgb_for_experts", MODEL_DIR / "fallback_xgb" / "runtime.py"
    )
    module = _load_module(
        "calendar_experts", MODEL_DIR / "calendar_experts" / "runtime.py"
    )
    return module.predict(
        frame,
        MODEL_DIR / "calendar_experts",
        base_module,
        MODEL_DIR / "fallback_xgb",
    )

'''
    if marker not in text:
        raise RuntimeError("champion helper marker missing")
    text = text.replace(marker, helper + marker)
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
    v244_output = jy_probability.copy()
    for mask, weight in routes:
        v244_output[mask] = np.clip(
            jy_probability[mask]
            + weight * (xgb_probability[mask] - jy_probability[mask]),
            0.001,
            0.999,
        )
    route_active = np.logical_or.reduce([mask for mask, _weight in routes])
    calendar = frame["game_month"].between(4, 9).to_numpy()
    expert_active = route_active & calendar
    output = v244_output
    if np.any(expert_active):
        command_probability, batter_probability = _predict_calendar_experts(frame)
        command_scope = expert_active & (
            np.abs(command_probability - 0.5) < np.abs(xgb_probability - 0.5)
        )
        batter_scope = expert_active & ~command_scope
        expert_fallback = xgb_probability.copy()
        expert_fallback[command_scope] = (
            0.40 * xgb_probability[command_scope]
            + 0.60 * command_probability[command_scope]
        )
        expert_fallback[batter_scope] = (
            0.75 * xgb_probability[batter_scope]
            + 0.25 * batter_probability[batter_scope]
        )
        output = v244_output.copy()
        for mask, weight in routes:
            selected = mask & calendar
            output[selected] = np.clip(
                jy_probability[selected]
                + weight * (expert_fallback[selected] - jy_probability[selected]),
                0.001,
                0.999,
            )
    return parent, h1, c3, active, output
'''
    return text[:start] + replacement + text[end:]


def run(
    source_package: Path,
    finalized_models: Path,
    runtime_source: Path,
    v261_summary: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_summary = json.loads(
        (finalized_models / "summary.json").read_text(encoding="utf-8")
    )
    if final_summary.get("status") != "final_models_ready":
        raise ValueError("v263 finalized models are not ready")
    local_summary = json.loads(v261_summary.read_text(encoding="utf-8"))
    if local_summary.get("protocol") != "V261_REGULAR_CALENDAR_EXPERT_GATE_V1":
        raise ValueError("unexpected v261 summary protocol")
    output_zip = output_dir / "submit_v264_calendar_experts.zip"
    with tempfile.TemporaryDirectory(prefix="v264_calendar_experts_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(source_package) as archive:
            archive.extractall(stage)
        script = stage / "script.py"
        script.write_text(
            _patch_script(script.read_text(encoding="utf-8")), encoding="utf-8"
        )
        asset = stage / "model" / "calendar_experts"
        asset.mkdir(parents=True, exist_ok=True)
        names = (
            "command_expert_xgb.json",
            "batter_expert_xgb.json",
            "command_feature_columns.json",
            "batter_feature_columns.json",
            "command_profiles.joblib",
            "batter_profiles.joblib",
        )
        for name in names:
            source = finalized_models / name
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, asset / name)
        shutil.copy2(runtime_source, asset / "runtime.py")
        metadata = {
            "protocol": PROTOCOL,
            "source_package_sha256": _sha256(source_package),
            "v263_summary_sha256": _sha256(finalized_models / "summary.json"),
            "v261_summary_sha256": _sha256(v261_summary),
            "calendar_window": [4, 9],
            "command_weight": 0.60,
            "batter_weight": 0.25,
            "test_row_aggregation": False,
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
        "command_weight": 0.60,
        "batter_weight": 0.25,
        "source_v244_unchanged": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--finalized-models", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--v261-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_package,
        args.finalized_models,
        args.runtime_source,
        args.v261_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
