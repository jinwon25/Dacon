"""Build v352 by adding the frozen v38 TrackMan-PFD student to v345."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


PROTOCOL = "V352_BUILD_TRACKMAN_PFD_PACKAGE_V1"
MODEL_NAMES = (
    "refit_without_ids_s1816.txt",
    "refit_without_ids_s1916.txt",
    "refit_without_ids_s2016.txt",
)
MODEL_ROOT = "model/trackman_pfd"
IMPORT_ANCHOR = "import joblib\nimport numpy as np\nimport pandas as pd\n"
FUNCTION_ANCHOR = "def _beta_cell_mask(frame: pd.DataFrame) -> np.ndarray:\n"

TRACKMAN_FUNCTIONS = '''_TRACKMAN_PFD_FEATURES = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_hand", "batter_hand",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n",
    "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
_TRACKMAN_PFD_CATEGORICAL = {
    "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand",
}
_TRACKMAN_PFD_MODELS = (
    "refit_without_ids_s1816.txt",
    "refit_without_ids_s1916.txt",
    "refit_without_ids_s2016.txt",
)


def _trackman_pfd_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[_TRACKMAN_PFD_FEATURES].copy()
    for column in _TRACKMAN_PFD_FEATURES:
        if column in _TRACKMAN_PFD_CATEGORICAL:
            output[column] = (
                output[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype("category")
            )
        else:
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).astype(np.float32)
    return output


def _predict_trackman_pfd(frame: pd.DataFrame) -> np.ndarray:
    features = _trackman_pfd_frame(frame)
    predictions = []
    for name in _TRACKMAN_PFD_MODELS:
        model_text = (MODEL_DIR / "trackman_pfd" / name).read_text(
            encoding="utf-8"
        )
        booster = lgb.Booster(model_str=model_text)
        predictions.append(
            booster.predict(features, num_threads=1).astype(np.float64)
        )
    return np.mean(np.vstack(predictions), axis=0)


'''

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
    trackman_active = anchor
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def patch_script(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if source.count(IMPORT_ANCHOR) != 1:
        raise ValueError("runtime import anchor is not unique")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("runtime function anchor is not unique")
    if source.count(OLD_TAIL) != 1:
        raise ValueError("v345 route tail is not unique")
    output = source.replace(
        IMPORT_ANCHOR,
        IMPORT_ANCHOR.replace("import joblib\n", "import joblib\nimport lightgbm as lgb\n"),
    )
    output = output.replace(FUNCTION_ANCHOR, TRACKMAN_FUNCTIONS + FUNCTION_ANCHOR)
    output = output.replace(OLD_TAIL, NEW_TAIL)
    if output.count("def _predict_trackman_pfd") != 1:
        raise ValueError("TrackMan-PFD runtime insertion failed")
    return output


def model_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        f"{MODEL_ROOT}/{name}",
        date_time=(2026, 8, 31, 0, 0, 0),
    )
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def normalised_model_bytes(path: Path) -> bytes:
    """Undo Windows newline expansion that invalidates LightGBM tree offsets."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")


def run(
    source_zip: Path,
    model_dir: Path,
    audit_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V352_TRACKMAN_PFD_REBASE_V345_V1":
        raise ValueError("unexpected v352 audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("v352 candidate gate did not pass")
    model_paths = [model_dir / name for name in MODEL_NAMES]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing TrackMan-PFD models: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v352.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        collisions = [f"{MODEL_ROOT}/{name}" for name in MODEL_NAMES if f"{MODEL_ROOT}/{name}" in names]
        if collisions:
            raise ValueError(f"source ZIP already contains v352 models: {collisions}")
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
            for name, path in zip(MODEL_NAMES, model_paths):
                target.writestr(model_zip_info(name), normalised_model_bytes(path))

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
        "models": [
            {
                "name": path.name,
                "source_sha256": sha256(path),
                "normalised_bytes": len(normalised_model_bytes(path)),
            }
            for path in model_paths
        ],
        "recipe": audit["recipe"],
        "local_metrics_vs_v345": audit["metrics_vs_v345"],
        "restrictions": {
            **audit["restrictions"],
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
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_zip,
                args.model_dir,
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
