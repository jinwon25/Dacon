"""Build the isolated v244 fallback-expansion research candidate."""

from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
INCUMBENT = ROOT.parent / "JY_fallback_XGB_active50_w030"
SOURCE = INCUMBENT / "parent" / "submit_jy_runners_high_li_bridge027_public1172.zip"
ASSET = INCUMBENT
RUNTIME = INCUMBENT / "fallback_xgb_frozen_runtime.py"
OUT = ROOT / "rebuilt" / "submit_jy_xgb_mechanism_w045_w015.zip"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="jy_xgb_mechanism_") as temp_dir:
        stage = Path(temp_dir)
        with zipfile.ZipFile(SOURCE) as archive:
            archive.extractall(stage)

        script = stage / "script.py"
        text = script.read_text(encoding="utf-8")
        text = text.replace(
            "BRIDGE_SCALE = 1.2\n",
            "BRIDGE_SCALE = 1.2\n"
            "XGB_ACTIVE_WEIGHT = 0.30\n"
            "XGB_ACTIVE_THRESHOLD = 0.50\n"
            "XGB_BOUNDARY_LOW = 0.48\n"
            "XGB_BOUNDARY_HIGH = 0.50\n"
            "XGB_BOUNDARY_AGREEMENT = 0.02\n"
            "XGB_BOUNDARY_WEIGHT = 0.45\n"
            "XGB_NONPRESSURE_SAME_WEIGHT = 0.15\n"
            "XGB_NONPRESSURE_OPPOSITE_THRESHOLD = 0.52\n"
            "XGB_NONPRESSURE_OPPOSITE_WEIGHT = 0.15\n",
        )
        marker = "\ndef _window_adjustment(\n"
        helper = '''
def _predict_fallback_xgb(frame: pd.DataFrame) -> np.ndarray:
    module = _load_module("jy_fallback_xgb", MODEL_DIR / "fallback_xgb" / "runtime.py")
    return np.asarray(module.predict(frame, MODEL_DIR / "fallback_xgb"), dtype=np.float64)

'''
        if marker not in text:
            raise RuntimeError("champion helper marker missing")
        text = text.replace(marker, helper + marker)

        old = '''    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * effective_bridge[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    return parent, h1, c3, active, output
'''
        new = '''    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * effective_bridge[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    jy_probability = output.copy()
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
        if old not in text:
            raise RuntimeError("champion formula marker missing")
        script.write_text(text.replace(old, new), encoding="utf-8")

        target = stage / "model" / "fallback_xgb"
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "fallback_xgb.json",
            "feature_columns.json",
            "fallback_lookups.joblib",
            "metadata.json",
        ):
            shutil.copy2(ASSET / name, target / name)
        shutil.copy2(RUNTIME, target / "runtime.py")
        requirements = stage / "requirements.txt"
        requirements.write_text(
            requirements.read_text(encoding="utf-8").rstrip()
            + "\nxgboost==3.2.0\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in stage.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
    print(OUT)


if __name__ == "__main__":
    main()
