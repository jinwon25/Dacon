"""Audit v320 F formula and row independence against the preserved v290 ZIP."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
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


def predict_partitioned(module, frame: pd.DataFrame) -> np.ndarray:
    split = max(1, len(frame) // 2)
    return np.concatenate([
        module.predict_dataframe(frame.iloc[:split].reset_index(drop=True)),
        module.predict_dataframe(frame.iloc[split:].reset_index(drop=True)),
    ])


def run(v290_zip: Path, v320_zip: Path, train_csv: Path, output_json: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="v320_audit_") as temporary:
        root = Path(temporary)
        old = root / "v290"
        new = root / "v320"
        with zipfile.ZipFile(v290_zip) as archive:
            archive.extractall(old)
        with zipfile.ZipFile(v320_zip) as archive:
            archive.extractall(new)
        v290 = load_module("audit_v290_runtime", old / "script.py")
        v320 = load_module("audit_v320_runtime", new / "script.py")
        frame = pd.read_csv(train_csv, low_memory=False).drop(
            columns=["control_success"], errors="ignore"
        )
        is_f = frame["game_type"].astype(str).eq("F").to_numpy()
        old_prediction = v290.predict_dataframe(frame.copy())
        new_prediction = v320.predict_dataframe(frame.copy())
        expert = v320._predict_recent_futures(frame.copy())
        lowrank = v320._predict_futures_lowrank(frame.copy())
        pre_direct = (old_prediction[is_f] - 0.10 * expert[is_f]) / 0.90
        expected_f = np.clip(
            pre_direct
            + 0.20 * (expert[is_f] - pre_direct)
            + 0.50 * lowrank[is_f],
            0.001,
            0.999,
        )
        shuffle_order = frame.sample(frac=1.0, random_state=320).index.to_numpy()
        shuffled = frame.iloc[shuffle_order].reset_index(drop=True)
        shuffled_prediction = v320.predict_dataframe(shuffled)
        restored = np.empty_like(shuffled_prediction)
        restored[shuffle_order] = shuffled_prediction
        partitioned = predict_partitioned(v320, frame)
        f_frame = frame.loc[is_f].reset_index(drop=True)
        f_full = v320.predict_dataframe(f_frame.copy())
        f_singletons = np.asarray([
            v320.predict_dataframe(f_frame.iloc[[index]].reset_index(drop=True))[0]
            for index in range(len(f_frame))
        ])
        result = {
            "rows": int(len(frame)),
            "f_rows": int(is_f.sum()),
            "r_route_parity_max_abs": float(
                np.max(np.abs(new_prediction[~is_f] - old_prediction[~is_f]))
            ),
            "f_formula_max_abs": float(
                np.max(np.abs(new_prediction[is_f] - expected_f))
            ),
            "shuffle_max_abs": float(np.max(np.abs(new_prediction - restored))),
            "partition_max_abs": float(np.max(np.abs(new_prediction - partitioned))),
            "f_singleton_max_abs": float(np.max(np.abs(f_full - f_singletons))),
            "finite": bool(np.isfinite(new_prediction).all()),
            "in_range": bool(np.all((new_prediction >= 0.0) & (new_prediction <= 1.0))),
        }
        result["status"] = "pass" if (
            result["r_route_parity_max_abs"] <= 1e-15
            and result["f_formula_max_abs"] <= 1e-15
            and result["shuffle_max_abs"] <= 1e-15
            and result["partition_max_abs"] <= 1e-15
            and result["f_singleton_max_abs"] <= 1e-15
            and result["finite"] and result["in_range"]
        ) else "fail"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v290-zip", type=Path, required=True)
    parser.add_argument("--v320-zip", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.v290_zip, args.v320_zip, args.train_csv, args.output_json), indent=2))


if __name__ == "__main__":
    main()
