"""Audit the competition's one-row-versus-batch inference invariant."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_BATCH_TOKENS = (
    "groupby(",
    "value_counts(",
    ".rolling(",
    ".expanding(",
    ".rank(",
)
FORBIDDEN_NETWORK_TOKENS = (
    "requests.",
    "urllib.request",
    "http://",
    "https://",
    "socket.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _predict(module, frame: pd.DataFrame) -> np.ndarray:
    # Reading a one-row CSV creates a fresh zero-based index, so reset here to
    # exercise the exact official file-level invariant rather than pandas'
    # caller-owned index labels.
    return np.asarray(
        module.predict_dataframe(frame.reset_index(drop=True)), dtype=np.float64
    )


def audit(project: Path, zip_path: Path) -> dict[str, object]:
    project = project.resolve()
    zip_path = zip_path.resolve()
    frame = pd.read_csv(
        project / "data" / "test.csv", encoding="utf-8-sig", low_memory=False
    )
    with tempfile.TemporaryDirectory(prefix="aimers9_independence_") as temp_name:
        extracted = Path(temp_name) / "package"
        extracted.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted)
        module_spec = importlib.util.spec_from_file_location(
            f"audit_{zip_path.stem}", extracted / "script.py"
        )
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("could not import archived script")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        full = _predict(module, frame)
        single = np.asarray(
            [_predict(module, frame.iloc[[index]])[0] for index in range(len(frame))]
        )
        permutation = np.arange(len(frame))[::-1]
        shuffled_raw = _predict(module, frame.iloc[permutation])
        shuffled = np.empty_like(shuffled_raw)
        shuffled[permutation] = shuffled_raw
        chunked = np.concatenate(
            [
                _predict(module, frame.iloc[start : start + 2])
                for start in range(0, len(frame), 2)
            ]
        )

        changed = frame.copy()
        anchor = 0
        other = np.arange(len(changed)) != anchor
        for column in (
            "game_month",
            "inning",
            "balls_before",
            "strikes_before",
            "outs_before",
            "run_total_before",
            "score_diff_pitcher_team",
            "num_runners_on",
            "asof_pitcher_n",
            "asof_batter_n",
        ):
            if column in changed:
                changed.loc[other, column] = changed.loc[other, column].iloc[::-1].to_numpy()
        for column in (
            "pitcher_id",
            "batter_id",
            "pitcher_team_id",
            "batter_team_id",
            "game_type",
            "top_bottom",
            "base_state",
        ):
            if column in changed:
                changed.loc[other, column] = changed.loc[other, column].iloc[::-1].to_numpy()
        changed_anchor = _predict(module, changed)[anchor]

        repeated = pd.concat([frame.iloc[[anchor]]] * 5, ignore_index=True)
        repeated_prediction = _predict(module, repeated)
        script_text = (extracted / "script.py").read_text(encoding="utf-8").lower()
        batch_tokens = [token for token in FORBIDDEN_BATCH_TOKENS if token in script_text]
        network_tokens = [token for token in FORBIDDEN_NETWORK_TOKENS if token in script_text]
        spec_path = extracted / "model" / "corrected_state_residual_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))

    deviations = {
        "single_vs_full": float(np.max(np.abs(single - full))),
        "shuffle_vs_full": float(np.max(np.abs(shuffled - full))),
        "chunk_vs_full": float(np.max(np.abs(chunked - full))),
        "changed_other_rows_anchor": float(abs(changed_anchor - full[anchor])),
        "repeated_anchor": float(np.max(np.abs(repeated_prediction - full[anchor]))),
    }
    tolerance = 1e-12
    result = {
        "zip": zip_path.name,
        "zip_sha256": _sha256(zip_path),
        "n_official_sample_rows": int(len(frame)),
        "max_absolute_deviation": deviations,
        "numeric_tolerance": tolerance,
        "all_invariance_checks_within_tolerance": all(
            value <= tolerance for value in deviations.values()
        ),
        "forbidden_batch_tokens": batch_tokens,
        "forbidden_network_tokens": network_tokens,
        "residual_train_sha256_matches": spec["train_sha256"]
        == _sha256(project / "data" / "train.csv"),
        "test_file_packaged": any(
            name.startswith("data/")
            for name in zipfile.ZipFile(zip_path).namelist()
        ),
    }
    if not result["all_invariance_checks_within_tolerance"]:
        raise ValueError(f"row-independence invariant failed: {deviations}")
    if batch_tokens or network_tokens or not result["residual_train_sha256_matches"]:
        raise ValueError(f"static/provenance audit failed: {result}")
    if result["test_file_packaged"]:
        raise ValueError("evaluation data must not be packaged")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    zip_path = args.zip if args.zip.is_absolute() else project / args.zip
    result = audit(project, zip_path)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else project / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
