from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_ZIP = ROOT / "submissions" / "releases" / "v167" / "submit_v167.zip"
PREVIOUS_ZIP = ROOT / "artifacts" / "jy_runners_gate_20260828_01" / "submit_jy_runners_gate.zip"
CANDIDATE_ZIP = ROOT / "artifacts" / "jy_next_variants_20260828_01" / "submit_jy_runners_high_li_bridge027.zip"
OUT_DIR = ROOT / "artifacts" / "jy_next_variants_20260828_01"
OFFICIAL_TEST = ROOT.parent / "data" / "test.csv"


SITE_CUSTOMIZE = """
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_probe_frame(rows: int = 3600) -> pd.DataFrame:
    seed = pd.read_csv(OFFICIAL_TEST, encoding="utf-8-sig")
    frame = pd.concat([seed] * ((rows // len(seed)) + 1), ignore_index=True).iloc[:rows].copy()
    frame["row_id"] = [f"JY_NEXT_{i:06d}" for i in range(len(frame))]
    for i in range(len(frame)):
        mode = i % 8
        frame.at[i, "game_type"] = "R"
        frame.at[i, "pitcher_team_id"] = 16
        frame.at[i, "batter_team_id"] = 21
        frame.at[i, "num_runners_on"] = 0
        frame.at[i, "runner_on_1b"] = 0
        frame.at[i, "runner_on_2b"] = 0
        frame.at[i, "runner_on_3b"] = 0
        frame.at[i, "base_state"] = "___"
        frame.at[i, "li"] = 0.7

        if mode in (0, 1, 6):
            frame.at[i, "num_runners_on"] = 1
            frame.at[i, "runner_on_1b"] = 1
            frame.at[i, "base_state"] = "1__"
        if mode in (1, 2, 7):
            frame.at[i, "li"] = 1.8
        if mode == 3:
            frame.at[i, "game_type"] = "F"
            frame.at[i, "li"] = 1.9
        if mode == 4:
            frame.at[i, "pitcher_team_id"] = 13
            frame.at[i, "num_runners_on"] = 1
            frame.at[i, "runner_on_1b"] = 1
            frame.at[i, "base_state"] = "1__"
        if mode == 5:
            frame.at[i, "batter_team_id"] = 13
            frame.at[i, "li"] = 2.0
    return frame


def run_zip(package_zip: Path, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory(prefix="jy-next-audit-") as tmp:
        work = Path(tmp)
        with zipfile.ZipFile(package_zip, "r") as zf:
            zf.extractall(work)
        (work / "sitecustomize.py").write_text(SITE_CUSTOMIZE, encoding="utf-8")
        (work / "data").mkdir(exist_ok=True)
        (work / "output").mkdir(exist_ok=True)
        frame.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"row_id": frame["row_id"], "control_success": 0.0}).to_csv(
            work / "data" / "sample_submission.csv", index=False, encoding="utf-8-sig"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(work) + os.pathsep + env.get("PYTHONPATH", "")
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "script.py"],
            cwd=work,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stdout)
        elapsed = time.perf_counter() - start
        output = pd.read_csv(work / "output" / "submission.csv", encoding="utf-8-sig")
        return output["control_success"].to_numpy(), elapsed


def active_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    r_core = (
        frame["game_type"].astype(str).eq("R").to_numpy()
        & ~(
            frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
            | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
        )
    )
    runners_on = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy() > 0
    high_li = pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy() >= 1.5
    return {
        "previous_active": r_core & runners_on,
        "candidate_active": r_core & (runners_on | high_li),
    }


def stats(delta: np.ndarray) -> dict[str, float]:
    return {
        "min": float(delta.min()),
        "mean": float(delta.mean()),
        "max": float(delta.max()),
        "mean_abs": float(np.abs(delta).mean()),
    }


def main() -> None:
    frame = build_probe_frame()
    base, base_seconds = run_zip(BASE_ZIP, frame)
    previous, previous_seconds = run_zip(PREVIOUS_ZIP, frame)
    candidate, candidate_seconds = run_zip(CANDIDATE_ZIP, frame)
    masks = active_masks(frame)
    protected = ~masks["candidate_active"]
    added = masks["candidate_active"] & ~masks["previous_active"]
    delta_base = candidate - base
    delta_previous = candidate - previous

    audit = {
        "candidate": "jy_runners_high_li_bridge027",
        "candidate_submit": {"path": str(CANDIDATE_ZIP.relative_to(ROOT)), "sha256": sha256(CANDIDATE_ZIP)},
        "previous_submit": {"path": str(PREVIOUS_ZIP.relative_to(ROOT)), "sha256": sha256(PREVIOUS_ZIP)},
        "base_submit": {"path": str(BASE_ZIP.relative_to(ROOT)), "sha256": sha256(BASE_ZIP)},
        "rows": int(len(frame)),
        "candidate_active_fraction": float(masks["candidate_active"].mean()),
        "previous_active_fraction": float(masks["previous_active"].mean()),
        "added_fraction": float(added.mean()),
        "changed_vs_base_fraction": float((np.abs(delta_base) > 1e-12).mean()),
        "changed_vs_previous_fraction": float((np.abs(delta_previous) > 1e-12).mean()),
        "protected_max_abs_delta_vs_base": float(np.abs(delta_base[protected]).max()),
        "delta_vs_base_active": stats(delta_base[masks["candidate_active"]]),
        "delta_vs_previous_added": stats(delta_previous[added]),
        "runtime": {
            "base_seconds": base_seconds,
            "previous_seconds": previous_seconds,
            "candidate_seconds": candidate_seconds,
        },
        "passes_structural_audit": bool(
            np.abs(delta_base[protected]).max() <= 1e-12
            and (np.abs(delta_base[masks["candidate_active"]]) > 1e-12).mean() >= 0.999
            and candidate.min() >= 0.0
            and candidate.max() <= 1.0
        ),
    }
    (OUT_DIR / "audit_jy_runners_high_li_bridge027.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
