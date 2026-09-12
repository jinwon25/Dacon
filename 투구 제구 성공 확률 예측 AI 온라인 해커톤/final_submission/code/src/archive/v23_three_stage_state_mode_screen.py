"""Three-stage state/mode screen above v22.

Stage 1 (2022) ranks only signal identities.  Stage 2 (honest late-2023 v22
OOF) selects one identity and its blend eta.  Stage 3 opens full-2024 once as
the final audit.  Each underlying raw model is trained strictly before its
audit year.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss
from src.core.axes import _load_axis


ETAS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.35, 0.50)
FORBIDDEN = ("oracle", "mode_label", "raw_by_mode")


def _mode_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        names = [str(value) for value in saved["names"].tolist()]
        return {
            f"mode::{name}": saved["raw"][:, index].astype(np.float64)
            for index, name in enumerate(names)
            if not any(token in name.lower() for token in FORBIDDEN)
        }


def _state_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as saved:
        return {
            f"state::{name}": saved[name].astype(np.float64)
            for name in saved.files
            if name.startswith(("global_", "domain_"))
        }


def _bank(project: Path, year: int) -> dict[str, np.ndarray]:
    mode_dir = (
        "latent_failure_mode_state_20260817_02"
        if year == 2022
        else "latent_failure_mode_state_20260816_01"
    )
    output = _mode_bank(
        project / "artifacts" / mode_dir / f"latent_failure_mode_o{year}.npz"
    )
    output.update(
        _state_bank(
            project
            / "artifacts"
            / "multi_year_state_variants_20260816_01"
            / f"state_variants_o{year}.npz"
        )
    )
    with np.load(
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / f"selected_state_o{year}.npz",
        allow_pickle=True,
    ) as saved:
        output["state::selected"] = saved["raw"].astype(np.float64)
    return output


def _diagnostics(
    target: np.ndarray,
    parent: np.ndarray,
    raw: np.ndarray,
    month: np.ndarray,
    domain: np.ndarray,
    eta: float,
) -> dict[str, float]:
    candidate = np.clip(parent + eta * (raw - parent), 0.001, 0.999)
    month_gains = []
    for value in sorted(np.unique(month)):
        selected = month == value
        month_gains.append(
            _bss(target[selected], candidate[selected])
            - _bss(target[selected], parent[selected])
        )
    domain_gains = []
    for value in ("R_CORE", "R_ANCHOR", "F"):
        selected = domain == value
        domain_gains.append(
            _bss(target[selected], candidate[selected])
            - _bss(target[selected], parent[selected])
        )
    gain = _bss(target, candidate) - _bss(target, parent)
    return {
        "gain": gain,
        "positive_month_fraction": float(np.mean(np.asarray(month_gains) > 0.0)),
        "worst_month_gain": float(min(month_gains)),
        "minimum_domain_gain": float(min(domain_gains)),
        "selection_score": float(min(gain, min(month_gains), min(domain_gains))),
    }


def _stage_rows(
    bank: dict[str, np.ndarray],
    names: list[str],
    target: np.ndarray,
    parent: np.ndarray,
    month: np.ndarray,
    domain: np.ndarray,
    mask: np.ndarray | None = None,
) -> list[dict[str, object]]:
    if mask is None:
        mask = np.ones(len(target), dtype=bool)
    rows = []
    for name in names:
        raw = bank[name]
        if len(raw) != len(mask):
            raise ValueError(f"row mismatch for {name}: {len(raw)} != {len(mask)}")
        for eta in ETAS:
            diagnostics = _diagnostics(
                target,
                parent,
                raw[mask],
                month,
                domain,
                eta,
            )
            rows.append({"signal": name, "eta": eta, **diagnostics})
    return rows


def run(project: Path, output_dir: Path, prefilter: int = 12) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    banks = {year: _bank(project, year) for year in (2022, 2023, 2024)}
    names = sorted(set.intersection(*(set(bank) for bank in banks.values())))

    with np.load(
        project
        / "artifacts"
        / "multi_year_state_selected_20260816_01"
        / "selected_state_o2022.npz",
        allow_pickle=True,
    ) as saved:
        target_2022 = saved["target"].astype(np.float64)
        parent_2022 = saved["incumbent"].astype(np.float64)
        month_2022 = saved["game_month"].astype(np.int16)
        domain_2022 = saved["domain3"].astype(str)
    stage1 = pd.DataFrame(
        _stage_rows(
            banks[2022],
            names,
            target_2022,
            parent_2022,
            month_2022,
            domain_2022,
        )
    )
    stage1.to_csv(output_dir / "stage1_2022_metrics.csv", index=False)
    ranked_signal = (
        stage1.sort_values(["selection_score", "gain"], ascending=False)
        .drop_duplicates("signal")
        .head(prefilter)["signal"]
        .tolist()
    )

    raw_train = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    frame_2023 = _load_axis(project, "y2023_early_to_late", raw_train)
    frame_2024 = _load_axis(project, "y2023_to_y2024", raw_train)
    del raw_train
    full_month_2023 = np.asarray(
        np.load(
            project
            / "artifacts"
            / "multi_year_state_selected_20260816_01"
            / "selected_state_o2023.npz",
            allow_pickle=True,
        )["game_month"]
    )
    late_mask = full_month_2023 >= 8
    if int(late_mask.sum()) != len(frame_2023):
        raise ValueError("late-2023 mask mismatch")

    stage2 = pd.DataFrame(
        _stage_rows(
            banks[2023],
            ranked_signal,
            frame_2023["target"].to_numpy(np.float64),
            frame_2023["v22"].to_numpy(np.float64),
            frame_2023["game_month"].to_numpy(np.int16),
            frame_2023["domain3"].astype(str).to_numpy(),
            late_mask,
        )
    ).sort_values(["selection_score", "gain"], ascending=False)
    stage2.to_csv(output_dir / "stage2_2023_metrics.csv", index=False)
    selected = stage2.iloc[0]
    name = str(selected["signal"])
    eta = float(selected["eta"])

    outer = _diagnostics(
        frame_2024["target"].to_numpy(np.float64),
        frame_2024["v22"].to_numpy(np.float64),
        banks[2024][name],
        frame_2024["game_month"].to_numpy(np.int16),
        frame_2024["domain3"].astype(str).to_numpy(),
        eta,
    )
    gate = {
        "stage1_signal_score_positive": bool(
            stage1.loc[stage1["signal"].eq(name), "selection_score"].max() > 0.0
        ),
        "stage2_selection_score_positive": bool(selected["selection_score"] > 0.0),
        "outer_gain_at_least_5": bool(outer["gain"] >= 5.0),
        "outer_positive_month_fraction_at_least_075": bool(
            outer["positive_month_fraction"] >= 0.75
        ),
        "outer_minimum_domain_positive": bool(outer["minimum_domain_gain"] > 0.0),
        "outer_worst_month_above_minus_10": bool(outer["worst_month_gain"] > -10.0),
    }
    summary = {
        "protocol": "V23_THREE_STAGE_STATE_MODE_V1",
        "signal_count": len(names),
        "stage1_prefilter": prefilter,
        "stage1_ranked_signals": ranked_signal,
        "selected": {key: selected[key] for key in stage2.columns},
        "outer_2024": outer,
        "gate": gate,
        "eligible_for_packaging": bool(all(gate.values())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v23_three_stage_state_mode_20260817_01"),
    )
    parser.add_argument("--prefilter", type=int, default=12)
    args = parser.parse_args()
    run(args.project, args.output_dir, args.prefilter)


if __name__ == "__main__":
    main()
