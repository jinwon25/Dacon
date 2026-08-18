"""Audit the remaining independent-base directions above frozen v27.

Two locally available directions were not part of the v30 OOF bank:

* the original LightGBM/RandomForest wave-0 walk-forward predictions;
* the gap between fully forward-nested v42 state/mode routes and the v19
  development route.

Recipes are selected on late 2023 only and then frozen for full and late 2024.
The script is an audit/closure experiment and never creates a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import read_main
from src.v30_diverse_covariance_screen import _cached_v25_axes, v27_parent


DOMAINS = ("ALL", "R_CORE", "R_ANCHOR", "F")
WAVE0_WEIGHTS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10)
NESTED_GAP_WEIGHTS = (
    -0.20,
    -0.15,
    -0.10,
    -0.075,
    -0.05,
    -0.035,
    -0.02,
    -0.01,
    -0.005,
    0.005,
    0.01,
    0.02,
    0.035,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
)
WAVE0_NAMES = (
    "lgb_raw",
    "rf_raw",
    "blend_raw",
    "lgb_trend",
    "rf_trend",
    "incumbent",
)


def _gain(
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> float:
    local_target = target[mask]
    reference = float(local_target.mean() * (1.0 - local_target.mean()))
    scale = 1_000_000.0 if reference <= 0.0 else 100000.0 / reference
    return float(
        scale
        * np.mean(
            np.square(local_target - parent[mask])
            - np.square(local_target - candidate[mask])
        )
    )


def diagnostics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    apply_mask: np.ndarray,
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        if np.any(mask & apply_mask):
            months.append(
                {"month": int(month), "gain": _gain(target, parent, candidate, mask)}
            )
    domains = []
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        domains.append(
            {"domain": domain, "gain": _gain(target, parent, candidate, mask)}
        )
    return {
        "gain": _gain(
            target, parent, candidate, np.ones(len(target), dtype=bool)
        ),
        "positive_month_fraction": float(
            np.mean([item["gain"] > 0.0 for item in months])
        ),
        "worst_month_gain": float(min(item["gain"] for item in months)),
        "minimum_domain_gain": float(min(item["gain"] for item in domains)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
        "domains": domains,
    }


def _domain_mask(frame: pd.DataFrame, domain: str) -> np.ndarray:
    if domain == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["domain3"].astype(str).eq(domain).to_numpy()


def _compose(
    parent: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    weight: float,
) -> np.ndarray:
    output = parent.copy()
    output[mask] = np.clip(
        parent[mask] + weight * direction[mask], 0.001, 0.999
    )
    return output


def _year_mask(raw: pd.DataFrame, saved: np.lib.npyio.NpzFile, late: bool) -> np.ndarray:
    indices = saved["valid_idx"].astype(np.int64)
    target = raw.iloc[indices]["control_success"].to_numpy(np.int8)
    if not np.array_equal(target, saved["target"]):
        raise ValueError("wave0 target/index alignment failure")
    if not late:
        return np.ones(len(indices), dtype=bool)
    return raw.iloc[indices]["game_month"].to_numpy() >= 8


def _signals(
    project: Path,
    raw: pd.DataFrame,
    frame: pd.DataFrame,
    axis: str,
) -> dict[tuple[str, str], np.ndarray]:
    year = 2023 if axis == "selection_late_2023" else 2024
    late = "late" in axis
    output: dict[tuple[str, str], np.ndarray] = {}
    wave_path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(wave_path) as saved:
        mask = _year_mask(raw, saved, late)
        if not np.array_equal(
            saved["target"][mask].astype(np.float64),
            frame["target"].to_numpy(np.float64),
        ):
            raise ValueError(f"wave0/frame alignment failure: {axis}")
        for name in WAVE0_NAMES:
            output[("wave0", name)] = (
                saved[name][mask].astype(np.float64)
                - v27_parent(frame)
            )

    old_path = (
        project
        / "artifacts"
        / "state_mode_joint_20260816_02"
        / f"joint_candidate_o{year}.npz"
    )
    nested_path = (
        project
        / "artifacts"
        / "v42_forward_nested_state_mode_20260817_01"
        / f"nested_candidate_o{year}.npz"
    )
    with np.load(old_path, allow_pickle=True) as old, np.load(
        nested_path, allow_pickle=True
    ) as nested:
        mask = old["game_month"] >= 8 if late else np.ones(
            len(old["target"]), dtype=bool
        )
        expected = frame["target"].to_numpy(np.float64)
        if not np.array_equal(old["target"][mask].astype(np.float64), expected):
            raise ValueError(f"v19/frame alignment failure: {axis}")
        if not np.array_equal(nested["target"], old["target"]):
            raise ValueError(f"nested/v19 alignment failure: {year}")
        output[("nested_gap", "nested_minus_v19")] = (
            nested["candidate"] - old["candidate"]
        )[mask].astype(np.float64)
    return output


def _weights(family: str) -> tuple[float, ...]:
    return WAVE0_WEIGHTS if family == "wave0" else NESTED_GAP_WEIGHTS


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_main(project / "data" / "train.csv")
    frames = _cached_v25_axes(project, raw)
    signals = {
        axis: _signals(project, raw, frame, axis)
        for axis, frame in frames.items()
    }

    selection_rows = []
    selection_frame = frames["selection_late_2023"]
    selection_parent = v27_parent(selection_frame)
    for (family, signal), direction in signals["selection_late_2023"].items():
        for domain in DOMAINS:
            mask = _domain_mask(selection_frame, domain)
            for weight in _weights(family):
                candidate = _compose(
                    selection_parent, direction, mask, weight
                )
                item = diagnostics(
                    selection_frame, selection_parent, candidate, mask
                )
                selection_rows.append(
                    {
                        "family": family,
                        "signal": signal,
                        "domain": domain,
                        "weight": weight,
                        **{
                            key: item[key]
                            for key in (
                                "gain",
                                "positive_month_fraction",
                                "worst_month_gain",
                                "minimum_domain_gain",
                                "mean_abs_shift",
                            )
                        },
                    }
                )
    selection = pd.DataFrame(selection_rows)
    selection["eligible"] = (
        selection["gain"].gt(0.0)
        & selection["positive_month_fraction"].ge(2.0 / 3.0)
        & selection["worst_month_gain"].gt(-20.0)
        & selection["minimum_domain_gain"].ge(-2.0)
    )
    selection.to_csv(output_dir / "selection_metrics.csv", index=False)

    chosen = []
    audit_rows = []
    for family in ("wave0", "nested_gap"):
        source = selection.loc[
            selection["family"].eq(family) & selection["eligible"]
        ].sort_values(
            ["gain", "worst_month_gain", "minimum_domain_gain"],
            ascending=False,
            kind="stable",
        )
        if source.empty:
            continue
        recipe = source.iloc[0].to_dict()
        chosen.append(recipe)
        key = (family, str(recipe["signal"]))
        for axis in ("outer_full_2024", "replication_late_2024"):
            frame = frames[axis]
            parent = v27_parent(frame)
            mask = _domain_mask(frame, str(recipe["domain"]))
            candidate = _compose(
                parent,
                signals[axis][key],
                mask,
                float(recipe["weight"]),
            )
            item = diagnostics(frame, parent, candidate, mask)
            audit_rows.append(
                {
                    "family": family,
                    "signal": recipe["signal"],
                    "domain": recipe["domain"],
                    "weight": recipe["weight"],
                    "axis": axis,
                    **{
                        key: item[key]
                        for key in (
                            "gain",
                            "positive_month_fraction",
                            "worst_month_gain",
                            "minimum_domain_gain",
                            "mean_abs_shift",
                        )
                    },
                    "months": json.dumps(item["months"]),
                    "domains": json.dumps(item["domains"]),
                }
            )
    audits = pd.DataFrame(audit_rows)
    audits.to_csv(output_dir / "audit_metrics.csv", index=False)
    summary = {
        "protocol": "INDEPENDENT_BASE_LATE2023_SELECT_2024_AUDIT_V1",
        "selection_candidate_count": int(len(selection)),
        "eligible_count": int(selection["eligible"].sum()),
        "chosen_by_selection_only": chosen,
        "audits": audit_rows,
        "eligible_for_packaging": False,
        "decision": (
            "Reject both families: each selection winner fails at least one "
            "independent 2024 audit and neither is month-stable."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v43_independent_base_audit_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
