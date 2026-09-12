"""Screen frozen exact-ASOF model diversity above the submitted v25 parent.

The candidate family is selected only on late-2023 rows.  The selected
domain-specific edits are then frozen before full-2024 and late-2024 audits.
All candidate predictions are season-forward and were produced without audit
labels.  The script is research-only and never reads evaluation data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.v25_recipe import ETA, MODEL_NAME
from src.core.overlay import _bss
from src.core.axes import _joint_domain
from src.champion.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.core.axes import _derived, _load_axis
from src.core.axes import _early_to_late_2024


DOMAINS = ("R_CORE", "R_ANCHOR", "F")
WEIGHTS = (0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20)
MODES = ("toward", "delta_base", "delta_exact_lgb")


def compose(
    parent: np.ndarray,
    signal: np.ndarray,
    old_base: np.ndarray,
    exact_lgb: np.ndarray,
    mask: np.ndarray,
    weight: float,
    mode: str,
) -> np.ndarray:
    """Apply one frozen probability direction to selected rows only."""
    output = np.asarray(parent, dtype=np.float64).copy()
    if mode == "toward":
        direction = np.asarray(signal, dtype=np.float64) - output
    elif mode == "delta_base":
        direction = np.asarray(signal, dtype=np.float64) - np.asarray(
            old_base, dtype=np.float64
        )
    elif mode == "delta_exact_lgb":
        direction = np.asarray(signal, dtype=np.float64) - np.asarray(
            exact_lgb, dtype=np.float64
        )
    else:
        raise ValueError(f"unknown composition mode: {mode}")
    output[mask] = np.clip(output[mask] + weight * direction[mask], 0.001, 0.999)
    return output


def diagnostics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    candidate: np.ndarray,
    apply_mask: np.ndarray,
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)

    def gain(mask: np.ndarray) -> float:
        return float(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )

    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "applied_rows": int(np.sum(mask & apply_mask)),
                "gain": gain(mask),
            }
        )
    active = [row for row in months if row["applied_rows"] > 0]
    domains = {}
    for domain in DOMAINS:
        mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        domains[domain] = gain(mask)
    return {
        "gain": gain(np.ones(len(frame), dtype=bool)),
        "applied_rows": int(apply_mask.sum()),
        "applied_domain_gain": gain(apply_mask),
        "positive_active_month_fraction": float(
            np.mean([row["gain"] > 0.0 for row in active])
        ),
        "worst_active_month_gain": float(min(row["gain"] for row in active)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
        "domain_gains": domains,
    }


def _v25_axes(project: Path, raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    model_spec = next(spec for spec in SPECS if spec.name == MODEL_NAME)
    year23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    derived23 = _derived(year23, _joint_domain(year23))

    selection = _load_axis(project, "y2023_early_to_late", raw)
    selection_fit = derived23.loc[derived23["game_month"].le(7)].reset_index(drop=True)
    selection_direct = _fit_predict(selection_fit, selection, model_spec)

    outer = _load_axis(project, "y2023_to_y2024", raw)
    outer_direct = _fit_predict(derived23, outer, model_spec)

    replication_fit, replication = _early_to_late_2024(project, raw)
    replication_direct = _fit_predict(replication_fit, replication, model_spec)

    axes = {
        "selection_late_2023": (selection, selection_direct),
        "outer_full_2024": (outer, outer_direct),
        "replication_late_2024": (replication, replication_direct),
    }
    output: dict[str, pd.DataFrame] = {}
    for name, (frame, direct) in axes.items():
        frame = frame.copy()
        parent = frame["v22"].to_numpy(np.float64).copy()
        anchor = frame["domain3"].astype(str).eq("R_ANCHOR").to_numpy()
        parent[anchor] = np.clip(
            parent[anchor] + ETA * (direct[anchor] - parent[anchor]), 0.001, 0.999
        )
        frame["v25"] = parent
        output[name] = frame
    return output


def _signal_bank(
    project: Path, raw: pd.DataFrame, frame: pd.DataFrame, year: int
) -> dict[str, np.ndarray]:
    recent_path = (
        project
        / "artifacts"
        / "recent_shared_exact_asof_20260815_02"
        / f"recent_shared_o{year}.npz"
    )
    diversity_path = (
        project
        / "artifacts"
        / "v14_exact_model_screen_20260815_01"
        / f"exact_model_screen_o{year}.npz"
    )
    with np.load(recent_path) as saved:
        recent = {key: saved[key].astype(np.float64) for key in saved.files}
    with np.load(diversity_path) as saved:
        diversity = {key: saved[key].astype(np.float64) for key in saved.files}
    season = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    if len(frame) == len(season):
        index = np.arange(len(season))
    else:
        index = np.flatnonzero(season["game_month"].ge(8).to_numpy())
    expected = frame["target"].to_numpy(np.float64)
    if not np.array_equal(recent["target"][index], expected):
        raise ValueError(f"recent exact target order mismatch for {year}")
    if not np.array_equal(diversity["target"][index], expected):
        raise ValueError(f"diversity target order mismatch for {year}")
    bank = {
        "old_base": recent["base"][index],
        "exact_lgb": recent["exact_lgb"][index],
        "exact_ridge": recent["exact_ridge"][index],
        "shared": recent["prediction"][index],
        "trend_lgb": recent["trend_lgb"][index],
    }
    for key, value in diversity.items():
        if key not in {"train_index", "target", "base", "exact_lgb", "exact_ridge", "trend_lgb"}:
            bank[key] = value[index]
    return bank


def _screen_selection(
    frame: pd.DataFrame, bank: dict[str, np.ndarray]
) -> pd.DataFrame:
    parent = frame["v25"].to_numpy(np.float64)
    rows = []
    for domain in DOMAINS:
        mask = frame["domain3"].astype(str).eq(domain).to_numpy()
        for signal_name, signal in bank.items():
            if signal_name in {"old_base"}:
                continue
            for mode in MODES:
                if signal_name == "exact_lgb" and mode == "delta_exact_lgb":
                    continue
                for weight in WEIGHTS:
                    candidate = compose(
                        parent,
                        signal,
                        bank["old_base"],
                        bank["exact_lgb"],
                        mask,
                        weight,
                        mode,
                    )
                    result = diagnostics(frame, parent, candidate, mask)
                    rows.append(
                        {
                            "domain": domain,
                            "signal": signal_name,
                            "mode": mode,
                            "weight": weight,
                            **{key: value for key, value in result.items() if key not in {"months", "domain_gains"}},
                        }
                    )
    metrics = pd.DataFrame(rows)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
        & metrics["positive_active_month_fraction"].eq(1.0)
        & metrics["worst_active_month_gain"].gt(0.0)
    )
    metrics["selection_score"] = metrics[
        ["gain", "worst_active_month_gain"]
    ].min(axis=1)
    return metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)


def _apply_recipe(
    frame: pd.DataFrame,
    bank: dict[str, np.ndarray],
    recipe: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    parent = frame["v25"].to_numpy(np.float64)
    output = parent.copy()
    applied = np.zeros(len(frame), dtype=bool)
    for item in recipe:
        mask = frame["domain3"].astype(str).eq(str(item["domain"])).to_numpy()
        output = compose(
            output,
            bank[str(item["signal"])],
            bank["old_base"],
            bank["exact_lgb"],
            mask,
            float(item["weight"]),
            str(item["mode"]),
        )
        applied |= mask
    return output, applied


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _v25_axes(project, raw)
    banks = {
        "selection_late_2023": _signal_bank(
            project, raw, axes["selection_late_2023"], 2023
        ),
        "outer_full_2024": _signal_bank(project, raw, axes["outer_full_2024"], 2024),
        "replication_late_2024": _signal_bank(
            project, raw, axes["replication_late_2024"], 2024
        ),
    }
    selection_metrics = _screen_selection(
        axes["selection_late_2023"], banks["selection_late_2023"]
    )
    selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)

    selected = []
    for domain in DOMAINS:
        eligible = selection_metrics.loc[
            selection_metrics["domain"].eq(domain)
            & selection_metrics["passes_selection_gate"]
        ]
        if len(eligible):
            row = eligible.iloc[0]
            selected.append(
                {
                    "domain": domain,
                    "signal": str(row["signal"]),
                    "mode": str(row["mode"]),
                    "weight": float(row["weight"]),
                    "selection_gain": float(row["gain"]),
                    "selection_worst_month_gain": float(row["worst_active_month_gain"]),
                }
            )

    audits = {}
    for axis_name, frame in axes.items():
        parent = frame["v25"].to_numpy(np.float64)
        candidate, apply_mask = _apply_recipe(frame, banks[axis_name], selected)
        audits[axis_name] = diagnostics(frame, parent, candidate, apply_mask)
        np.savez_compressed(
            output_dir / f"{axis_name}.npz",
            target=frame["target"].to_numpy(np.float64),
            v25=parent,
            candidate=candidate,
            domain3=frame["domain3"].astype(str).to_numpy(),
            game_month=frame["game_month"].to_numpy(np.int16),
        )

    gates = {
        "all_domains_selected": len(selected) == len(DOMAINS),
        "selection_gain_positive": audits["selection_late_2023"]["gain"] > 0.0,
        "selection_all_active_months_positive": audits["selection_late_2023"][
            "positive_active_month_fraction"
        ] == 1.0,
        "outer_gain_at_least_5": audits["outer_full_2024"]["gain"] >= 5.0,
        "outer_positive_active_month_fraction_at_least_075": audits[
            "outer_full_2024"
        ]["positive_active_month_fraction"] >= 0.75,
        "outer_worst_month_above_minus_10": audits["outer_full_2024"][
            "worst_active_month_gain"
        ] > -10.0,
        "outer_all_domain_gains_positive": min(
            audits["outer_full_2024"]["domain_gains"].values()
        ) > 0.0,
        "replication_gain_positive": audits["replication_late_2024"]["gain"] > 0.0,
        "replication_all_active_months_positive": audits[
            "replication_late_2024"
        ]["positive_active_month_fraction"] == 1.0,
    }
    summary = {
        "protocol": "V26_EXACT_DIVERSITY_ABOVE_V25_V1",
        "selection_policy": "one candidate per domain chosen only on late-2023 v25 analogue",
        "candidate_family_size": int(len(selection_metrics)),
        "selected": selected,
        "audits": audits,
        "gates": {key: bool(value) for key, value in gates.items()},
        "eligible_for_packaging": bool(all(gates.values())),
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
        default=Path("artifacts/v26_exact_diversity_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
