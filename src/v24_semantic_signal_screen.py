"""Screen semantic signals discovered from the official-data audit.

Two low-capacity, inference-safe ideas are tested above a reconstructed v22:

* within-level leverage/score shape learned from an earlier labelled window;
* historical pitcher/batter effects split by Regular versus Futures level.

All choices are made on late 2023.  A frozen choice is evaluated once on the
full 2024 season.  No test-row aggregate is used by either signal.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.overlay import _bss


TARGET = "control_success"
DOMAINS = ("R_CORE", "R_ANCHOR", "F")
SHAPE_MODELS = ("li5", "li10", "score", "li5_score")
SHAPE_ALPHAS = (100.0, 500.0, 2000.0)
LEVEL_HALF_LIVES = (0.5, 1.0, 2.0, 4.0)
LEVEL_ALPHAS = (200.0, 1000.0, 5000.0)
ETAS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0)


def _domain(frame: pd.DataFrame) -> np.ndarray:
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13)
        | frame["batter_team_id"].eq(13)
    ).to_numpy()
    return np.where(~regular, "F", np.where(anchor, "R_ANCHOR", "R_CORE"))


def _v22(rows: pd.DataFrame, v21: np.ndarray, domain: np.ndarray) -> np.ndarray:
    parent = np.asarray(v21, dtype=np.float64)
    correction = np.zeros(len(rows), dtype=np.float64)
    parameters = {
        "R_CORE": (0.44, 0.035),
        "R_ANCHOR": (0.48, 0.020),
        "F": (0.52, 0.020),
    }
    for name, (anchor, weight) in parameters.items():
        mask = domain == name
        correction[mask] += weight * (anchor - parent[mask])
    pitcher = pd.to_numeric(
        rows["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    batter = pd.to_numeric(
        rows["asof_batter_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    prior = 0.75 * pitcher + 0.25 * batter
    correction += 0.05 * (prior - parent)
    return np.clip(parent + correction, 0.001, 0.999)


def _edges(value: pd.Series, bins: int) -> np.ndarray:
    numeric = pd.to_numeric(value, errors="coerce").dropna().to_numpy(np.float64)
    edge = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, bins + 1)))
    if len(edge) < 3:
        return np.asarray([-np.inf, np.inf])
    edge[0] = -np.inf
    edge[-1] = np.inf
    return edge


def _shape_keys(
    source: pd.DataFrame,
    audit: pd.DataFrame,
    model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame({"game_type": source["game_type"].astype(str).to_numpy()})
    right = pd.DataFrame({"game_type": audit["game_type"].astype(str).to_numpy()})
    if model.startswith("li"):
        bins = 5 if model.startswith("li5") else 10
        edge = _edges(source["li"], bins)
        left["li_bin"] = np.digitize(
            pd.to_numeric(source["li"], errors="coerce").fillna(0.0), edge[1:-1]
        )
        right["li_bin"] = np.digitize(
            pd.to_numeric(audit["li"], errors="coerce").fillna(0.0), edge[1:-1]
        )
    if model in {"score", "li5_score"}:
        fixed = np.asarray([-np.inf, -4, -2, -1, 0, 1, 2, 4, np.inf])
        left["score_bin"] = np.digitize(
            pd.to_numeric(source["score_diff_pitcher_team"], errors="coerce").fillna(0.0),
            fixed[1:-1],
        )
        right["score_bin"] = np.digitize(
            pd.to_numeric(audit["score_diff_pitcher_team"], errors="coerce").fillna(0.0),
            fixed[1:-1],
        )
    return left, right


def shape_signal(
    source: pd.DataFrame,
    audit: pd.DataFrame,
    model: str,
    alpha: float,
) -> np.ndarray:
    """Return a within-level conditional-rate deviation."""

    left, right = _shape_keys(source, audit, model)
    columns = list(left.columns)
    work = left.copy()
    work[TARGET] = source[TARGET].to_numpy(np.float64)
    prior = work.groupby("game_type", observed=True)[TARGET].mean()
    stats = (
        work.groupby(columns, observed=True)[TARGET]
        .agg(n="size", successes="sum")
        .reset_index()
    )
    stats["level_prior"] = stats["game_type"].map(prior)
    stats["signal"] = (
        stats["successes"] + alpha * stats["level_prior"]
    ) / (stats["n"] + alpha) - stats["level_prior"]
    joined = right.merge(stats[columns + ["signal"]], on=columns, how="left", sort=False)
    return joined["signal"].fillna(0.0).to_numpy(np.float64)


def level_history_signal(
    history: pd.DataFrame,
    audit: pd.DataFrame,
    half_life: float,
    alpha: float,
) -> np.ndarray:
    """Historical player random effects centered within season and level."""

    maximum = int(history["season"].max())
    work = history[
        ["season", "game_type", "pitcher_id", "batter_id", TARGET]
    ].copy()
    level_mean = work.groupby(["season", "game_type"], observed=True)[TARGET].transform("mean")
    work["residual"] = work[TARGET].to_numpy(np.float64) - level_mean.to_numpy(np.float64)
    work["weight"] = np.exp2(
        -(maximum - work["season"].to_numpy(np.float64)) / float(half_life)
    )

    def effect(entity: str) -> np.ndarray:
        group = work.groupby([entity, "game_type"], observed=True, sort=False)
        numerator = (work["weight"] * work["residual"]).groupby(
            [work[entity], work["game_type"]], observed=True
        ).sum()
        denominator = group["weight"].sum()
        table = (numerator / (denominator + alpha)).rename("effect").reset_index()
        return (
            audit[[entity, "game_type"]]
            .merge(table, on=[entity, "game_type"], how="left", sort=False)["effect"]
            .fillna(0.0)
            .to_numpy(np.float64)
        )

    return 0.75 * effect("pitcher_id") + 0.25 * effect("batter_id")


def _diagnostics(
    rows: pd.DataFrame,
    target: np.ndarray,
    parent: np.ndarray,
    candidate: np.ndarray,
    domain: np.ndarray,
) -> dict[str, object]:
    month_rows = []
    for month in sorted(rows["game_month"].unique()):
        mask = rows["game_month"].eq(month).to_numpy()
        month_rows.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    domain_rows = []
    for name in DOMAINS:
        mask = domain == name
        domain_rows.append(
            {
                "domain": name,
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    return {
        "gain": _bss(target, candidate) - _bss(target, parent),
        "positive_month_fraction": float(np.mean([row["gain"] > 0 for row in month_rows])),
        "worst_month_gain": float(min(row["gain"] for row in month_rows)),
        "minimum_domain_gain": float(min(row["gain"] for row in domain_rows)),
        "months": month_rows,
        "domains": domain_rows,
        "mean_shift": float(np.mean(candidate - parent)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
    }


def _load_axis(
    project: Path,
    train: pd.DataFrame,
    axis: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    path = project / "artifacts/champion_oof_20260817_01" / f"{axis}.npz"
    with np.load(path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)
    if axis == "y2023_early_to_late":
        rows = train.loc[train["season"].eq(2023) & train["game_month"].ge(8)].copy()
    elif axis == "y2023_to_y2024":
        rows = train.loc[train["season"].eq(2024)].copy()
    else:
        raise ValueError(axis)
    rows = rows.reset_index(drop=True)
    if not np.array_equal(rows[TARGET].to_numpy(np.float64), target):
        raise ValueError(f"target alignment failed for {axis}")
    if not np.array_equal(_domain(rows), domain):
        raise ValueError(f"domain alignment failed for {axis}")
    return rows, target, _v22(rows, v21, domain), domain


def _selection_score(result: dict[str, object]) -> float:
    return float(
        min(
            result["gain"],
            result["worst_month_gain"],
            result["minimum_domain_gain"],
        )
    )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project / "data/train.csv", low_memory=False)
    selection_rows, selection_target, selection_parent, selection_domain = _load_axis(
        project, train, "y2023_early_to_late"
    )
    source_early = train.loc[
        train["season"].eq(2023) & train["game_month"].le(7)
    ].reset_index(drop=True)
    history_2022 = train.loc[train["season"].le(2022)].reset_index(drop=True)

    candidates: list[dict[str, object]] = []
    for model, alpha in itertools.product(SHAPE_MODELS, SHAPE_ALPHAS):
        signal = shape_signal(source_early, selection_rows, model, alpha)
        for eta in ETAS:
            candidate = np.clip(selection_parent + eta * signal, 0.001, 0.999)
            diagnostic = _diagnostics(
                selection_rows,
                selection_target,
                selection_parent,
                candidate,
                selection_domain,
            )
            candidates.append(
                {
                    "family": "shape",
                    "model": model,
                    "alpha": alpha,
                    "half_life": np.nan,
                    "eta": eta,
                    **{key: diagnostic[key] for key in (
                        "gain", "positive_month_fraction", "worst_month_gain", "minimum_domain_gain", "mean_shift", "mean_abs_shift"
                    )},
                    "selection_score": _selection_score(diagnostic),
                }
            )
    for half_life, alpha in itertools.product(LEVEL_HALF_LIVES, LEVEL_ALPHAS):
        signal = level_history_signal(history_2022, selection_rows, half_life, alpha)
        for eta in ETAS:
            candidate = np.clip(selection_parent + eta * signal, 0.001, 0.999)
            diagnostic = _diagnostics(
                selection_rows,
                selection_target,
                selection_parent,
                candidate,
                selection_domain,
            )
            candidates.append(
                {
                    "family": "level_history",
                    "model": "pitcher75_batter25_level_effect",
                    "alpha": alpha,
                    "half_life": half_life,
                    "eta": eta,
                    **{key: diagnostic[key] for key in (
                        "gain", "positive_month_fraction", "worst_month_gain", "minimum_domain_gain", "mean_shift", "mean_abs_shift"
                    )},
                    "selection_score": _selection_score(diagnostic),
                }
            )

    table = pd.DataFrame(candidates).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    table.to_csv(output_dir / "selection.csv", index=False)
    selected = table.iloc[0].to_dict()
    selection_gate = bool(
        selected["gain"] > 0
        and selected["positive_month_fraction"] >= 1.0
        and selected["minimum_domain_gain"] > 0
    )
    summary: dict[str, object] = {
        "protocol": "V24_SEMANTIC_SIGNAL_NESTED_V1",
        "selection": "late-2023 only above reconstructed v22",
        "selected": selected,
        "selection_gate_passed": selection_gate,
        "outer_audit_run": selection_gate,
        "eligible_for_packaging": False,
    }
    if selection_gate:
        outer_rows, outer_target, outer_parent, outer_domain = _load_axis(
            project, train, "y2023_to_y2024"
        )
        if selected["family"] == "shape":
            source = train.loc[train["season"].eq(2023)].reset_index(drop=True)
            signal = shape_signal(
                source,
                outer_rows,
                str(selected["model"]),
                float(selected["alpha"]),
            )
        else:
            history = train.loc[train["season"].le(2023)].reset_index(drop=True)
            signal = level_history_signal(
                history,
                outer_rows,
                float(selected["half_life"]),
                float(selected["alpha"]),
            )
        outer_candidate = np.clip(
            outer_parent + float(selected["eta"]) * signal, 0.001, 0.999
        )
        outer = _diagnostics(
            outer_rows, outer_target, outer_parent, outer_candidate, outer_domain
        )
        summary["outer_diagnostics"] = outer
        summary["eligible_for_packaging"] = bool(
            outer["gain"] >= 5.0
            and outer["positive_month_fraction"] >= 0.75
            and outer["minimum_domain_gain"] > 0.0
            and outer["worst_month_gain"] > -10.0
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=outer_target,
            parent=outer_parent,
            candidate=outer_candidate,
            signal=signal,
            domain=outer_domain,
            game_month=outer_rows["game_month"].to_numpy(np.int16),
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v24_semantic_signal_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
