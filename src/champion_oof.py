"""Reconstruct the v19 -> v20 -> v21 forward OOF ladder.

The submitted archives contain the frozen 2025 artifacts, while follow-up
research needs the equivalent predictions on historical audit rows.  This
module rebuilds the two overlays with the same fixed recipes on three strictly
forward axes:

* 2023 full season -> 2024 full season
* 2023 March-July -> 2023 August-October
* 2024 March-July -> 2024 August-October

Every empirical-Bayes table is fitted on the left side of an axis and frozen
before it is mapped to the right side.  The latent-mode and PFD columns are
already season-forward OOF predictions produced by their source modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_v20_target1160 import OVERLAY_RECIPES
from src.train_v21_context_state_eb import RECIPES, add_v21_features
from src.v20_residual_overlay_screen import _bss, _feature_frame


MODE_NAME = "conditional_mode_lgb_h0.5_pow1.5"
MODE_WEIGHT = 0.045
PFD_NAME = "pfd_softlabel_l050"
PFD_WEIGHT = 0.26


def _load_year(project: Path, year: int) -> pd.DataFrame:
    columns = sorted(
        {
            "season",
            "game_month",
            "inning",
            "top_bottom",
            "outs_before",
            "balls_before",
            "strikes_before",
            "base_state",
            "li",
            "pitcher_id",
            "batter_id",
            "pitcher_hand",
            "batter_hand",
            "pitcher_team_id",
            "batter_team_id",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_reverse_rate",
            "control_success",
        }
    )
    pieces = []
    for chunk in pd.read_csv(
        project / "data" / "train.csv",
        usecols=columns,
        chunksize=200_000,
        low_memory=False,
    ):
        selected = chunk.loc[chunk["season"].eq(year)]
        if len(selected):
            pieces.append(selected.copy())
    if not pieces:
        raise ValueError(f"season {year} is absent from train.csv")
    frame = add_v21_features(_feature_frame(pd.concat(pieces, ignore_index=True)))

    joint_path = (
        project
        / "artifacts"
        / "state_mode_joint_20260816_02"
        / f"joint_candidate_o{year}.npz"
    )
    with np.load(joint_path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        if not np.array_equal(
            target, frame["control_success"].to_numpy(np.float64)
        ):
            raise ValueError(f"v19 OOF order mismatch for {year}")
        frame["target"] = target
        frame["v17"] = saved["incumbent"].astype(np.float64)
        frame["v19"] = saved["candidate"].astype(np.float64)
        frame["domain3"] = saved["domain3"].astype(str)

    mode_path = (
        project
        / "artifacts"
        / "latent_failure_mode_state_20260816_01"
        / f"latent_failure_mode_o{year}.npz"
    )
    with np.load(mode_path, allow_pickle=True) as saved:
        if not np.array_equal(target, saved["target"].astype(np.float64)):
            raise ValueError(f"latent-mode order mismatch for {year}")
        names = [str(value) for value in saved["names"].tolist()]
        mode = saved["raw"][:, names.index(MODE_NAME)].astype(np.float64)
        frame["v20_mode_delta"] = mode - frame["v17"].to_numpy(np.float64)

    pfd_path = (
        project
        / "artifacts"
        / "trackman_distillation_20260816_02"
        / f"distillation_o{year}.npz"
    )
    with np.load(pfd_path, allow_pickle=True) as saved:
        if not np.array_equal(target, saved["target"].astype(np.float64)):
            raise ValueError(f"PFD order mismatch for {year}")
        frame["v20_pfd_delta"] = saved[PFD_NAME].astype(np.float64)

    frame["residual_v19"] = frame["target"] - frame["v19"]
    return frame


def _key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, list(columns)])


def _eb_correction(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    domain: str,
    alpha: float,
    half_life: float | None = None,
) -> np.ndarray:
    fit_mask = np.ones(len(fit), dtype=bool)
    audit_mask = np.ones(len(audit), dtype=bool)
    if domain != "ALL":
        fit_mask &= fit["domain3"].eq(domain).to_numpy()
        audit_mask &= audit["domain3"].eq(domain).to_numpy()
    source = fit.loc[fit_mask].reset_index(drop=True)
    source_key = _key(source, columns)
    codes, unique = pd.factorize(source_key, sort=False)
    if half_life is None:
        row_weight = np.ones(len(source), dtype=np.float64)
    else:
        month = source["game_month"].to_numpy(np.float64)
        row_weight = np.exp2(-(float(month.max()) - month) / half_life)
    residual = source["residual_v19"].to_numpy(np.float64)
    numerator = np.bincount(
        codes, weights=row_weight * residual, minlength=len(unique)
    )
    effective_n = np.bincount(codes, weights=row_weight, minlength=len(unique))

    destination = audit.loc[audit_mask].reset_index(drop=True)
    destination_code = unique.get_indexer(_key(destination, columns))
    known = destination_code >= 0
    value = np.zeros(len(audit), dtype=np.float64)
    positions = np.flatnonzero(audit_mask)[known]
    index = destination_code[known]
    value[positions] = numerator[index] / (effective_n[index] + alpha)
    return value


def _v20(fit: pd.DataFrame, audit: pd.DataFrame) -> np.ndarray:
    correction = np.zeros(len(audit), dtype=np.float64)
    for recipe in OVERLAY_RECIPES:
        correction += float(recipe["weight"]) * _eb_correction(
            fit,
            audit,
            columns=tuple(str(value) for value in recipe["columns"]),
            domain=str(recipe["domain"]),
            alpha=float(recipe["alpha"]),
        )
    correction += MODE_WEIGHT * audit["v20_mode_delta"].to_numpy(np.float64)
    correction += PFD_WEIGHT * audit["v20_pfd_delta"].to_numpy(np.float64)
    return np.clip(audit["v19"].to_numpy(np.float64) + correction, 0.001, 0.999)


def _v21(fit: pd.DataFrame, audit: pd.DataFrame, v20: np.ndarray) -> np.ndarray:
    correction = np.zeros(len(audit), dtype=np.float64)
    for recipe in RECIPES:
        correction += float(recipe["weight"]) * _eb_correction(
            fit,
            audit,
            columns=tuple(str(value) for value in recipe["columns"]),
            domain=str(recipe["domain"]),
            alpha=float(recipe["alpha"]),
            half_life=float(recipe["half_life"]),
        )
    return np.clip(v20 + correction, 0.001, 0.999)


def _diagnostics(frame: pd.DataFrame, parent: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    gain = _bss(target, candidate) - _bss(target, parent)
    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        months.append(
            {
                "month": int(month),
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    domains = []
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        mask = frame["domain3"].eq(domain).to_numpy()
        domains.append(
            {
                "domain": domain,
                "rows": int(mask.sum()),
                "gain": _bss(target[mask], candidate[mask])
                - _bss(target[mask], parent[mask]),
            }
        )
    return {
        "gain": gain,
        "positive_month_fraction": float(
            np.mean([row["gain"] > 0.0 for row in months])
        ),
        "worst_month_gain": float(min(row["gain"] for row in months)),
        "minimum_domain_gain": float(min(row["gain"] for row in domains)),
        "months": months,
        "domains": domains,
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    years = {year: _load_year(project, year) for year in (2023, 2024)}
    axes = {
        "y2023_to_y2024": (years[2023], years[2024]),
        "y2023_early_to_late": (
            years[2023].loc[years[2023]["game_month"].le(7)].reset_index(drop=True),
            years[2023].loc[years[2023]["game_month"].ge(8)].reset_index(drop=True),
        ),
        "y2024_early_to_late": (
            years[2024].loc[years[2024]["game_month"].le(7)].reset_index(drop=True),
            years[2024].loc[years[2024]["game_month"].ge(8)].reset_index(drop=True),
        ),
    }
    rows = []
    for name, (fit, audit) in axes.items():
        v19 = audit["v19"].to_numpy(np.float64)
        v20 = _v20(fit, audit)
        v21 = _v21(fit, audit, v20)
        np.savez_compressed(
            output_dir / f"{name}.npz",
            target=audit["target"].to_numpy(np.float64),
            v17=audit["v17"].to_numpy(np.float64),
            v19=v19,
            v20=v20,
            v21=v21,
            season=audit["season"].to_numpy(np.int16),
            game_month=audit["game_month"].to_numpy(np.int16),
            domain3=audit["domain3"].astype(str).to_numpy(),
            pitcher_id=audit["pitcher_id"].to_numpy(),
            batter_id=audit["batter_id"].to_numpy(),
        )
        rows.append(
            {
                "axis": name,
                "fit_rows": int(len(fit)),
                "audit_rows": int(len(audit)),
                "v20_vs_v19": _diagnostics(audit, v19, v20),
                "v21_vs_v20": _diagnostics(audit, v20, v21),
                "v21_vs_v19": _diagnostics(audit, v19, v21),
            }
        )
        print(
            f"[{name}] v20-v19={rows[-1]['v20_vs_v19']['gain']:.9f} "
            f"v21-v20={rows[-1]['v21_vs_v20']['gain']:.9f}",
            flush=True,
        )
    summary = {
        "protocol": "CHAMPION_V19_V20_V21_FORWARD_OOF_V1",
        "mode_signal": {"name": MODE_NAME, "weight": MODE_WEIGHT},
        "pfd_signal": {"name": PFD_NAME, "weight": PFD_WEIGHT},
        "axes": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/champion_oof_20260817_01"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
