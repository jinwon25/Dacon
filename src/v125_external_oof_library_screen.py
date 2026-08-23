"""Source-only screen of independent annual OOF predictions above frozen v104.

The candidate and route are ranked only on full-2022 and late-2023.  The
matching 2024 predictions are loaded only after the source ranking is frozen,
and are reported as a transfer audit rather than used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROTOCOL = "V125_EXTERNAL_OOF_LIBRARY_SCREEN_V1"
SOURCE_AXES = ("full_2022", "late_2023")
AUDIT_AXES = ("full_2024", "late_2024")
ROUTES = {
    "ALL": ("R_CORE", "R_ANCHOR", "F"),
    "R_CORE": ("R_CORE",),
    "R_ANCHOR": ("R_ANCHOR",),
    "F": ("F",),
    "R": ("R_CORE", "R_ANCHOR"),
}


def _load_axis(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _gain(y: np.ndarray, parent: np.ndarray, candidate: np.ndarray) -> float:
    return float(400000.0 * np.mean(np.square(parent - y) - np.square(candidate - y)))


def _fit_alpha(
    targets: list[np.ndarray],
    parents: list[np.ndarray],
    corrections: list[np.ndarray],
    cap: float,
) -> float:
    error = np.concatenate([y - p for y, p in zip(targets, parents)])
    delta = np.concatenate(corrections)
    denominator = float(delta @ delta)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip((error @ delta) / denominator, 0.0, cap))


def _metrics(
    axis: dict[str, np.ndarray], parent: np.ndarray, correction: np.ndarray, alpha: float
) -> dict[str, float]:
    exact = axis["exact_mask"].astype(bool)
    target = axis["target"].astype(np.float64)
    candidate = np.clip(parent + alpha * correction, 0.001, 0.999)
    row_gain = np.square(parent - target) - np.square(candidate - target)
    months = []
    for month in np.unique(axis["game_month"]):
        mask = exact & (axis["game_month"] == month)
        months.append(_gain(target[mask], parent[mask], candidate[mask]))
    return {
        "gain": _gain(target[exact], parent[exact], candidate[exact]),
        "mean_abs_shift": float(np.mean(np.abs(candidate[exact] - parent[exact]))),
        "positive_month_fraction": float(np.mean(np.asarray(months) > 0.0)),
        "worst_month_gain": float(np.min(months)),
        "row_gain_se": float(400000.0 * np.std(row_gain[exact], ddof=1) / np.sqrt(exact.sum())),
    }


def _prediction_families(root: Path) -> list[tuple[str, dict[int, Path]]]:
    families = []
    for path_2022 in sorted(root.rglob("*2022.npy")):
        if "prediction" not in path_2022.name.lower():
            continue
        prefix = path_2022.name.removesuffix("2022.npy")
        paths = {year: path_2022.with_name(f"{prefix}{year}.npy") for year in (2022, 2023, 2024)}
        if all(path.is_file() for path in paths.values()):
            name = str(path_2022.parent.relative_to(root) / prefix.rstrip("_"))
            families.append((name.replace("\\", "/"), paths))
    return families


def _target_path(prediction_path: Path, year: int) -> Path:
    direct = prediction_path.parent / f"targets_{year}.npy"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"no target beside {prediction_path}")


def run(
    train_csv: Path,
    contract_dir: Path,
    v104_dir: Path,
    external_root: Path,
    output_dir: Path,
    alpha_cap: float,
    audit_top: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    axes = {
        "full_2022": _load_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_axis(contract_dir / "v84_full_2024.npz"),
    }
    late24 = axes["full_2024"]["game_month"].astype(np.int16) >= 8
    axes["late_2024"] = {key: np.asarray(value)[late24] for key, value in axes["full_2024"].items()}
    with np.load(v104_dir / "selected_axes.npz", allow_pickle=False) as saved:
        v104 = {name: saved[name].astype(np.float64) for name in (*SOURCE_AXES, *AUDIT_AXES)}

    expected = {
        year: raw.loc[raw["season"].eq(year), "control_success"].to_numpy(np.float64)
        for year in (2022, 2023, 2024)
    }
    late23 = raw.loc[raw["season"].eq(2023), "game_month"].ge(8).to_numpy()
    source_rows: list[dict[str, Any]] = []
    loaded_source: dict[str, dict[str, np.ndarray]] = {}
    rejected: list[dict[str, str]] = []
    families = _prediction_families(external_root)

    for candidate_id, paths in families:
        try:
            yearly: dict[int, np.ndarray] = {}
            for year in (2022, 2023):
                target = np.load(_target_path(paths[year], year), allow_pickle=False).astype(np.float64)
                if not np.array_equal(target, expected[year]):
                    raise ValueError(f"target/order mismatch {year}")
                prediction = np.load(paths[year], allow_pickle=False).astype(np.float64)
                if prediction.shape != target.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"invalid prediction {year}")
                if prediction.min() < 0.0 or prediction.max() > 1.0:
                    raise ValueError(f"probability outside [0,1] {year}")
                yearly[year] = prediction
            candidate = {"full_2022": yearly[2022], "late_2023": yearly[2023][late23]}
            loaded_source[candidate_id] = candidate
            for route, domains in ROUTES.items():
                correction = {}
                for name in SOURCE_AXES:
                    mask = np.isin(axes[name]["domain3"].astype(str), domains)
                    correction[name] = np.where(mask, candidate[name] - v104[name], 0.0)
                alpha = _fit_alpha(
                    [axes[name]["target"][axes[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
                    [v104[name][axes[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
                    [correction[name][axes[name]["exact_mask"].astype(bool)] for name in SOURCE_AXES],
                    alpha_cap,
                )
                metrics = {name: _metrics(axes[name], v104[name], correction[name], alpha) for name in SOURCE_AXES}
                source_rows.append({
                    "candidate_id": candidate_id,
                    "route": route,
                    "alpha": alpha,
                    **{f"{axis}_{key}": value for axis, values in metrics.items() for key, value in values.items()},
                    "source_min_gain": min(values["gain"] for values in metrics.values()),
                    "source_mean_gain": float(np.mean([values["gain"] for values in metrics.values()])),
                    "source_worst_month_gain": min(values["worst_month_gain"] for values in metrics.values()),
                    "source_min_positive_month_fraction": min(values["positive_month_fraction"] for values in metrics.values()),
                })
        except (FileNotFoundError, ValueError) as error:
            rejected.append({"candidate_id": candidate_id, "reason": str(error)})

    ranking = pd.DataFrame(source_rows).sort_values(
        ["source_min_gain", "source_mean_gain", "source_worst_month_gain"], ascending=False
    ).reset_index(drop=True)
    ranking.insert(0, "source_rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_dir / "source_ranking.csv", index=False)

    selected = ranking.head(audit_top).copy()
    audit_rows: list[dict[str, Any]] = []
    for row in selected.to_dict("records"):
        candidate_id = str(row["candidate_id"])
        paths = dict(families)[candidate_id]
        target_2024 = np.load(_target_path(paths[2024], 2024), allow_pickle=False).astype(np.float64)
        if not np.array_equal(target_2024, expected[2024]):
            raise ValueError(f"locked target/order mismatch: {candidate_id}")
        prediction_2024 = np.load(paths[2024], allow_pickle=False).astype(np.float64)
        candidate = {"full_2024": prediction_2024, "late_2024": prediction_2024[late24]}
        domains = ROUTES[str(row["route"])]
        metrics = {}
        for name in AUDIT_AXES:
            mask = np.isin(axes[name]["domain3"].astype(str), domains)
            correction = np.where(mask, candidate[name] - v104[name], 0.0)
            metrics[name] = _metrics(axes[name], v104[name], correction, float(row["alpha"]))
        audit_rows.append({
            **row,
            **{f"{axis}_{key}": value for axis, values in metrics.items() for key, value in values.items()},
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "locked_2024_audit.csv", index=False)
    result = {
        "protocol": PROTOCOL,
        "candidate_families_found": len(families),
        "source_routes_scored": len(ranking),
        "rejected_families": rejected,
        "alpha_cap": alpha_cap,
        "audit_top_selected_on_sources_only": audit_top,
        "selection_axes": list(SOURCE_AXES),
        "locked_audit_axes": list(AUDIT_AXES),
        "top_source_rows": ranking.head(min(audit_top, 20)).to_dict("records"),
        "locked_audit_rows": audit.to_dict("records"),
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "public_score_used_for_candidate_or_weight": False,
            "locked_2024_used_for_source_ranking": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "candidate_families_found": len(families),
        "source_routes_scored": len(ranking),
        "rejected": len(rejected),
        "top_locked_audit": audit.head(20).to_dict("records"),
    }, ensure_ascii=False, indent=2, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v104-dir", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-cap", type=float, default=0.5)
    parser.add_argument("--audit-top", type=int, default=25)
    args = parser.parse_args()
    run(args.train_csv, args.contract_dir, args.v104_dir, args.external_root, args.output_dir, args.alpha_cap, args.audit_top)


if __name__ == "__main__":
    main()
