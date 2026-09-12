"""Joint robust routing of multi-year state and latent failure-mode signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import cluster_bootstrap_delta
from src.archive.multi_year_state_ensemble_screen import DOMAINS, _load_fold
from src.trackman_privileged_distillation import bss


ABS_RANGES = {
    "allmag": (0.0, np.inf),
    "lt005": (0.0, 0.005),
    "005to020": (0.005, 0.020),
    "020to040": (0.020, 0.040),
    "ge040": (0.040, np.inf),
    "lt020": (0.0, 0.020),
    "lt040": (0.0, 0.040),
}
BANDS = {
    "p_all": (0.0, 1.0),
    "p_20_80": (0.2, 0.8),
    "p_30_70": (0.3, 0.7),
    "p_40_60": (0.4, 0.6),
}
STATE_MULTIPLIERS = np.arange(0.25, 1.5001, 0.0625)
MODE_MULTIPLIERS = np.arange(0.0, 0.6001, 0.025)


def _mode_fold(mode_dir: Path, year: int) -> dict[str, object]:
    with np.load(
        mode_dir / f"latent_failure_mode_o{year}.npz", allow_pickle=True
    ) as z:
        output = {key: z[key] for key in z.files}
    output["names"] = [str(value) for value in output["names"].tolist()]
    return output


def _state_correction(
    fold: dict[str, object], domain: str, config: str
) -> np.ndarray:
    raw_name, sign, abs_name, band_name, agreement_text, weight_text = config.split(
        "|"
    )
    incumbent = fold["incumbent"]
    correction = fold["candidate"][raw_name] - incumbent
    magnitude = np.abs(correction)
    individual = fold["individual_correction"]
    agreement = np.where(
        (correction >= 0.0)[:, None], individual >= 0.0, individual < 0.0
    ).mean(axis=1)
    mask = fold["domain3"] == domain
    if sign == "positive":
        mask &= correction > 0.0
    elif sign == "negative":
        mask &= correction < 0.0
    low_abs, high_abs = ABS_RANGES[abs_name]
    low_p, high_p = BANDS[band_name]
    mask &= (magnitude >= low_abs) & (magnitude < high_abs)
    mask &= (incumbent >= low_p) & (incumbent <= high_p)
    mask &= agreement >= float(agreement_text) - 1e-12
    output = np.zeros(len(incumbent), dtype=np.float64)
    output[mask] = float(weight_text) * correction[mask]
    return output


def _top_configs(metrics: pd.DataFrame, domain: str, top_n: int) -> list[str]:
    source = metrics.loc[
        metrics["audit_year"].isin((2023, 2024)) & metrics["domain"].eq(domain)
    ]
    robust = (
        source.groupby("config", observed=True)["gain"]
        .agg(min_gain="min", mean_gain="mean")
        .reset_index()
        .sort_values(["min_gain", "mean_gain"], ascending=False, kind="stable")
    )
    return robust.head(top_n)["config"].tolist()


def _search_domain(
    domain: str,
    configs: list[str],
    state_folds: dict[int, dict[str, object]],
    mode_folds: dict[int, dict[str, object]],
) -> dict[str, object]:
    legal_names = [
        name
        for name in mode_folds[2023]["names"]
        if name != "oracle" and name in mode_folds[2024]["names"]
    ]
    constants: dict[int, dict[str, np.ndarray]] = {}
    for year in (2023, 2024):
        fold = state_folds[year]
        target = fold["target"]
        incumbent = fold["incumbent"]
        domain_mask = fold["domain3"] == domain
        error = incumbent[domain_mask] - target[domain_mask]
        state_matrix = np.column_stack(
            [_state_correction(fold, domain, config)[domain_mask] for config in configs]
        )
        mode_saved = mode_folds[year]
        mode_indices = [mode_saved["names"].index(name) for name in legal_names]
        mode_matrix = (
            mode_saved["raw"][:, mode_indices].astype(np.float64)
            - incumbent[:, None]
        )[domain_mask]
        reference = float(target.mean() * (1.0 - target.mean()))
        scale = -100000.0 / (reference * len(target))
        constants[year] = {
            "state_linear": scale * (2.0 * error) @ state_matrix,
            "state_quadratic": scale * np.sum(np.square(state_matrix), axis=0),
            "mode_linear": scale * (2.0 * error) @ mode_matrix,
            "mode_quadratic": scale * np.sum(np.square(mode_matrix), axis=0),
            "cross": scale * (2.0 * state_matrix.T @ mode_matrix),
        }

    best: tuple[float, float, int, int, int, int] | None = None
    a = STATE_MULTIPLIERS[:, None]
    b = MODE_MULTIPLIERS[None, :]
    for state_index, config in enumerate(configs):
        for mode_index, mode_name in enumerate(legal_names):
            gains = []
            for year in (2023, 2024):
                c = constants[year]
                gains.append(
                    a * c["state_linear"][state_index]
                    + np.square(a) * c["state_quadratic"][state_index]
                    + b * c["mode_linear"][mode_index]
                    + np.square(b) * c["mode_quadratic"][mode_index]
                    + a * b * c["cross"][state_index, mode_index]
                )
            minimum = np.minimum(gains[0], gains[1])
            mean = 0.5 * (gains[0] + gains[1])
            flat = int(np.argmax(minimum + 1e-9 * mean))
            ai, bi = np.unravel_index(flat, minimum.shape)
            value = (
                float(minimum[ai, bi]),
                float(mean[ai, bi]),
                state_index,
                mode_index,
                ai,
                bi,
            )
            if best is None or value[:2] > best[:2]:
                best = value
    if best is None:
        raise ValueError(f"no joint candidate for {domain}")
    minimum, mean, state_index, mode_index, ai, bi = best
    config = configs[state_index]
    mode_name = legal_names[mode_index]
    state_multiplier = float(STATE_MULTIPLIERS[ai])
    mode_multiplier = float(MODE_MULTIPLIERS[bi])
    fold_gains = {}
    for year in (2023, 2024):
        c = constants[year]
        fold_gains[str(year)] = float(
            state_multiplier * c["state_linear"][state_index]
            + state_multiplier**2 * c["state_quadratic"][state_index]
            + mode_multiplier * c["mode_linear"][mode_index]
            + mode_multiplier**2 * c["mode_quadratic"][mode_index]
            + state_multiplier
            * mode_multiplier
            * c["cross"][state_index, mode_index]
        )
    return {
        "domain": domain,
        "state_config": config,
        "state_multiplier": state_multiplier,
        "mode_name": mode_name,
        "mode_multiplier": mode_multiplier,
        "minimum_gain": minimum,
        "mean_gain": mean,
        "fold_gains": fold_gains,
    }


def _candidate(
    fold: dict[str, object],
    mode_saved: dict[str, object],
    choices: list[dict[str, object]],
) -> np.ndarray:
    incumbent = fold["incumbent"]
    output = incumbent.copy()
    for choice in choices:
        domain = str(choice["domain"])
        mask = fold["domain3"] == domain
        state = _state_correction(fold, domain, str(choice["state_config"]))
        mode_index = mode_saved["names"].index(str(choice["mode_name"]))
        mode = mode_saved["raw"][:, mode_index].astype(np.float64) - incumbent
        output[mask] = np.clip(
            incumbent[mask]
            + float(choice["state_multiplier"]) * state[mask]
            + float(choice["mode_multiplier"]) * mode[mask],
            0.001,
            0.999,
        )
    return output


def _diagnostics(
    year: int,
    fold: dict[str, object],
    candidate: np.ndarray,
    n_bootstrap: int,
) -> dict[str, object]:
    target = fold["target"]
    incumbent = fold["incumbent"]
    gain = bss(target, candidate) - bss(target, incumbent)
    months = []
    for month in sorted(np.unique(fold["game_month"])):
        mask = fold["game_month"] == month
        months.append(
            {
                "month": int(month),
                "n_rows": int(mask.sum()),
                "gain": bss(target[mask], candidate[mask])
                - bss(target[mask], incumbent[mask]),
            }
        )
    domains = []
    for domain in DOMAINS:
        mask = fold["domain3"] == domain
        domains.append(
            {
                "domain": domain,
                "n_rows": int(mask.sum()),
                "gain": bss(target[mask], candidate[mask])
                - bss(target[mask], incumbent[mask]),
            }
        )
    reference = float(target.mean() * (1.0 - target.mean()))
    cluster_schemes = {
        "pitcher": fold["pitcher_id"],
        "batter": fold["batter_id"],
        "pitcher_batter": (
            pd.Series(fold["pitcher_id"], dtype="string")
            + "\x1f"
            + pd.Series(fold["batter_id"], dtype="string")
        ).to_numpy(),
    }
    bootstraps = {}
    for offset, (name, clusters) in enumerate(cluster_schemes.items()):
        bootstrap = cluster_bootstrap_delta(
            target,
            candidate,
            incumbent,
            clusters,
            n_resamples=n_bootstrap,
            seed=8300 + year + 100 * offset,
        )
        bootstrap["gain_p05_fixed_reference"] = float(
            -100000.0 * bootstrap["ci_upper"] / reference
        )
        bootstrap["gain_p95_fixed_reference"] = float(
            -100000.0 * bootstrap["ci_lower"] / reference
        )
        bootstraps[name] = bootstrap
    return {
        "audit_year": year,
        "gain": gain,
        "candidate_bss": bss(target, candidate),
        "incumbent_bss": bss(target, incumbent),
        "positive_month_fraction": float(np.mean([row["gain"] > 0 for row in months])),
        "worst_month_gain": float(min(row["gain"] for row in months)),
        "minimum_domain_gain": float(min(row["gain"] for row in domains)),
        "months": months,
        "domains": domains,
        "cluster_bootstraps": bootstraps,
    }


def run(
    project: Path,
    variants_dir: Path,
    selected_dir: Path,
    state_metrics_path: Path,
    mode_dir: Path,
    output_dir: Path,
    top_n: int,
    n_bootstrap: int,
) -> dict[str, object]:
    project = project.resolve()
    variants_dir = (project / variants_dir).resolve()
    selected_dir = (project / selected_dir).resolve()
    state_metrics_path = (project / state_metrics_path).resolve()
    mode_dir = (project / mode_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_folds = {
        year: _load_fold(project, variants_dir, selected_dir, year)
        for year in (2023, 2024)
    }
    mode_folds = {year: _mode_fold(mode_dir, year) for year in (2023, 2024)}
    metrics = pd.read_parquet(state_metrics_path)
    choices = []
    for domain in DOMAINS:
        print(f"[joint-screen] domain={domain}", flush=True)
        configs = _top_configs(metrics, domain, top_n)
        choices.append(
            _search_domain(domain, configs, state_folds, mode_folds)
        )
    fold_results = []
    for year in (2023, 2024):
        candidate = _candidate(state_folds[year], mode_folds[year], choices)
        np.savez_compressed(
            output_dir / f"joint_candidate_o{year}.npz",
            target=state_folds[year]["target"],
            incumbent=state_folds[year]["incumbent"],
            candidate=candidate,
            domain3=state_folds[year]["domain3"],
            game_month=state_folds[year]["game_month"],
            pitcher_id=state_folds[year]["pitcher_id"],
            batter_id=state_folds[year]["batter_id"],
        )
        fold_results.append(
            _diagnostics(year, state_folds[year], candidate, n_bootstrap)
        )
    gains = [row["gain"] for row in fold_results]
    summary = {
        "protocol": "JOINT_STATE_LATENT_FAILURE_MODE_ROBUST_ROUTE_V1",
        "selection_caveat": (
            "Top state configs and route parameters use both 2023 and 2024; "
            "this is development evidence, not independent confirmation."
        ),
        "top_state_configs_per_domain": top_n,
        "choices": choices,
        "folds": fold_results,
        "aggregate": {
            "minimum_fold_gain": float(min(gains)),
            "mean_fold_gain": float(np.mean(gains)),
            "latest_fold_gain": float(fold_results[-1]["gain"]),
            "minimum_bootstrap_p05": float(
                min(
                    bootstrap["gain_p05_fixed_reference"]
                    for row in fold_results
                    for bootstrap in row["cluster_bootstraps"].values()
                )
            ),
            "minimum_positive_month_fraction": float(
                min(row["positive_month_fraction"] for row in fold_results)
            ),
            "minimum_domain_gain": float(
                min(row["minimum_domain_gain"] for row in fold_results)
            ),
            "positive_forward_folds": int(sum(gain > 0 for gain in gains)),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--variants-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_variants_20260816_01"),
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("artifacts/multi_year_state_selected_20260816_01"),
    )
    parser.add_argument(
        "--state-metrics-path",
        type=Path,
        default=Path("artifacts/multi_year_state_ensemble_20260816_01/metrics.parquet"),
    )
    parser.add_argument(
        "--mode-dir",
        type=Path,
        default=Path("artifacts/latent_failure_mode_state_20260816_01"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/state_mode_joint_20260816_01"),
    )
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()
    run(
        args.project,
        args.variants_dir,
        args.selected_dir,
        args.state_metrics_path,
        args.mode_dir,
        args.output_dir,
        args.top_n,
        args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
