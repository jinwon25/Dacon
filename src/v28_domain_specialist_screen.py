"""Audit latest-season direct specialists for R_CORE and F above v27.

Each specialist is fit only on rows from its own baseball competition level.
Model and blend strength are selected on early-2023 -> late-2023, then frozen
for a full-2023 -> full-2024 outer audit and early-2024 -> late-2024 replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_v25_postbreak_anchor import ETA as V25_ETA
from src.v20_residual_overlay_screen import _bss
from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.v23_structural_residual_screen import _derived, _load_axis
from src.v25_postbreak_anchor_audit import _early_to_late_2024
from src.v26_exact_diversity_screen import _v25_axes


DOMAINS = ("R_CORE", "F")
ETAS = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
V27_ETA = 0.10


def _v27_parent(frame: pd.DataFrame) -> np.ndarray:
    v22 = frame["v22"].to_numpy(np.float64)
    v25 = frame["v25"].to_numpy(np.float64)
    return np.clip(v22 + (V27_ETA / V25_ETA) * (v25 - v22), 0.001, 0.999)


def _diagnostics(
    frame: pd.DataFrame,
    parent: np.ndarray,
    direct: np.ndarray,
    eta: float,
    domain: str,
) -> dict[str, object]:
    target = frame["target"].to_numpy(np.float64)
    apply_mask = frame["domain3"].astype(str).eq(domain).to_numpy()
    candidate = parent.copy()
    candidate[apply_mask] = np.clip(
        parent[apply_mask] + eta * (direct[apply_mask] - parent[apply_mask]),
        0.001,
        0.999,
    )

    def gain(mask: np.ndarray) -> float:
        return float(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )

    months = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        if np.any(mask & apply_mask):
            months.append(
                {
                    "month": int(month),
                    "rows": int(mask.sum()),
                    "applied_rows": int(np.sum(mask & apply_mask)),
                    "gain": gain(mask),
                }
            )
    return {
        "gain": gain(np.ones(len(frame), dtype=bool)),
        "applied_domain_gain": gain(apply_mask),
        "positive_active_month_fraction": float(
            np.mean([row["gain"] > 0.0 for row in months])
        ),
        "worst_active_month_gain": float(min(row["gain"] for row in months)),
        "mean_abs_shift": float(np.mean(np.abs(candidate - parent))),
        "months": months,
    }


def _domain_fit(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    return frame.loc[frame["domain3"].astype(str).eq(domain)].reset_index(drop=True)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    axes = _v25_axes(project, raw)
    for frame in axes.values():
        frame["v27"] = _v27_parent(frame)

    year23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    year23 = _derived(year23, _joint_domain(year23))
    early23 = year23.loc[year23["game_month"].le(7)].reset_index(drop=True)
    selection = axes["selection_late_2023"]

    selection_rows = []
    selection_predictions: dict[tuple[str, str], np.ndarray] = {}
    for domain in DOMAINS:
        fit = _domain_fit(early23, domain)
        for spec in SPECS:
            direct = _fit_predict(fit, selection, spec)
            selection_predictions[(domain, spec.name)] = direct
            for eta in ETAS:
                result = _diagnostics(
                    selection, selection["v27"].to_numpy(np.float64), direct, eta, domain
                )
                selection_rows.append(
                    {
                        "domain": domain,
                        "model": spec.name,
                        "family": spec.family,
                        "strength": spec.strength,
                        "eta": eta,
                        **{key: value for key, value in result.items() if key != "months"},
                    }
                )
            print(f"[specialist] selection {domain} {spec.name}", flush=True)
    metrics = pd.DataFrame(selection_rows)
    metrics["passes_selection_gate"] = (
        metrics["gain"].gt(0.0)
        & metrics["applied_domain_gain"].gt(0.0)
        & metrics["positive_active_month_fraction"].eq(1.0)
        & metrics["worst_active_month_gain"].gt(0.0)
    )
    metrics["selection_score"] = metrics[
        ["gain", "worst_active_month_gain"]
    ].min(axis=1)
    metrics = metrics.sort_values(
        ["passes_selection_gate", "selection_score", "gain"], ascending=False
    ).reset_index(drop=True)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)

    selected = []
    for domain in DOMAINS:
        eligible = metrics.loc[
            metrics["domain"].eq(domain) & metrics["passes_selection_gate"]
        ]
        if len(eligible):
            row = eligible.iloc[0]
            selected.append(
                {
                    "domain": domain,
                    "model": str(row["model"]),
                    "eta": float(row["eta"]),
                    "selection": {
                        "gain": float(row["gain"]),
                        "applied_domain_gain": float(row["applied_domain_gain"]),
                        "positive_active_month_fraction": float(
                            row["positive_active_month_fraction"]
                        ),
                        "worst_active_month_gain": float(row["worst_active_month_gain"]),
                    },
                }
            )

    audit_results = {}
    combined_predictions = {
        name: frame["v27"].to_numpy(np.float64).copy()
        for name, frame in axes.items()
    }
    for item in selected:
        domain = str(item["domain"])
        spec = next(spec for spec in SPECS if spec.name == str(item["model"]))
        eta = float(item["eta"])
        outer_direct = _fit_predict(_domain_fit(year23, domain), axes["outer_full_2024"], spec)
        year24 = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
        year24 = _derived(year24, _joint_domain(year24))
        early24 = year24.loc[year24["game_month"].le(7)].reset_index(drop=True)
        replication_direct = _fit_predict(
            _domain_fit(early24, domain), axes["replication_late_2024"], spec
        )
        axis_direct = {
            "selection_late_2023": selection_predictions[(domain, spec.name)],
            "outer_full_2024": outer_direct,
            "replication_late_2024": replication_direct,
        }
        audit_results[domain] = {}
        for axis_name, direct in axis_direct.items():
            frame = axes[axis_name]
            parent = frame["v27"].to_numpy(np.float64)
            audit_results[domain][axis_name] = _diagnostics(
                frame, parent, direct, eta, domain
            )
            mask = frame["domain3"].astype(str).eq(domain).to_numpy()
            combined_predictions[axis_name][mask] = np.clip(
                parent[mask] + eta * (direct[mask] - parent[mask]), 0.001, 0.999
            )

    combined = {}
    for axis_name, frame in axes.items():
        target = frame["target"].to_numpy(np.float64)
        parent = frame["v27"].to_numpy(np.float64)
        candidate = combined_predictions[axis_name]
        combined[axis_name] = {
            "gain": float(_bss(target, candidate) - _bss(target, parent)),
            "domain_gains": {
                domain: float(
                    _bss(
                        target[frame["domain3"].astype(str).eq(domain).to_numpy()],
                        candidate[frame["domain3"].astype(str).eq(domain).to_numpy()],
                    )
                    - _bss(
                        target[frame["domain3"].astype(str).eq(domain).to_numpy()],
                        parent[frame["domain3"].astype(str).eq(domain).to_numpy()],
                    )
                )
                for domain in DOMAINS
            },
        }

    gates = {
        "both_domains_selected": len(selected) == len(DOMAINS),
        "outer_combined_gain_at_least_5": combined.get("outer_full_2024", {}).get(
            "gain", -np.inf
        ) >= 5.0,
        "replication_combined_gain_positive": combined.get(
            "replication_late_2024", {}
        ).get("gain", -np.inf) > 0.0,
        "each_outer_domain_positive": bool(selected)
        and all(
            audit_results[str(item["domain"])]["outer_full_2024"][
                "applied_domain_gain"
            ]
            > 0.0
            for item in selected
        ),
        "each_replication_domain_positive": bool(selected)
        and all(
            audit_results[str(item["domain"])]["replication_late_2024"][
                "applied_domain_gain"
            ]
            > 0.0
            for item in selected
        ),
    }
    summary = {
        "protocol": "V28_DOMAIN_SPECIALIST_ABOVE_V27_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "selected": selected,
        "audits": audit_results,
        "combined": combined,
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
        default=Path("artifacts/v28_domain_specialist_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
