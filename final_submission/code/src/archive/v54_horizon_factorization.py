"""Two-source horizon-stabilized factorization offsets above v27.

The experiment implements the preregistered horizon-MoE idea without a
learned evaluation-time gate.  At every origin, two independent domain-centred
FM offsets are trained on the two latest available OOF source periods.  Two
fixed experts are exposed to the selection protocol:

* ``mean``: equal average of the older and latest corrections;
* ``agree``: the same average only when the two corrections share a sign.

The exact rank/expert/route/damping recipe must pass both 2022 and late-2023.
The full and late-2024 labels are opened only after that recipe is frozen.
No source model sees another source period or any evaluation-row aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.champion.v50_low_rank_pitcher_context import select_consensus
from src.champion.v53_factorization_offset import (
    RANKS,
    TARGET,
    _frame,
    _screen,
    apply_offset,
    fit_predict_offset,
    prepare_fields,
)


EXPERTS = ("mean", "agree")


def combine_horizons(
    older: np.ndarray, latest: np.ndarray, expert: str
) -> np.ndarray:
    """Combine two independently centred source corrections."""

    older = np.asarray(older, dtype=np.float64)
    latest = np.asarray(latest, dtype=np.float64)
    if len(older) != len(latest):
        raise ValueError("horizon correction length mismatch")
    mean = 0.5 * (older + latest)
    if expert == "mean":
        return mean
    if expert == "agree":
        return np.where(older * latest > 0.0, mean, 0.0)
    raise ValueError(f"unknown horizon expert: {expert}")


def _load_wave0(
    project: Path, year: int
) -> tuple[np.ndarray, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / f"wave0_incumbent_validate_{year}.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["target"].astype(np.float64),
            saved["incumbent"].astype(np.float64),
        )


def _fit_pair_bank(
    older_rows: pd.DataFrame,
    older_parent: np.ndarray,
    latest_rows: pd.DataFrame,
    latest_parent: np.ndarray,
    audit_rows: pd.DataFrame,
    *,
    origin: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    bank: dict[str, np.ndarray] = {}
    audits: dict[str, object] = {}
    for rank in RANKS:
        print(f"[v54] origin={origin} rank={rank} source=older", flush=True)
        older, audits[f"{origin}_rank{rank}_older"] = fit_predict_offset(
            older_rows, older_parent, audit_rows, rank=rank
        )
        print(f"[v54] origin={origin} rank={rank} source=latest", flush=True)
        latest, audits[f"{origin}_rank{rank}_latest"] = fit_predict_offset(
            latest_rows, latest_parent, audit_rows, rank=rank
        )
        for expert in EXPERTS:
            bank[f"fm_rank{rank}_{expert}"] = combine_horizons(
                older, latest, expert
            )
    return bank, audits


def _fit_selected_pair(
    older_rows: pd.DataFrame,
    older_parent: np.ndarray,
    latest_rows: pd.DataFrame,
    latest_parent: np.ndarray,
    audit_rows: pd.DataFrame,
    *,
    rank: int,
    expert: str,
) -> tuple[np.ndarray, dict[str, object]]:
    older, older_audit = fit_predict_offset(
        older_rows, older_parent, audit_rows, rank=rank
    )
    latest, latest_audit = fit_predict_offset(
        latest_rows, latest_parent, audit_rows, rank=rank
    )
    return combine_horizons(older, latest, expert), {
        "older": older_audit,
        "latest": latest_audit,
    }


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2020, 2021, 2022)
    }
    axes = _cached_v25_axes(project, raw)

    wave = {}
    for year in (2020, 2021):
        target, parent = _load_wave0(project, year)
        if not np.array_equal(target, rows[year][TARGET].to_numpy(np.float64)):
            raise ValueError(f"{year} wave0 target/order mismatch")
        wave[year] = parent
    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], rows[2022][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 target/order mismatch")
    prepared22 = prepare_fields(rows[2022])
    if not np.array_equal(
        prepared22["domain3"].astype(str).to_numpy(), meta22["domain"]
    ):
        raise ValueError("2022 domain/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])

    frame23 = axes["selection_late_2023"]
    parent23 = v27_parent(frame23)
    frame24 = axes["outer_full_2024"]
    parent24 = v27_parent(frame24)
    late24 = axes["replication_late_2024"]
    parent_late24 = v27_parent(late24)

    bank22, fit22 = _fit_pair_bank(
        rows[2020],
        wave[2020],
        rows[2021],
        wave[2021],
        rows[2022],
        origin="2022",
    )
    bank23, fit23 = _fit_pair_bank(
        rows[2021],
        wave[2021],
        rows[2022],
        meta22["parent"],
        frame23,
        origin="late2023",
    )
    selection22 = _screen(frame22, meta22["parent"], bank22)
    selection23 = _screen(frame23, parent23, bank23)
    selection22.to_csv(output_dir / "selection_2022.csv", index=False)
    selection23.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(selection22, selection23)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    signal = str(chosen["signal"])
    tokens = signal.replace("fm_rank", "").split("_", maxsplit=1)
    recipe = {
        "signal": signal,
        "rank": int(tokens[0]),
        "expert": tokens[1],
        "domain": str(chosen["domain"]),
        "eta": float(chosen["weight"]),
    }
    selection_pass = bool(chosen["passes_consensus_gate"])

    print(f"[v54] frozen outer recipe={recipe}", flush=True)
    correction24, fit24 = _fit_selected_pair(
        rows[2022],
        meta22["parent"],
        frame23,
        parent23,
        frame24,
        rank=recipe["rank"],
        expert=recipe["expert"],
    )
    candidate24, active24 = apply_offset(
        frame24, parent24, correction24, recipe["domain"], recipe["eta"]
    )
    full_audit = diagnostics(frame24, parent24, candidate24, active24)
    late_mask = frame24["game_month"].ge(8).to_numpy()
    if not np.array_equal(
        frame24.loc[late_mask, "target"].to_numpy(np.float64),
        late24["target"].to_numpy(np.float64),
    ):
        raise ValueError("late-2024 order mismatch")
    candidate_late, active_late = apply_offset(
        late24,
        parent_late24,
        correction24[late_mask],
        recipe["domain"],
        recipe["eta"],
    )
    late_audit = diagnostics(
        late24, parent_late24, candidate_late, active_late
    )

    gates = {
        "two_origin_consensus": selection_pass,
        "outer_gain_at_least_5": bool(full_audit["gain"] >= 5.0),
        "outer_month_fraction_at_least_075": bool(
            full_audit["positive_month_fraction"] >= 0.75
        ),
        "outer_worst_month_above_minus_10": bool(
            full_audit["worst_month_gain"] > -10.0
        ),
        "outer_minimum_domain_nonnegative": bool(
            full_audit["minimum_domain_gain"] >= 0.0
        ),
        "replication_gain_positive": bool(late_audit["gain"] > 0.0),
        "replication_month_fraction_at_least_two_thirds": bool(
            late_audit["positive_month_fraction"] >= 2.0 / 3.0
        ),
    }
    eligible = bool(all(gates.values()))
    np.savez_compressed(
        output_dir / "outer_full_2024.npz",
        target=frame24["target"].to_numpy(np.float64),
        v27=parent24,
        correction=correction24,
        candidate=candidate24,
        active=active24,
    )
    summary = {
        "protocol": "V54_TWO_SOURCE_HORIZON_FACTORIZATION_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "preregistration": "reports/top1100/experiment_registry.csv F6_horizon_01",
        "configuration": {
            "ranks": list(RANKS),
            "experts": list(EXPERTS),
            "source_policy": "two latest independent OOF periods",
            "mean_weights": [0.5, 0.5],
            "agreement_rule": "zero when source corrections have different signs",
        },
        "fit_audits": {**fit22, **fit23, "selected_2024": fit24},
        "selection": "exact rank/expert/route/damping consensus on 2022 and late-2023 only",
        "chosen": {
            **recipe,
            "gain_2022": float(chosen["gain_2022"]),
            "gain_late_2023": float(chosen["gain_2023"]),
            "worst_month_2022": float(chosen["worst_month_gain_2022"]),
            "worst_month_late_2023": float(chosen["worst_month_gain_2023"]),
            "consensus_score": float(chosen["consensus_score"]),
        },
        "consensus_candidate_count": int(len(consensus)),
        "consensus_gate_count": int(consensus["passes_consensus_gate"].sum()),
        "audits": {
            "outer_full_2024": full_audit,
            "replication_late_2024": late_audit,
        },
        "gates": gates,
        "eligible_for_packaging": eligible,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
        "family_audit_note": "2024 was previously inspected for v53 short-horizon FM; v54 is not a pristine family-level audit",
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
        default=Path("artifacts/v54_horizon_factorization_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
