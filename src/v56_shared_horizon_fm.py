"""Shared two-source low-rank interaction model above frozen v27.

Unlike v54, which averaged independently fitted seasonal FMs, this model fits
one set of interaction embeddings across the two latest legal OOF source
periods.  It therefore keeps only interaction structure that can be expressed
with shared parameters.  Two fixed risks are compared: uniform row risk and
equal total risk for every source-period x deployment-domain group.

The source parent logit remains a fixed per-row offset.  After training, each
source-period/domain correction mean is measured and the equal-period domain
mean is removed from audit corrections.  The model cannot transport a global
or domain calibration level.  Selection uses 2022 and late-2023 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v50_low_rank_pitcher_context import select_consensus
from src.v53_factorization_offset import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FIELDS,
    FieldEncoder,
    LEARNING_RATE,
    PairwiseFM,
    SEED,
    TARGET,
    WEIGHT_DECAY,
    _frame,
    _logit,
    _predict_raw,
    _screen,
    apply_offset,
    centre_by_domain,
    prepare_fields,
)


RANK = 16
RISKS = ("uniform", "source_domain_equal")


def source_domain_weights(
    periods: np.ndarray, domains: np.ndarray, risk: str
) -> np.ndarray:
    periods = np.asarray(periods)
    domains = np.asarray(domains).astype(str)
    if len(periods) != len(domains):
        raise ValueError("period/domain length mismatch")
    if risk == "uniform":
        return np.ones(len(periods), dtype=np.float32)
    if risk != "source_domain_equal":
        raise ValueError(f"unknown shared-FM risk: {risk}")
    keys = np.asarray(
        [f"{period}\x1f{domain}" for period, domain in zip(periods, domains, strict=True)]
    )
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    del unique
    weight = 1.0 / counts[inverse].astype(np.float64)
    weight /= weight.mean()
    return weight.astype(np.float32)


def equal_period_domain_centres(
    values: np.ndarray, periods: np.ndarray, domains: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Return domain centres giving every source period equal influence."""

    values = np.asarray(values, dtype=np.float64)
    periods = np.asarray(periods)
    domains = np.asarray(domains).astype(str)
    if not (len(values) == len(periods) == len(domains)):
        raise ValueError("centre input length mismatch")
    group_centres: dict[str, float] = {}
    domain_centres: dict[str, float] = {}
    for domain in sorted(np.unique(domains)):
        local = []
        for period in sorted(np.unique(periods)):
            selected = (periods == period) & (domains == domain)
            if selected.any():
                value = float(values[selected].mean())
                group_centres[f"{period}:{domain}"] = value
                local.append(value)
        domain_centres[str(domain)] = float(np.mean(local)) if local else 0.0
    return domain_centres, group_centres


def fit_shared_offset(
    source_frames: list[pd.DataFrame],
    source_parents: list[np.ndarray],
    source_periods: list[int],
    audit: pd.DataFrame,
    *,
    risk: str,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, object]]:
    if not (
        len(source_frames) == len(source_parents) == len(source_periods) == 2
    ):
        raise ValueError("shared FM requires exactly two source periods")
    prepared = [prepare_fields(frame).reset_index(drop=True) for frame in source_frames]
    source = pd.concat(prepared, ignore_index=True)
    parent = np.concatenate(
        [np.asarray(value, dtype=np.float64) for value in source_parents]
    )
    periods = np.concatenate(
        [np.full(len(frame), period, dtype=np.int16) for frame, period in zip(prepared, source_periods, strict=True)]
    )
    if len(source) != len(parent):
        raise ValueError("shared source parent length mismatch")
    audit = prepare_fields(audit).reset_index(drop=True)
    encoder = FieldEncoder.fit(source)
    source_code = encoder.transform(source)
    audit_code = encoder.transform(audit)
    target = source[TARGET].to_numpy(np.float32)
    domains = source["domain3"].astype(str).to_numpy()
    weights = source_domain_weights(periods, domains, risk)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    model = PairwiseFM(encoder.cardinalities, RANK, DROPOUT)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    category_tensor = torch.from_numpy(source_code)
    target_tensor = torch.from_numpy(target)
    parent_logit = torch.from_numpy(_logit(parent).astype(np.float32))
    weight_tensor = torch.from_numpy(weights)
    losses = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(category_tensor), generator=generator)
        numerator = 0.0
        denominator = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            correction = model(category_tensor[index])
            row_loss = F.binary_cross_entropy_with_logits(
                parent_logit[index] + correction,
                target_tensor[index],
                reduction="none",
            )
            loss = torch.sum(row_loss * weight_tensor[index]) / torch.sum(
                weight_tensor[index]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            numerator += float(torch.sum(row_loss.detach() * weight_tensor[index]))
            denominator += float(torch.sum(weight_tensor[index]))
        losses.append(numerator / denominator)

    source_raw = _predict_raw(model, source_code, batch_size)
    audit_raw = _predict_raw(model, audit_code, batch_size)
    centres, group_centres = equal_period_domain_centres(
        source_raw, periods, domains
    )
    correction = centre_by_domain(
        audit_raw, audit["domain3"].astype(str).to_numpy(), centres
    )
    correction = np.clip(correction, -0.25, 0.25)
    group_totals = {
        f"{period}:{domain}": float(weights[(periods == period) & (domains == domain)].sum())
        for period in sorted(np.unique(periods))
        for domain in sorted(np.unique(domains))
        if np.any((periods == period) & (domains == domain))
    }
    return correction, {
        "risk": risk,
        "rank": RANK,
        "seed": seed,
        "epochs": epochs,
        "source_periods": list(source_periods),
        "source_rows": [int(len(frame)) for frame in prepared],
        "losses": losses,
        "domain_centres": centres,
        "source_period_domain_centres": group_centres,
        "source_group_weight_totals": group_totals,
        "audit_seen_rates": {
            field: float((audit_code[:, index] != 0).mean())
            for index, field in enumerate(FIELDS)
        },
        "audit_mean_abs_correction": float(np.mean(np.abs(correction))),
        "audit_max_abs_correction": float(np.max(np.abs(correction))),
    }


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
        return saved["target"].astype(np.float64), saved["incumbent"].astype(np.float64)


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
    if not np.array_equal(meta22["target"], rows[2022][TARGET].to_numpy(np.float64)):
        raise ValueError("2022 target/order mismatch")
    prepared22 = prepare_fields(rows[2022])
    if not np.array_equal(prepared22["domain3"].astype(str).to_numpy(), meta22["domain"]):
        raise ValueError("2022 domain/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])
    frame23 = axes["selection_late_2023"]
    parent23 = v27_parent(frame23)
    frame24 = axes["outer_full_2024"]
    parent24 = v27_parent(frame24)
    late24 = axes["replication_late_2024"]
    parent_late24 = v27_parent(late24)

    bank22: dict[str, np.ndarray] = {}
    bank23: dict[str, np.ndarray] = {}
    fit_audits: dict[str, object] = {}
    for risk in RISKS:
        print(f"[v56] origin=2022 risk={risk}", flush=True)
        bank22[f"shared_{risk}"], fit_audits[f"2020_2021_to_2022_{risk}"] = fit_shared_offset(
            [rows[2020], rows[2021]],
            [wave[2020], wave[2021]],
            [2020, 2021],
            rows[2022],
            risk=risk,
        )
        print(f"[v56] origin=late2023 risk={risk}", flush=True)
        bank23[f"shared_{risk}"], fit_audits[f"2021_2022_to_late2023_{risk}"] = fit_shared_offset(
            [rows[2021], rows[2022]],
            [wave[2021], meta22["parent"]],
            [2021, 2022],
            frame23,
            risk=risk,
        )

    stage1 = _screen(frame22, meta22["parent"], bank22)
    stage2 = _screen(frame23, parent23, bank23)
    stage1.to_csv(output_dir / "selection_2022.csv", index=False)
    stage2.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(stage1, stage2)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "risk": str(chosen["signal"]).replace("shared_", ""),
        "domain": str(chosen["domain"]),
        "eta": float(chosen["weight"]),
    }
    selection_pass = bool(chosen["passes_consensus_gate"])

    print(f"[v56] frozen outer recipe={recipe}", flush=True)
    correction24, fit_audits["2022_late2023_to_2024_selected"] = fit_shared_offset(
        [rows[2022], frame23],
        [meta22["parent"], parent23],
        [2022, 2023],
        frame24,
        risk=recipe["risk"],
    )
    candidate24, active24 = apply_offset(
        frame24, parent24, correction24, recipe["domain"], recipe["eta"]
    )
    full_audit = diagnostics(frame24, parent24, candidate24, active24)
    late_mask = frame24["game_month"].ge(8).to_numpy()
    if not np.array_equal(frame24.loc[late_mask, "target"].to_numpy(np.float64), late24["target"].to_numpy(np.float64)):
        raise ValueError("late-2024 order mismatch")
    candidate_late, active_late = apply_offset(
        late24,
        parent_late24,
        correction24[late_mask],
        recipe["domain"],
        recipe["eta"],
    )
    late_audit = diagnostics(late24, parent_late24, candidate_late, active_late)
    gates = {
        "two_origin_consensus": selection_pass,
        "outer_gain_at_least_5": bool(full_audit["gain"] >= 5.0),
        "outer_month_fraction_at_least_075": bool(full_audit["positive_month_fraction"] >= 0.75),
        "outer_worst_month_above_minus_10": bool(full_audit["worst_month_gain"] > -10.0),
        "outer_minimum_domain_nonnegative": bool(full_audit["minimum_domain_gain"] >= 0.0),
        "replication_gain_positive": bool(late_audit["gain"] > 0.0),
        "replication_month_fraction_at_least_two_thirds": bool(late_audit["positive_month_fraction"] >= 2.0 / 3.0),
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
        "protocol": "V56_TWO_SOURCE_SHARED_INTERACTION_FM_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "configuration": {
            "rank": RANK,
            "risks": list(RISKS),
            "source_periods_per_origin": 2,
            "source_parent_logit_offset": True,
            "equal_period_domain_center_removed": True,
        },
        "fit_audits": fit_audits,
        "selection": "exact risk/route/damping consensus on 2022 and late-2023 only",
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
        "audits": {"outer_full_2024": full_audit, "replication_late_2024": late_audit},
        "gates": gates,
        "eligible_for_packaging": eligible,
        "row_local_inference": True,
        "test_aggregate_used": False,
        "audit_labels_used_for_selection": False,
        "family_audit_note": "2024 was previously inspected for v53/v54 FM variants; v56 is not pristine at family level",
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
        default=Path("artifacts/v56_shared_horizon_fm_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
