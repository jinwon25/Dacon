"""Rolling-origin low-rank interaction offset above frozen v27.

This experiment executes the factorization-machine family preregistered in
``reports/top1100/experiment_registry.csv`` but never previously run.  It is
deliberately narrower than the failed embedding MLPs: there are no hidden
layers or main effects.  Only twelve baseball-motivated pair interactions are
represented by low-rank dot products.

Each origin fits a logistic correction with the source OOF parent logit used
as a fixed offset.  Source-domain mean corrections are removed before the
next origin is scored, so the model can add resolution but cannot carry a
season-level calibration shift.  Vocabularies, centres, weights and model
parameters are source-only.  Evaluation rows are always independent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.temporal_stable_conditional import _add_domain_and_pressure
from src.core.diagnostics import diagnostics, v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.champion.v50_low_rank_pitcher_context import select_consensus


TARGET = "control_success"
RANKS = (8, 16)
ROUTES = ("ALL", "R_CORE", "R_ANCHOR", "F")
ETAS = (0.10, 0.25, 0.50, 1.00)
EPOCHS = 4
BATCH_SIZE = 8192
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-3
DROPOUT = 0.10
SEED = 42
EPS = 1e-5

FIELDS = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "count_state",
    "hand_matchup",
    "inning_bucket",
    "base_state",
    "domain3",
    "pressure",
)
PAIR_NAMES = (
    ("pitcher_id", "batter_id"),
    ("pitcher_id", "count_state"),
    ("pitcher_id", "hand_matchup"),
    ("pitcher_id", "domain3"),
    ("batter_id", "hand_matchup"),
    ("batter_id", "domain3"),
    ("pitcher_team_id", "batter_team_id"),
    ("count_state", "hand_matchup"),
    ("count_state", "base_state"),
    ("count_state", "pressure"),
    ("domain3", "pressure"),
    ("inning_bucket", "pressure"),
)
PAIR_INDEX = tuple((FIELDS.index(left), FIELDS.index(right)) for left, right in PAIR_NAMES)


def _logit(probability: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(probability, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(value / (1.0 - value))


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def prepare_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only row-local categorical fields used by the FM."""

    output = _add_domain_and_pressure(frame.copy())
    output["count_state"] = (
        output["balls_before"].astype("Int64").astype(str)
        + "-"
        + output["strikes_before"].astype("Int64").astype(str)
    )
    output["hand_matchup"] = (
        output["pitcher_hand"].astype("string").fillna("__MISSING__")
        + "-"
        + output["batter_hand"].astype("string").fillna("__MISSING__")
    )
    output["inning_bucket"] = pd.cut(
        pd.to_numeric(output["inning"], errors="coerce"),
        bins=(-np.inf, 3, 6, np.inf),
        labels=("early", "middle", "late"),
    ).astype("string")
    missing = sorted(set(FIELDS) - set(output.columns))
    if missing:
        raise ValueError(f"missing FM fields: {missing}")
    return output


@dataclass(frozen=True)
class FieldEncoder:
    vocabularies: tuple[dict[str, int], ...]

    @classmethod
    def fit(cls, source: pd.DataFrame) -> "FieldEncoder":
        vocabularies = []
        for field in FIELDS:
            values = source[field].astype("string").fillna("__MISSING__")
            vocabularies.append(
                {value: index + 1 for index, value in enumerate(values.unique())}
            )
        return cls(tuple(vocabularies))

    @property
    def cardinalities(self) -> list[int]:
        return [len(vocabulary) + 1 for vocabulary in self.vocabularies]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for field, vocabulary in zip(FIELDS, self.vocabularies, strict=True):
            values = frame[field].astype("string").fillna("__MISSING__")
            columns.append(values.map(vocabulary).fillna(0).to_numpy(np.int64))
        return np.column_stack(columns)


class PairwiseFM(nn.Module):
    """Main-effect-free field-aware factorization offset."""

    def __init__(self, cardinalities: list[int], rank: int, dropout: float):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, rank, padding_idx=0)
                for cardinality in cardinalities
            ]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.dropout = nn.Dropout(dropout)

    def forward(self, category: torch.Tensor) -> torch.Tensor:
        embedded = [
            layer(category[:, field_index])
            for field_index, layer in enumerate(self.embeddings)
        ]
        interactions = torch.stack(
            [
                torch.sum(embedded[left] * embedded[right], dim=1)
                for left, right in PAIR_INDEX
            ],
            dim=1,
        )
        return self.dropout(interactions).sum(dim=1)


def _predict_raw(
    model: PairwiseFM, category: np.ndarray, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(np.asarray(category, dtype=np.int64))
    output = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            output.append(model(tensor[start : start + batch_size]).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def domain_centres(
    values: np.ndarray, domains: np.ndarray
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    domains = np.asarray(domains).astype(str)
    if len(values) != len(domains):
        raise ValueError("value/domain length mismatch")
    return {
        domain: float(values[domains == domain].mean())
        for domain in sorted(np.unique(domains))
    }


def centre_by_domain(
    values: np.ndarray, domains: np.ndarray, centres: dict[str, float]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    domains = np.asarray(domains).astype(str)
    offsets = np.asarray([centres.get(domain, 0.0) for domain in domains])
    return values - offsets


def fit_predict_offset(
    source: pd.DataFrame,
    source_parent: np.ndarray,
    audit: pd.DataFrame,
    *,
    rank: int,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit on source OOF labels and return centred audit logit correction."""

    source = prepare_fields(source).reset_index(drop=True)
    audit = prepare_fields(audit).reset_index(drop=True)
    parent = np.asarray(source_parent, dtype=np.float64)
    if len(source) != len(parent):
        raise ValueError("source parent length mismatch")
    target = source[TARGET].to_numpy(np.float32)
    encoder = FieldEncoder.fit(source)
    source_code = encoder.transform(source)
    audit_code = encoder.transform(audit)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    model = PairwiseFM(encoder.cardinalities, int(rank), DROPOUT)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    source_tensor = torch.from_numpy(source_code)
    target_tensor = torch.from_numpy(target)
    parent_logit = torch.from_numpy(_logit(parent).astype(np.float32))
    losses = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(source_tensor), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            correction = model(source_tensor[index])
            loss = F.binary_cross_entropy_with_logits(
                parent_logit[index] + correction, target_tensor[index]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
        losses.append(loss_sum / len(source_tensor))

    source_raw = _predict_raw(model, source_code, batch_size)
    audit_raw = _predict_raw(model, audit_code, batch_size)
    centres = domain_centres(
        source_raw, source["domain3"].astype(str).to_numpy()
    )
    audit_centred = centre_by_domain(
        audit_raw, audit["domain3"].astype(str).to_numpy(), centres
    )
    audit_centred = np.clip(audit_centred, -0.25, 0.25)
    seen_rates = {
        field: float((audit_code[:, index] != 0).mean())
        for index, field in enumerate(FIELDS)
    }
    return audit_centred, {
        "rank": int(rank),
        "seed": int(seed),
        "epochs": int(epochs),
        "rows": int(len(source)),
        "losses": losses,
        "domain_centres": centres,
        "audit_seen_rates": seen_rates,
        "audit_mean_abs_correction": float(np.mean(np.abs(audit_centred))),
        "audit_max_abs_correction": float(np.max(np.abs(audit_centred))),
    }


def apply_offset(
    frame: pd.DataFrame,
    parent: np.ndarray,
    correction: np.ndarray,
    route: str,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.asarray(parent, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if not (len(frame) == len(parent) == len(correction)):
        raise ValueError("candidate length mismatch")
    active = (
        np.ones(len(frame), dtype=bool)
        if route == "ALL"
        else frame["domain3"].astype(str).eq(route).to_numpy()
    )
    candidate = parent.copy()
    candidate[active] = _expit(
        _logit(parent[active]) + float(eta) * correction[active]
    )
    return np.clip(candidate, 0.001, 0.999), active


def _screen(
    frame: pd.DataFrame,
    parent: np.ndarray,
    bank: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for signal, correction in bank.items():
        for route in ROUTES:
            for eta in ETAS:
                candidate, active = apply_offset(
                    frame, parent, correction, route, eta
                )
                result = diagnostics(frame, parent, candidate, active)
                applied_gain = (
                    min(result["domain_gains"].values())
                    if route == "ALL"
                    else result["domain_gains"][route]
                )
                rows.append(
                    {
                        "signal": signal,
                        "domain": route,
                        "weight": float(eta),
                        "gain": float(result["gain"]),
                        "positive_month_fraction": float(
                            result["positive_month_fraction"]
                        ),
                        "worst_month_gain": float(result["worst_month_gain"]),
                        "minimum_domain_gain": float(
                            result["minimum_domain_gain"]
                        ),
                        "applied_domain_gain": float(applied_gain),
                        "selection_score": float(
                            min(
                                result["gain"],
                                result["worst_month_gain"],
                                applied_gain,
                            )
                        ),
                        "mean_abs_shift": float(result["mean_abs_shift"]),
                    }
                )
    return pd.DataFrame(rows)


def _frame(
    target: np.ndarray, month: np.ndarray, domain: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": np.asarray(target, dtype=np.float64),
            "game_month": np.asarray(month, dtype=np.int16),
            "domain3": np.asarray(domain).astype(str),
        }
    )


def _load_wave0_2021(project: Path) -> tuple[np.ndarray, np.ndarray]:
    path = (
        project
        / "artifacts"
        / "followup"
        / "oof"
        / "wave0_incumbent_validate_2021.npz"
    )
    with np.load(path, allow_pickle=False) as saved:
        return (
            saved["target"].astype(np.float64),
            saved["incumbent"].astype(np.float64),
        )


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    rows = {
        year: raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        for year in (2021, 2022)
    }
    axes = _cached_v25_axes(project, raw)

    target21, parent21 = _load_wave0_2021(project)
    if not np.array_equal(target21, rows[2021][TARGET].to_numpy(np.float64)):
        raise ValueError("2021 wave0 target/order mismatch")
    meta22 = _metadata(project, 2022)
    if not np.array_equal(
        meta22["target"], rows[2022][TARGET].to_numpy(np.float64)
    ):
        raise ValueError("2022 target/order mismatch")
    audit22_rows = prepare_fields(rows[2022])
    if not np.array_equal(
        audit22_rows["domain3"].astype(str).to_numpy(), meta22["domain"]
    ):
        raise ValueError("2022 domain/order mismatch")
    frame22 = _frame(meta22["target"], meta22["month"], meta22["domain"])

    frame23 = axes["selection_late_2023"]
    parent23 = v27_parent(frame23)
    frame24 = axes["outer_full_2024"]
    parent24 = v27_parent(frame24)
    late24 = axes["replication_late_2024"]
    parent_late24 = v27_parent(late24)

    banks22: dict[str, np.ndarray] = {}
    banks23: dict[str, np.ndarray] = {}
    fit_audits: dict[str, object] = {}
    for rank in RANKS:
        name = f"fm_rank{rank}"
        print(f"[v53] selection origin=2022 rank={rank}", flush=True)
        banks22[name], fit_audits[f"2021_to_2022_rank{rank}"] = fit_predict_offset(
            rows[2021], parent21, rows[2022], rank=rank
        )
        print(f"[v53] selection origin=late2023 rank={rank}", flush=True)
        banks23[name], fit_audits[f"2022_to_late2023_rank{rank}"] = fit_predict_offset(
            rows[2022], meta22["parent"], frame23, rank=rank
        )

    selection22 = _screen(frame22, meta22["parent"], banks22)
    selection23 = _screen(frame23, parent23, banks23)
    selection22.to_csv(output_dir / "selection_2022.csv", index=False)
    selection23.to_csv(output_dir / "selection_late_2023.csv", index=False)
    consensus, chosen = select_consensus(selection22, selection23)
    consensus.to_csv(output_dir / "consensus_metrics.csv", index=False)
    recipe = {
        "signal": str(chosen["signal"]),
        "rank": int(str(chosen["signal"]).replace("fm_rank", "")),
        "domain": str(chosen["domain"]),
        "eta": float(chosen["weight"]),
    }
    selection_pass = bool(chosen["passes_consensus_gate"])

    print(f"[v53] frozen outer recipe={recipe}", flush=True)
    correction24, fit_audits["late2023_to_2024_selected"] = fit_predict_offset(
        frame23, parent23, frame24, rank=recipe["rank"]
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
        "protocol": "V53_ROLLING_DOMAIN_CENTERED_FM_OFFSET_V1",
        "parent": "submit_v27.zip / Public 1157.9736407889",
        "preregistration": "reports/top1100/experiment_registry.csv F1_fm_pilot_01/02",
        "configuration": {
            "ranks": list(RANKS),
            "pairs": [list(pair) for pair in PAIR_NAMES],
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "seed": SEED,
            "routes": list(ROUTES),
            "etas": list(ETAS),
            "main_effects": False,
            "source_domain_centered": True,
        },
        "fit_audits": fit_audits,
        "selection": "exact rank/route/damping consensus on 2022 and late-2023 only",
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
        default=Path("artifacts/v53_factorization_offset_20260817_01"),
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
