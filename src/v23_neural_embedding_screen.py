"""Brier-trained neural embedding challenger above v22.

The model is intentionally representation-diverse from the tree and lookup
champion: player, team, count, hand, and game-state categories are embedded,
while row-local numeric state is standardized and passed through a small MLP.
Only pre-2023 labels are used for checkpoint/eta selection on late-2023 OOF.
The 2024 audit is run only when that selection gate is positive.
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

from src.core.overlay import _bss
from src.core.axes import _joint_domain
from src.core.axes import _derived, _load_axis


CATEGORICAL = (
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
    "count_state",
    "hand_matchup",
    "inning_bucket",
    "domain3",
)
EXCLUDED_NUMERIC = {
    "row_id",
    "season",
    "control_success",
    "target",
    "v21",
    "v22",
    *CATEGORICAL,
}
ETAS = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
DOMAINS = ("R_CORE", "R_ANCHOR", "F")
DOMAIN_SUBSETS = (
    ("R_CORE",),
    ("R_ANCHOR",),
    ("F",),
    ("R_CORE", "R_ANCHOR"),
    ("R_CORE", "F"),
    ("R_ANCHOR", "F"),
    DOMAINS,
)


@dataclass
class Prepared:
    numeric_fit: np.ndarray
    category_fit: np.ndarray
    target_fit: np.ndarray
    weight_fit: np.ndarray
    numeric_audit: np.ndarray
    category_audit: np.ndarray
    numeric_columns: list[str]
    cardinalities: list[int]


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in EXCLUDED_NUMERIC
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _category_codes(
    fit: pd.Series, audit: pd.Series
) -> tuple[np.ndarray, np.ndarray, int]:
    source = fit.astype("string").fillna("__MISSING__")
    vocabulary = {value: index + 1 for index, value in enumerate(source.unique())}
    fit_code = source.map(vocabulary).fillna(0).to_numpy(np.int64)
    audit_code = (
        audit.astype("string")
        .fillna("__MISSING__")
        .map(vocabulary)
        .fillna(0)
        .to_numpy(np.int64)
    )
    return fit_code, audit_code, len(vocabulary) + 1


def prepare(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    half_life: float = 2.0,
    weighting: str = "recency",
) -> Prepared:
    numeric_columns = _numeric_columns(fit)
    fit_numeric = fit[numeric_columns].apply(pd.to_numeric, errors="coerce")
    audit_numeric = audit[numeric_columns].apply(pd.to_numeric, errors="coerce")
    median = fit_numeric.median(axis=0).fillna(0.0)
    fit_numeric = fit_numeric.fillna(median)
    audit_numeric = audit_numeric.fillna(median)
    mean = fit_numeric.mean(axis=0)
    scale = fit_numeric.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    numeric_fit = np.clip(
        ((fit_numeric - mean) / scale).to_numpy(np.float32), -8.0, 8.0
    )
    numeric_audit = np.clip(
        ((audit_numeric - mean) / scale).to_numpy(np.float32), -8.0, 8.0
    )
    fit_codes = []
    audit_codes = []
    cardinalities = []
    for column in CATEGORICAL:
        fit_code, audit_code, cardinality = _category_codes(
            fit[column], audit[column]
        )
        fit_codes.append(fit_code)
        audit_codes.append(audit_code)
        cardinalities.append(cardinality)
    if weighting == "recency":
        latest = int(fit["season"].max())
        weight = np.exp2(
            -(latest - fit["season"].to_numpy(np.float64)) / half_life
        ).astype(np.float32)
    elif weighting == "uniform":
        weight = np.ones(len(fit), dtype=np.float32)
    elif weighting == "season_domain_equal":
        key = (
            fit["season"].astype("string")
            + "\x1f"
            + fit["domain3"].astype("string").fillna("__MISSING__")
        )
        count = key.map(key.value_counts()).to_numpy(np.float64)
        weight = (1.0 / count).astype(np.float32)
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    weight /= weight.mean()
    return Prepared(
        numeric_fit=numeric_fit,
        category_fit=np.column_stack(fit_codes),
        target_fit=fit["control_success"].to_numpy(np.float32),
        weight_fit=weight,
        numeric_audit=numeric_audit,
        category_audit=np.column_stack(audit_codes),
        numeric_columns=numeric_columns,
        cardinalities=cardinalities,
    )


class EmbeddingBrierNet(nn.Module):
    def __init__(self, numeric_count: int, cardinalities: list[int]):
        super().__init__()
        dimensions = [min(8, max(2, int(round(value ** 0.25)) + 1)) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension) for cardinality, dimension in zip(cardinalities, dimensions, strict=True)]
        )
        input_count = numeric_count + sum(dimensions)
        self.network = nn.Sequential(
            nn.Linear(input_count, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, numeric: torch.Tensor, category: torch.Tensor) -> torch.Tensor:
        embedded = [layer(category[:, index]) for index, layer in enumerate(self.embeddings)]
        joined = torch.cat([numeric, *embedded], dim=1)
        return torch.sigmoid(self.network(joined).squeeze(1))


def _predict(
    model: EmbeddingBrierNet,
    numeric: torch.Tensor,
    category: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            stop = min(start + batch_size, len(numeric))
            pieces.append(model(numeric[start:stop], category[start:stop]).numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_checkpoints(
    prepared: Prepared,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    numeric = torch.from_numpy(prepared.numeric_fit)
    category = torch.from_numpy(prepared.category_fit)
    target = torch.from_numpy(prepared.target_fit)
    weight = torch.from_numpy(prepared.weight_fit)
    audit_numeric = torch.from_numpy(prepared.numeric_audit)
    audit_category = torch.from_numpy(prepared.category_audit)
    model = EmbeddingBrierNet(numeric.shape[1], prepared.cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    output = {}
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(numeric), generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(numeric[index], category[index])
            batch_weight = weight[index]
            loss = torch.sum(batch_weight * torch.square(prediction - target[index])) / torch.sum(batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(torch.sum(batch_weight))
            weight_sum += float(torch.sum(batch_weight))
        output[epoch] = _predict(model, audit_numeric, audit_category, batch_size)
        print(f"[neural] epoch={epoch} weighted_brier={loss_sum/weight_sum:.9f}", flush=True)
    return output


def _diagnostics(
    frame: pd.DataFrame,
    direct: np.ndarray,
    eta: float,
    apply_domains: tuple[str, ...] = DOMAINS,
) -> dict[str, float]:
    target = frame["target"].to_numpy(np.float64)
    parent = frame["v22"].to_numpy(np.float64)
    domain_values = frame["domain3"].astype(str).to_numpy()
    apply_mask = np.isin(domain_values, apply_domains)
    candidate = parent.copy()
    candidate[apply_mask] = np.clip(
        parent[apply_mask] + eta * (direct[apply_mask] - parent[apply_mask]),
        0.001,
        0.999,
    )
    month_gains = []
    for month in sorted(frame["game_month"].unique()):
        mask = frame["game_month"].eq(month).to_numpy()
        month_gains.append(
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    domain_gains = {}
    for domain in DOMAINS:
        mask = frame["domain3"].eq(domain).to_numpy()
        domain_gains[domain] = (
            _bss(target[mask], candidate[mask]) - _bss(target[mask], parent[mask])
        )
    gain = _bss(target, candidate) - _bss(target, parent)
    minimum_applied_domain = min(domain_gains[domain] for domain in apply_domains)
    return {
        "gain": gain,
        "positive_month_fraction": float(np.mean(np.asarray(month_gains) > 0.0)),
        "worst_month_gain": float(min(month_gains)),
        "minimum_applied_domain_gain": float(minimum_applied_domain),
        **{f"{domain.lower()}_gain": float(value) for domain, value in domain_gains.items()},
        "selection_score": float(min(gain, min(month_gains), minimum_applied_domain)),
    }


def _history(raw: pd.DataFrame, through_year: int) -> pd.DataFrame:
    selected = raw.loc[raw["season"].le(through_year)].reset_index(drop=True)
    return _derived(selected, _joint_domain(selected))


def run(
    project: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16384,
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    selection = _load_axis(project, "y2023_early_to_late", raw)
    selection_fit = _history(raw, 2022)
    prepared = prepare(selection_fit, selection)
    predictions = train_checkpoints(prepared, epochs, batch_size, seed=23017)
    rows = []
    for epoch, prediction in predictions.items():
        for eta in ETAS:
            for domains in DOMAIN_SUBSETS:
                rows.append(
                    {
                        "epoch": epoch,
                        "eta": eta,
                        "domains": "+".join(domains),
                        **_diagnostics(selection, prediction, eta, domains),
                    }
                )
    metrics = pd.DataFrame(rows).sort_values(["selection_score", "gain"], ascending=False)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    selected = metrics.iloc[0]
    selection_gate = bool(
        selected["gain"] > 0.0
        and selected["worst_month_gain"] > 0.0
        and selected["minimum_applied_domain_gain"] > 0.0
    )
    summary: dict[str, object] = {
        "protocol": "V23_NEURAL_EMBEDDING_BRIER_V1",
        "selection": "fit 2019-2022; checkpoint and eta selected on honest late-2023 v22 OOF",
        "numeric_columns": prepared.numeric_columns,
        "cardinalities": prepared.cardinalities,
        "selected": {key: selected[key] for key in metrics.columns},
        "selection_gate_passed": selection_gate,
        "outer_audit_run": False,
    }
    if selection_gate:
        outer = _load_axis(project, "y2023_to_y2024", raw)
        outer_fit = _history(raw, 2023)
        outer_prepared = prepare(outer_fit, outer)
        outer_prediction = train_checkpoints(
            outer_prepared, int(selected["epoch"]), batch_size, seed=23017
        )[int(selected["epoch"])]
        selected_domains = tuple(str(selected["domains"]).split("+"))
        outer_diagnostics = _diagnostics(
            outer, outer_prediction, float(selected["eta"]), selected_domains
        )
        outer_gate = bool(
            outer_diagnostics["gain"] >= 5.0
            and outer_diagnostics["positive_month_fraction"] >= 0.75
            and outer_diagnostics["minimum_applied_domain_gain"] > 0.0
            and outer_diagnostics["worst_month_gain"] > -10.0
        )
        summary.update(
            {
                "outer_audit_run": True,
                "outer_diagnostics": outer_diagnostics,
                "eligible_for_packaging": outer_gate,
            }
        )
        parent = outer["v22"].to_numpy(np.float64)
        apply_mask = outer["domain3"].astype(str).isin(selected_domains).to_numpy()
        candidate = parent.copy()
        candidate[apply_mask] = np.clip(
            parent[apply_mask]
            + float(selected["eta"])
            * (outer_prediction[apply_mask] - parent[apply_mask]),
            0.001,
            0.999,
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=outer["target"].to_numpy(np.float64),
            v22=parent,
            neural=outer_prediction,
            candidate=candidate,
            apply_mask=apply_mask,
        )
    else:
        summary["eligible_for_packaging"] = False
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
        default=Path("artifacts/v23_neural_embedding_20260817_01"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16384)
    args = parser.parse_args()
    run(args.project, args.output_dir, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
