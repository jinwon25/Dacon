"""Full-history conditional MLP and architecture-delta forward screen."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.v97_conditional_direct_forward import (
    CATEGORICAL,
    _bank,
    _diagnostics,
    _feature_frame,
    _load_contract_axis,
)


PROTOCOL = "V99_CONDITIONAL_MLP_AXIS_V1"


@dataclass
class Prepared:
    numeric_fit: np.ndarray
    categorical_fit: np.ndarray
    target_fit: np.ndarray
    weight_fit: np.ndarray
    numeric_audit: np.ndarray
    categorical_audit: np.ndarray
    cardinalities: list[int]
    numeric_columns: list[str]


def _category_codes(fit: pd.Series, audit: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    values = fit.astype("string").fillna("__MISSING__")
    vocabulary = {value: index + 1 for index, value in enumerate(values.unique())}
    fit_codes = values.map(vocabulary).fillna(0).to_numpy(np.int64)
    audit_codes = (
        audit.astype("string").fillna("__MISSING__")
        .map(vocabulary).fillna(0).to_numpy(np.int64)
    )
    return fit_codes, audit_codes, len(vocabulary) + 1


def prepare(
    fit: pd.DataFrame,
    audit: pd.DataFrame,
    fit_target: np.ndarray,
    fit_season: np.ndarray,
    audit_year: int,
    half_life: float,
) -> Prepared:
    categorical = set(CATEGORICAL)
    numeric_columns = sorted(column for column in fit.columns if column not in categorical)
    fit_values = fit[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    audit_values = audit[numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    fit_missing = ~np.isfinite(fit_values)
    audit_missing = ~np.isfinite(audit_values)
    keep_missing = fit_missing.any(axis=0)
    median = np.nanmedian(np.where(np.isfinite(fit_values), fit_values, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    fit_values = np.where(fit_missing, median, fit_values)
    audit_values = np.where(audit_missing, median, audit_values)
    mean = fit_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit_values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
    fit_values = np.clip((fit_values - mean) / scale, -8.0, 8.0)
    audit_values = np.clip((audit_values - mean) / scale, -8.0, 8.0)
    if keep_missing.any():
        fit_values = np.concatenate(
            [fit_values, fit_missing[:, keep_missing].astype(np.float32)], axis=1
        )
        audit_values = np.concatenate(
            [audit_values, audit_missing[:, keep_missing].astype(np.float32)], axis=1
        )
        numeric_columns += [
            f"{column}__missing"
            for column, keep in zip(numeric_columns, keep_missing, strict=True)
            if keep
        ]
    fit_codes, audit_codes, cardinalities = [], [], []
    for column in CATEGORICAL:
        left, right, cardinality = _category_codes(fit[column], audit[column])
        fit_codes.append(left)
        audit_codes.append(right)
        cardinalities.append(cardinality)
    weights = np.exp2(-((audit_year - 1) - fit_season) / float(half_life)).astype(np.float32)
    weights /= weights.mean()
    return Prepared(
        numeric_fit=fit_values,
        categorical_fit=np.column_stack(fit_codes),
        target_fit=np.asarray(fit_target, dtype=np.float32),
        weight_fit=weights,
        numeric_audit=audit_values,
        categorical_audit=np.column_stack(audit_codes),
        cardinalities=cardinalities,
        numeric_columns=numeric_columns,
    )


class ConditionalMLP(nn.Module):
    def __init__(
        self,
        numeric_count: int,
        cardinalities: list[int],
        widths: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = [min(8, max(2, int(round(value**0.25)) + 1)) for value in cardinalities]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(cardinalities, dimensions, strict=True)
            ]
        )
        layers: list[nn.Module] = []
        previous = numeric_count + sum(dimensions)
        for width in widths:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU(), nn.Dropout(dropout)])
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [
            layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)
        ]
        joined = torch.cat([numeric, *embedded], dim=1)
        return torch.sigmoid(self.network(joined).squeeze(1))


def _predict(
    model: ConditionalMLP,
    numeric: torch.Tensor,
    categorical: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            stop = min(start + batch_size, len(numeric))
            pieces.append(model(numeric[start:stop], categorical[start:stop]).numpy())
    return np.concatenate(pieces).astype(np.float64)


def train_predict(prepared: Prepared, network: dict[str, Any]) -> np.ndarray:
    seed = int(network["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(network["threads"]))
    numeric = torch.from_numpy(prepared.numeric_fit)
    categorical = torch.from_numpy(prepared.categorical_fit)
    target = torch.from_numpy(prepared.target_fit)
    weight = torch.from_numpy(prepared.weight_fit)
    audit_numeric = torch.from_numpy(prepared.numeric_audit)
    audit_categorical = torch.from_numpy(prepared.categorical_audit)
    model = ConditionalMLP(
        numeric.shape[1], prepared.cardinalities,
        [int(value) for value in network["widths"]], float(network["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(network["learning_rate"]),
        weight_decay=float(network["weight_decay"]),
    )
    batch_size = int(network["batch_size"])
    for epoch in range(1, int(network["epochs"]) + 1):
        model.train()
        order = torch.randperm(
            len(numeric), generator=torch.Generator().manual_seed(seed + epoch)
        )
        total_loss, total_weight = 0.0, 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            probability = model(numeric[index], categorical[index])
            batch_weight = weight[index]
            loss = torch.sum(batch_weight * torch.square(probability - target[index])) / torch.sum(batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * float(batch_weight.sum())
            total_weight += float(batch_weight.sum())
        print(f"[v99] epoch={epoch} weighted_brier={total_loss/total_weight:.9f}", flush=True)
    return np.clip(
        _predict(model, audit_numeric, audit_categorical, batch_size), 0.001, 0.999
    )


def _optimal_eta(
    target: np.ndarray, parent: np.ndarray, correction: np.ndarray, cap: float
) -> float:
    denominator = float(np.dot(correction, correction))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(np.dot(target - parent, correction) / denominator, 0.0, cap))


def _subset_exact(
    exact: dict[str, np.ndarray], direct: np.ndarray, correction: np.ndarray
) -> dict[str, np.ndarray]:
    mask = exact["exact_mask"].astype(bool) & exact["domain3"].astype(str).__eq__("R_CORE")
    parent = exact["parent"][mask].astype(np.float64)
    return {
        "target": exact["target"][mask].astype(np.float64),
        "parent": parent,
        "direct": parent + correction[mask],
        "domain3": exact["domain3"][mask],
        "game_month": exact["game_month"][mask],
        "mlp": direct[mask],
    }


def _align_direct(
    full_axis: dict[str, np.ndarray], exact_axis: dict[str, np.ndarray], prediction: np.ndarray
) -> np.ndarray:
    location = {int(value): index for index, value in enumerate(full_axis["raw_index"])}
    take = np.asarray([location[int(value)] for value in exact_axis["raw_index"]], dtype=np.int64)
    return prediction[take]


def run(
    train_csv: Path,
    contract_dir: Path,
    v97_config_path: Path,
    v97_prediction_dir: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    v97_config = json.loads(v97_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(train_csv, low_memory=False)
    audit_years = [int(value) for value in config["audit_years"]]
    feature_years = list(range(int(raw["season"].min()), max(audit_years) + 1))
    features: dict[int, pd.DataFrame] = {}
    for year in feature_years:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        history = raw.loc[raw["season"].lt(year)]
        features[year] = _feature_frame(
            rows, _bank(history, v97_config["conditional_strength"])
        )
        print(f"[v99 features] year={year} rows={len(rows)}", flush=True)

    mlp_predictions: dict[int, np.ndarray] = {}
    lgb_predictions: dict[int, np.ndarray] = {}
    common_axes: dict[int, dict[str, np.ndarray]] = {}
    for year in audit_years:
        fit_years = [value for value in feature_years if value < year]
        fit = pd.concat([features[value] for value in fit_years], ignore_index=True)
        audit = features[year]
        fit_mask = raw["season"].lt(year).to_numpy()
        prepared = prepare(
            fit, audit,
            raw.loc[fit_mask, "control_success"].to_numpy(np.float32),
            raw.loc[fit_mask, "season"].to_numpy(np.int16),
            year, float(config["recency_half_life"]),
        )
        print(
            f"[v99 model] audit_year={year} fit_rows={len(fit)} numeric={prepared.numeric_fit.shape[1]}",
            flush=True,
        )
        mlp_predictions[year] = train_predict(prepared, config["network"])
        with np.load(v97_prediction_dir / f"direct_full_{year}.npz") as saved:
            lgb_predictions[year] = saved["direct"].astype(np.float64)
        common_axes[year] = _load_contract_axis(contract_dir / f"common_full_{year}.npz")
        np.savez_compressed(
            output_dir / f"mlp_full_{year}.npz",
            raw_index=common_axes[year]["raw_index"],
            target=common_axes[year]["target"],
            prediction=mlp_predictions[year],
        )
        del fit, audit, prepared
        gc.collect()

    selection_exact = _load_contract_axis(contract_dir / "v84_full_2022.npz")
    selection_mask = selection_exact["exact_mask"].astype(bool) & (
        selection_exact["domain3"].astype(str) == "R_CORE"
    )
    selection_parent = selection_exact["parent"][selection_mask].astype(np.float64)
    selection_target = selection_exact["target"][selection_mask].astype(np.float64)
    family_rows = []
    family_corrections: dict[str, dict[int, np.ndarray]] = {
        "direct_mlp": {}, "architecture_delta": {}
    }
    for year in audit_years:
        family_corrections["direct_mlp"][year] = (
            mlp_predictions[year] - common_axes[year]["parent"].astype(np.float64)
        )
        family_corrections["architecture_delta"][year] = (
            mlp_predictions[year] - lgb_predictions[year]
        )
    for family, values in family_corrections.items():
        correction = (
            mlp_predictions[2022][selection_mask] - selection_parent
            if family == "direct_mlp"
            else values[2022][selection_mask]
        )
        eta = _optimal_eta(
            selection_target, selection_parent, correction,
            float(config["eta_caps"][family]),
        )
        selection_full_correction = (
            mlp_predictions[2022] - selection_exact["parent"].astype(np.float64)
            if family == "direct_mlp"
            else values[2022]
        )
        axis = _subset_exact(
            selection_exact, mlp_predictions[2022], selection_full_correction
        )
        metrics = _diagnostics(axis, ("R_CORE",), eta)
        gate = (
            metrics["gain"] > 0.0
            and metrics["positive_month_fraction"] >= float(config["selection_gate"]["minimum_positive_month_fraction"])
            and metrics["worst_month_gain"] > float(config["selection_gate"]["worst_month_gain_strictly_above"])
        )
        family_rows.append(
            {"family": family, "eta": eta, "selection_gate": gate, **metrics}
        )
    ledger = pd.DataFrame(family_rows).sort_values(
        ["selection_gate", "gain"], ascending=False
    ).reset_index(drop=True)
    ledger.to_csv(output_dir / "selection_ledger.csv", index=False)
    selected_family = str(ledger.iloc[0]["family"])
    selected_eta = float(ledger.iloc[0]["eta"])

    exact_metrics: dict[str, Any] = {}
    exact_files = {2022: "v84_full_2022.npz", 2023: "v84_late_2023.npz", 2024: "v84_full_2024.npz"}
    for year, filename in exact_files.items():
        exact = _load_contract_axis(contract_dir / filename)
        direct = _align_direct(common_axes[year], exact, mlp_predictions[year])
        correction_full = (
            direct - exact["parent"].astype(np.float64)
            if selected_family == "direct_mlp"
            else _align_direct(
                common_axes[year], exact,
                family_corrections[selected_family][year],
            )
        )
        axis = _subset_exact(exact, direct, correction_full)
        exact_metrics[str(year)] = _diagnostics(axis, ("R_CORE",), selected_eta)

    result = {
        "protocol": PROTOCOL,
        "selected_recipe": {"family": selected_family, "route": "R_CORE", "eta": selected_eta},
        "family_trial_count": int(len(ledger)),
        "selection_axis": config["selection_axis"],
        "locked_axes": config["locked_axes"],
        "exact_v84_diagnostics": exact_metrics,
        "test_csv_read": False,
        "test_aggregate_used": False,
        "row_local_inference": True,
        "eligible_for_packaging": False,
        "packaging_reason": "screen only; locked-axis gate and promotion statistics pending",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--v97-config", type=Path, required=True)
    parser.add_argument("--v97-prediction-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.train_csv, args.contract_dir, args.v97_config,
        args.v97_prediction_dir, args.config, args.output_dir,
    )


if __name__ == "__main__":
    main()
