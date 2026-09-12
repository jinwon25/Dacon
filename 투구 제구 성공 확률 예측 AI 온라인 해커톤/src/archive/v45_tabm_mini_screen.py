"""CPU-sized TabM-mini challenger above frozen v27.

This is a self-contained, dependency-free implementation of the core
parameter-efficient ensemble idea from TabM.  Ensemble members receive
different feature-wise affine views before the first shared linear layer,
share the MLP backbone, and use independent heads.  Member Brier losses are
optimized independently and probabilities are averaged only at inference.

The model uses only two immediately preceding seasons.  Architecture and
training policy are fixed; late 2023 selects checkpoint/domain/eta and the
chosen recipe is frozen before full and late 2024 are opened.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.core.axes import _joint_domain
from src.champion.v23_neural_embedding_screen import (
    CATEGORICAL,
    Prepared,
    _category_codes,
    _numeric_columns,
)
from src.core.axes import _derived, _load_axis
from src.core.axes import _early_to_late_2024
from src.core.diagnostics import diagnostics, v27_parent
from src.archive.v37_latest_season_catboost_residual import _attach_v25


ROUTES: dict[str, tuple[str, ...]] = {
    "R_CORE": ("R_CORE",),
    "R_ANCHOR": ("R_ANCHOR",),
    "R_ALL": ("R_CORE", "R_ANCHOR"),
    "F": ("F",),
    "ALL": ("R_CORE", "R_ANCHOR", "F"),
}
ETAS = (0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15)


def recent_history(raw: pd.DataFrame, through_year: int) -> pd.DataFrame:
    """Return exactly the two latest labelled seasons before an origin."""
    selected = raw.loc[
        raw["season"].between(through_year - 1, through_year)
    ].reset_index(drop=True)
    return _derived(selected, _joint_domain(selected))


def prepare_recent(fit: pd.DataFrame, audit: pd.DataFrame) -> Prepared:
    """Fit all preprocessing on the two source seasons only."""
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
    return Prepared(
        numeric_fit=numeric_fit,
        category_fit=np.column_stack(fit_codes),
        target_fit=fit["control_success"].to_numpy(np.float32),
        weight_fit=np.ones(len(fit), dtype=np.float32),
        numeric_audit=numeric_audit,
        category_audit=np.column_stack(audit_codes),
        numeric_columns=numeric_columns,
        cardinalities=cardinalities,
    )


class TabMMini(nn.Module):
    """Shared MLP with diversified inputs and independent ensemble heads."""

    def __init__(
        self,
        numeric_count: int,
        cardinalities: list[int],
        *,
        k: int = 8,
        width: int = 48,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.k = int(k)
        dimensions = [
            min(8, max(2, int(round(value**0.25)) + 1))
            for value in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dimension)
                for cardinality, dimension in zip(
                    cardinalities, dimensions, strict=True
                )
            ]
        )
        input_count = numeric_count + sum(dimensions)
        self.input_scale = nn.Parameter(torch.empty(self.k, input_count))
        nn.init.normal_(self.input_scale, mean=1.0, std=0.35)
        self.backbone = nn.Sequential(
            nn.Linear(input_count, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.head_weight = nn.Parameter(torch.empty(self.k, width))
        self.head_bias = nn.Parameter(torch.zeros(self.k))
        nn.init.kaiming_uniform_(self.head_weight, a=np.sqrt(5.0))

    def forward(
        self, numeric: torch.Tensor, category: torch.Tensor
    ) -> torch.Tensor:
        embedded = [
            layer(category[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        joined = torch.cat([numeric, *embedded], dim=1)
        # The member-specific views are created before the first feature-mixing
        # linear layer, which is the essential TabM-mini ordering constraint.
        member_input = joined[:, None, :] * self.input_scale[None, :, :]
        hidden = self.backbone(member_input)
        logits = torch.sum(
            hidden * self.head_weight[None, :, :], dim=2
        ) + self.head_bias[None, :]
        return torch.sigmoid(logits)


def _predict(
    model: TabMMini,
    numeric: torch.Tensor,
    category: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(numeric), batch_size):
            stop = min(start + batch_size, len(numeric))
            probability = model(
                numeric[start:stop], category[start:stop]
            )
            output.append(probability.mean(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def train_checkpoints(
    prepared: Prepared,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    k: int,
    width: int,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    numeric = torch.from_numpy(prepared.numeric_fit)
    category = torch.from_numpy(prepared.category_fit)
    target = torch.from_numpy(prepared.target_fit)
    audit_numeric = torch.from_numpy(prepared.numeric_audit)
    audit_category = torch.from_numpy(prepared.category_audit)
    model = TabMMini(
        numeric.shape[1], prepared.cardinalities, k=k, width=width
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=2e-4
    )
    checkpoints = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(
            len(numeric), generator=torch.Generator().manual_seed(seed + epoch)
        )
        loss_sum = 0.0
        rows = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            probability = model(numeric[index], category[index])
            # Do not average predictions before the loss.  Each implicit model
            # receives its own Brier loss, as required by TabM training.
            loss = torch.square(
                probability - target[index, None]
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(index)
            rows += len(index)
        checkpoints[epoch] = _predict(
            model, audit_numeric, audit_category, batch_size
        )
        print(
            f"[v45] epoch={epoch} member_mean_brier={loss_sum/rows:.9f}",
            flush=True,
        )
    return checkpoints


def apply_direct(
    frame: pd.DataFrame,
    direct: np.ndarray,
    route: str,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    parent = v27_parent(frame)
    active = frame["domain3"].astype(str).isin(ROUTES[route]).to_numpy()
    candidate = parent.copy()
    candidate[active] = np.clip(
        parent[active]
        + float(eta) * (np.asarray(direct)[active] - parent[active]),
        0.001,
        0.999,
    )
    return candidate, active


def selection_metrics(
    frame: pd.DataFrame,
    checkpoints: dict[int, np.ndarray],
) -> pd.DataFrame:
    parent = v27_parent(frame)
    rows = []
    for epoch, direct in checkpoints.items():
        for route in ROUTES:
            for eta in ETAS:
                candidate, active = apply_direct(frame, direct, route, eta)
                result = diagnostics(frame, parent, candidate, active)
                applied_gain = min(
                    result["domain_gains"][domain]
                    for domain in ROUTES[route]
                )
                rows.append(
                    {
                        "epoch": epoch,
                        "route": route,
                        "eta": eta,
                        "applied_domain_gain": applied_gain,
                        **{
                            key: value
                            for key, value in result.items()
                            if key not in {"months", "domain_gains"}
                        },
                    }
                )
    output = pd.DataFrame(rows)
    output["selection_score"] = output[
        ["gain", "worst_month_gain", "applied_domain_gain"]
    ].min(axis=1)
    output["passes_selection_gate"] = (
        output["gain"].gt(0.0)
        & output["positive_month_fraction"].eq(1.0)
        & output["worst_month_gain"].gt(0.0)
        & output["applied_domain_gain"].gt(0.0)
        & output["minimum_domain_gain"].gt(-5.0)
    )
    return output.sort_values(
        ["passes_selection_gate", "selection_score", "gain"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def run(
    project: Path,
    output_dir: Path,
    *,
    epochs: int = 2,
    batch_size: int = 8192,
    k: int = 8,
    width: int = 48,
) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)

    selection = _attach_v25(
        project,
        _load_axis(project, "y2023_early_to_late", raw),
        "selection_late_2023",
    )
    fit = recent_history(raw, 2022)
    prepared = prepare_recent(fit, selection)
    checkpoints = train_checkpoints(
        prepared,
        epochs=epochs,
        batch_size=batch_size,
        seed=45017,
        k=k,
        width=width,
    )
    metrics = selection_metrics(selection, checkpoints)
    metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    passing = metrics.loc[metrics["passes_selection_gate"]]
    selected = passing.iloc[0] if len(passing) else metrics.iloc[0]
    recipe = {
        "epoch": int(selected["epoch"]),
        "route": str(selected["route"]),
        "eta": float(selected["eta"]),
    }
    summary: dict[str, object] = {
        "protocol": "V45_TABM_MINI_RECENT2_ABOVE_V27_V1",
        "reference": {
            "paper": "https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html",
            "official_code": "https://github.com/yandex-research/tabm",
        },
        "architecture": {
            "type": "TabM-mini-inspired shared MLP ensemble",
            "k": k,
            "width": width,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "selection_fit": "2021+2022 only",
        "selection_axis": "late 2023 frozen v27 OOF",
        "selection_candidate_count": int(len(metrics)),
        "selection_gate_count": int(metrics["passes_selection_gate"].sum()),
        "chosen": recipe,
        "selection": {
            key: float(selected[key])
            for key in (
                "gain",
                "positive_month_fraction",
                "worst_month_gain",
                "minimum_domain_gain",
                "applied_domain_gain",
                "mean_abs_shift",
            )
        },
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if not bool(selected["passes_selection_gate"]):
        summary["decision"] = "Stop before 2024: no late-2023 recipe passed."
    else:
        del fit, prepared, checkpoints
        gc.collect()
        outer = _attach_v25(
            project,
            _load_axis(project, "y2023_to_y2024", raw),
            "outer_full_2024",
        )
        outer_fit = recent_history(raw, 2023)
        outer_prepared = prepare_recent(outer_fit, outer)
        outer_direct = train_checkpoints(
            outer_prepared,
            epochs=recipe["epoch"],
            batch_size=batch_size,
            seed=45017,
            k=k,
            width=width,
        )[recipe["epoch"]]
        outer_parent = v27_parent(outer)
        outer_candidate, outer_active = apply_direct(
            outer, outer_direct, recipe["route"], recipe["eta"]
        )
        outer_result = diagnostics(
            outer, outer_parent, outer_candidate, outer_active
        )

        _, late = _early_to_late_2024(project, raw)
        late = _attach_v25(project, late, "replication_late_2024")
        late_mask = outer["game_month"].ge(8).to_numpy()
        if not np.array_equal(
            outer["target"].to_numpy(np.float64)[late_mask],
            late["target"].to_numpy(np.float64),
        ):
            raise ValueError("full/late 2024 target alignment failure")
        late_parent = v27_parent(late)
        late_candidate, late_active = apply_direct(
            late,
            outer_direct[late_mask],
            recipe["route"],
            recipe["eta"],
        )
        late_result = diagnostics(
            late, late_parent, late_candidate, late_active
        )
        gates = {
            "outer_gain_at_least_5": outer_result["gain"] >= 5.0,
            "outer_month_fraction_at_least_075": outer_result[
                "positive_month_fraction"
            ]
            >= 0.75,
            "late_gain_positive": late_result["gain"] > 0.0,
            "late_all_months_positive": late_result[
                "positive_month_fraction"
            ]
            == 1.0,
            "both_minimum_domains_nonnegative": min(
                outer_result["minimum_domain_gain"],
                late_result["minimum_domain_gain"],
            )
            >= 0.0,
            "both_worst_months_above_minus_5": min(
                outer_result["worst_month_gain"],
                late_result["worst_month_gain"],
            )
            > -5.0,
        }
        summary.update(
            {
                "outer_audit_run": True,
                "outer_fit": "2022+2023 only",
                "outer_2024": outer_result,
                "late_2024": late_result,
                "gates": gates,
                "eligible_for_packaging": bool(all(gates.values())),
            }
        )
        np.savez_compressed(
            output_dir / "audit_predictions.npz",
            full_target=outer["target"].to_numpy(np.float64),
            full_parent=outer_parent,
            full_direct=outer_direct,
            full_candidate=outer_candidate,
            late_target=late["target"].to_numpy(np.float64),
            late_parent=late_parent,
            late_direct=outer_direct[late_mask],
            late_candidate=late_candidate,
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v45_tabm_mini_20260817_01"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    args = parser.parse_args()
    run(
        args.project,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k=args.k,
        width=args.width,
    )


if __name__ == "__main__":
    main()
