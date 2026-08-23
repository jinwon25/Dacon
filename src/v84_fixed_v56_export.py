"""Fit and export the frozen v56 shared FM as NumPy-only inference assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from src.core.diagnostics import v27_parent
from src.core.axes import _cached_v25_axes
from src.core.banks import _metadata
from src.v53_factorization_offset import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FIELDS,
    LEARNING_RATE,
    PAIR_INDEX,
    PAIR_NAMES,
    SEED,
    TARGET,
    WEIGHT_DECAY,
    FieldEncoder,
    PairwiseFM,
    _logit,
    _predict_raw,
    prepare_fields,
)
from src.v56_shared_horizon_fm import (
    RANK,
    equal_period_domain_centres,
    source_domain_weights,
)


RISK = "source_domain_equal"
ROUTE = "F"
ETA = 0.10
CORRECTION_CLIP = 0.25
SOURCE_PERIODS = (2023, 2024)


def numpy_raw(category: np.ndarray, embeddings: list[np.ndarray]) -> np.ndarray:
    """Evaluate the trained main-effect-free FM without a torch dependency."""

    code = np.asarray(category, dtype=np.int64)
    if code.ndim != 2 or code.shape[1] != len(FIELDS):
        raise ValueError("FM category matrix has invalid shape")
    if len(embeddings) != len(FIELDS):
        raise ValueError("FM embedding count mismatch")
    output = np.zeros(len(code), dtype=np.float64)
    for left, right in PAIR_INDEX:
        left_embedding = np.asarray(embeddings[left], dtype=np.float32)
        right_embedding = np.asarray(embeddings[right], dtype=np.float32)
        output += np.sum(
            left_embedding[code[:, left]] * right_embedding[code[:, right]],
            axis=1,
            dtype=np.float64,
        )
    return output


def _train_shared_model(
    source: pd.DataFrame,
    parent: np.ndarray,
    periods: np.ndarray,
    *,
    seed: int = SEED,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[FieldEncoder, PairwiseFM, dict[str, Any]]:
    prepared = prepare_fields(source).reset_index(drop=True)
    parent = np.asarray(parent, dtype=np.float64)
    periods = np.asarray(periods, dtype=np.int16)
    if not (len(prepared) == len(parent) == len(periods)):
        raise ValueError("v84 source arrays are not aligned")
    if tuple(sorted(np.unique(periods).tolist())) != SOURCE_PERIODS:
        raise ValueError("v84 requires the frozen 2023/2024 source periods")

    encoder = FieldEncoder.fit(prepared)
    source_code = encoder.transform(prepared)
    target = prepared[TARGET].to_numpy(np.float32)
    domains = prepared["domain3"].astype(str).to_numpy()
    weights = source_domain_weights(periods, domains, RISK)

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
    losses: list[float] = []
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
        print(f"[v84] epoch={epoch} weighted_logloss={losses[-1]:.8f}", flush=True)

    model.eval()
    source_raw = _predict_raw(model, source_code, batch_size)
    centres, group_centres = equal_period_domain_centres(
        source_raw, periods, domains
    )
    embeddings = [
        layer.weight.detach().cpu().numpy().astype(np.float32)
        for layer in model.embeddings
    ]
    exported_raw = numpy_raw(source_code, embeddings)
    parity = float(np.max(np.abs(source_raw - exported_raw)))
    if parity > 2e-6:
        raise ValueError(f"NumPy FM export parity failure: {parity}")
    group_totals = {
        f"{period}:{domain}": float(
            weights[(periods == period) & (domains == domain)].sum()
        )
        for period in sorted(np.unique(periods))
        for domain in sorted(np.unique(domains))
        if np.any((periods == period) & (domains == domain))
    }
    audit = {
        "rank": RANK,
        "risk": RISK,
        "route": ROUTE,
        "eta": ETA,
        "correction_clip": CORRECTION_CLIP,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout_training_only": DROPOUT,
        "source_periods": list(SOURCE_PERIODS),
        "source_rows": int(len(prepared)),
        "source_domain_rows": {
            str(key): int(value)
            for key, value in prepared["domain3"].value_counts().items()
        },
        "losses": losses,
        "domain_centres": centres,
        "source_period_domain_centres": group_centres,
        "source_group_weight_totals": group_totals,
        "numpy_export_max_abs_parity": parity,
    }
    return encoder, model, audit


def export_model(
    encoder: FieldEncoder,
    model: PairwiseFM,
    audit: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocabularies = {
        field: vocabulary
        for field, vocabulary in zip(FIELDS, encoder.vocabularies, strict=True)
    }
    (output_dir / "vocabularies.json").write_text(
        json.dumps(vocabularies, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    arrays = {
        f"field_{index}": layer.weight.detach().cpu().numpy().astype(np.float32)
        for index, layer in enumerate(model.embeddings)
    }
    np.savez_compressed(output_dir / "embeddings.npz", **arrays)
    metadata = {
        "protocol": "V84_FIXED_V56_F_ROUTE_NUMPY_EXPORT_V1",
        "fields": list(FIELDS),
        "pair_names": [list(pair) for pair in PAIR_NAMES],
        "pair_index": [list(pair) for pair in PAIR_INDEX],
        **audit,
        "row_local_inference": True,
        "test_aggregate_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )


def run(project: Path, output_dir: Path) -> dict[str, Any]:
    project = project.resolve()
    output_dir = output_dir.resolve()
    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    full23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    meta23 = _metadata(project, 2023)
    if not np.array_equal(
        meta23["target"], full23[TARGET].to_numpy(np.float64)
    ):
        raise ValueError("full-2023 target/order mismatch")
    prepared23 = prepare_fields(full23)
    if not np.array_equal(
        prepared23["domain3"].astype(str).to_numpy(), meta23["domain"]
    ):
        raise ValueError("full-2023 domain/order mismatch")

    axes = _cached_v25_axes(project, raw)
    late24 = axes["replication_late_2024"].reset_index(drop=True)
    parent24 = v27_parent(late24)
    source = pd.concat([full23, late24], ignore_index=True)
    parent = np.concatenate([meta23["parent"], parent24])
    periods = np.concatenate(
        [
            np.full(len(full23), 2023, dtype=np.int16),
            np.full(len(late24), 2024, dtype=np.int16),
        ]
    )
    encoder, model, audit = _train_shared_model(source, parent, periods)
    audit["source_windows"] = ["full_2023", "late_2024"]
    audit["source_window_rows"] = [int(len(full23)), int(len(late24))]
    audit["source_parent"] = [
        "selected_state_o2023 incumbent",
        "v27_parent(replication_late_2024)",
    ]
    export_model(encoder, model, audit, output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=float), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
