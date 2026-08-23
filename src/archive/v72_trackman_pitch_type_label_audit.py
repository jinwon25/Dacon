"""Compare training-only ASOF pitch labels with aligned TrackMan labels.

This determines whether TrackMan can materially improve the latent pitch-type
student through cleaner/additional labels.  The audit uses no target column and
never reads test.csv.  Three deployable modelling classes are compared;
TrackMan's ``other`` group is reported separately and excluded from accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.latent_pitch_type_state_model import TYPE_NAMES, reconstruct_current_pitch_type


TYPE_TO_LABEL = {name: index for index, name in enumerate(TYPE_NAMES)}


def label_agreement_rows(
    season: np.ndarray,
    reconstructed: np.ndarray,
    trackman_group: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trackman = np.asarray(
        [TYPE_TO_LABEL.get(str(value), -1) for value in trackman_group],
        dtype=np.int8,
    )
    reconstructed = np.asarray(reconstructed, dtype=np.int8)
    season = np.asarray(season, dtype=np.int16)
    valid_trackman = trackman >= 0
    valid_both = valid_trackman & (reconstructed >= 0)
    rows = []
    for name, mask in [
        ("ALL", np.ones(len(season), dtype=bool)),
        *[(str(year), season == year) for year in sorted(np.unique(season))],
    ]:
        local_tm = mask & valid_trackman
        local_both = mask & valid_both
        rows.append(
            {
                "season": name,
                "aligned_rows": int(mask.sum()),
                "three_class_trackman_rows": int(local_tm.sum()),
                "reconstructed_rows": int((mask & (reconstructed >= 0)).sum()),
                "both_labelled_rows": int(local_both.sum()),
                "agreement": float(
                    np.mean(reconstructed[local_both] == trackman[local_both])
                )
                if local_both.any()
                else None,
                "trackman_additional_labels": int(
                    np.sum(local_tm & (reconstructed < 0))
                ),
                "trackman_other_rows": int(
                    np.sum(mask & (np.asarray(trackman_group).astype(str) == "other"))
                ),
            }
        )
    confusion = pd.crosstab(
        pd.Categorical(
            [TYPE_NAMES[value] for value in reconstructed[valid_both]],
            categories=TYPE_NAMES,
        ),
        pd.Categorical(
            [TYPE_NAMES[value] for value in trackman[valid_both]],
            categories=TYPE_NAMES,
        ),
        rownames=["reconstructed"],
        colnames=["trackman"],
        dropna=False,
    ).reset_index()
    return pd.DataFrame(rows), confusion


def run(
    project: Path,
    alignment_dir: Path,
    latent_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    project = project.resolve()
    alignment_dir = alignment_dir.resolve()
    latent_dir = latent_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=["season", "pitch_type_group"],
        low_memory=False,
    )
    reconstructed = reconstruct_current_pitch_type(main)
    with np.load(alignment_dir / "pitch_alignment.npz", allow_pickle=False) as saved:
        main_index = saved["main_index"].astype(np.int64)
        trackman_index = saved["trackman_index"].astype(np.int64)
        aligned_season = saved["season"].astype(np.int16)
    aligned_reconstructed = reconstructed[main_index]
    aligned_group = trackman.iloc[trackman_index]["pitch_type_group"].astype(str).to_numpy()
    if not np.array_equal(
        aligned_season, main.iloc[main_index]["season"].to_numpy(np.int16)
    ):
        raise ValueError("main/alignment season mismatch")
    if not np.array_equal(
        aligned_season, trackman.iloc[trackman_index]["season"].to_numpy(np.int16)
    ):
        raise ValueError("TrackMan/alignment season mismatch")
    agreement, confusion = label_agreement_rows(
        aligned_season, aligned_reconstructed, aligned_group
    )

    classifier_rows: list[dict[str, object]] = []
    trackman_label = np.asarray(
        [TYPE_TO_LABEL.get(value, -1) for value in aligned_group], dtype=np.int8
    )
    for year in (2023, 2024):
        artifact = np.load(latent_dir / f"latent_pitch_type_o{year}.npz")
        probability = artifact["type_probability"].astype(np.float64)
        predicted = np.argmax(probability, axis=1).astype(np.int8)
        global_year_index = np.flatnonzero(main["season"].eq(year).to_numpy())
        if len(global_year_index) != len(predicted):
            raise ValueError(f"classifier row count mismatch for {year}")
        global_to_local = np.full(len(main), -1, dtype=np.int64)
        global_to_local[global_year_index] = np.arange(len(global_year_index))
        aligned_mask = aligned_season == year
        local_position = global_to_local[main_index[aligned_mask]]
        label = trackman_label[aligned_mask]
        valid = (local_position >= 0) & (label >= 0)
        rec = aligned_reconstructed[aligned_mask]
        both = valid & (rec >= 0)
        classifier_rows.append(
            {
                "season": year,
                "aligned_three_class_rows": int(valid.sum()),
                "accuracy_vs_trackman": float(
                    np.mean(predicted[local_position[valid]] == label[valid])
                ),
                "accuracy_vs_reconstructed_on_same_rows": float(
                    np.mean(predicted[local_position[both]] == rec[both])
                ),
                "label_agreement_on_classifier_rows": float(
                    np.mean(rec[both] == label[both])
                ),
            }
        )
    classifier = pd.DataFrame(classifier_rows)
    agreement.to_csv(output_dir / "label_agreement_by_season.csv", index=False)
    confusion.to_csv(output_dir / "label_confusion.csv", index=False)
    classifier.to_csv(output_dir / "classifier_recheck.csv", index=False)
    overall = agreement.loc[agreement["season"].eq("ALL")].iloc[0]
    maximum_accuracy_change = float(
        (
            classifier["accuracy_vs_trackman"]
            - classifier["accuracy_vs_reconstructed_on_same_rows"]
        ).abs().max()
    )
    result = {
        "protocol": "V72_TRACKMAN_VS_ASOF_PITCH_TYPE_LABEL_AUDIT_V1",
        "test_csv_read": False,
        "target_column_read_for_audit": False,
        "overall": overall.to_dict(),
        "classifier_recheck": classifier.to_dict(orient="records"),
        "maximum_classifier_accuracy_change": maximum_accuracy_change,
        "trackman_retraining_priority": (
            "low" if float(overall["agreement"]) >= 0.99 and maximum_accuracy_change < 0.01 else "review"
        ),
        "row_local_deployment_possible": True,
        "other_test_rows_required": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.project, args.alignment_dir, args.latent_dir, args.output_dir)


if __name__ == "__main__":
    main()
