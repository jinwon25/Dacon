"""Three-stage audit for low-DOF count/hand failure profiles.

2022 ranks only profile identities, late 2023 selects direction and strength,
and 2024 is opened once. This prevents the high-cardinality pitcher profile
that dominated late 2023 from being selected after its 2024 collapse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.failure_mode_privileged_distillation import MODE_NAMES, reconstruct_failure_mode
from src.champion.v22_failure_mode_profiles import _group_probability, _temperature
from src.champion.v25_failure_profile_nested_screen import (
    DIRECTIONS,
    WEIGHTS,
    _champion_axis,
    _load_profiles,
    compose,
    diagnostics,
)
from src.champion.v24_semantic_signal_screen import _v22


POWERS = (0.75, 1.0, 1.25, 1.5)
PROFILE_NAMES = ("count_hands", "team_count_hands")


def _conditional_success(
    target: np.ndarray,
    label: np.ndarray,
    season: np.ndarray,
    audit_year: int,
    half_life: float = 0.5,
) -> np.ndarray:
    weight = np.exp2(-(audit_year - 1.0 - season.astype(np.float64)) / half_life)
    return np.asarray(
        [
            np.average(target[label == index], weights=weight[label == index])
            for index in range(len(MODE_NAMES))
        ],
        dtype=np.float64,
    )


def low_dof_2022_bank(
    train: pd.DataFrame,
    label: np.ndarray,
    raw_by_mode: np.ndarray,
) -> dict[str, np.ndarray]:
    history_mask = train["season"].lt(2022).to_numpy() & (label >= 0)
    audit_mask = train["season"].eq(2022).to_numpy()
    history = train.loc[history_mask].reset_index(drop=True)
    query = train.loc[audit_mask].reset_index(drop=True)
    history_label = label[history_mask]
    counts = np.bincount(history_label, minlength=len(MODE_NAMES)).astype(np.float64)
    global_probability = (counts + 1.0) / (counts.sum() + len(MODE_NAMES))
    global_prior = np.tile(global_probability, (len(query), 1))
    count_probability, _ = _group_probability(
        history,
        history_label,
        query,
        ("balls_before", "strikes_before", "pitcher_hand", "batter_hand"),
        global_prior,
        1200.0,
    )
    team_probability, _ = _group_probability(
        history,
        history_label,
        query,
        (
            "pitcher_team_id",
            "balls_before",
            "strikes_before",
            "pitcher_hand",
            "batter_hand",
        ),
        count_probability,
        400.0,
    )
    conditional_success = _conditional_success(
        train.loc[history_mask, "control_success"].to_numpy(np.float64),
        history_label,
        train.loc[history_mask, "season"].to_numpy(np.int16),
        2022,
    )
    output: dict[str, np.ndarray] = {}
    for profile_name, probability in (
        ("count_hands", count_probability),
        ("team_count_hands", team_probability),
    ):
        for power in POWERS:
            transformed = _temperature(probability, power)
            suffix = f"profile_{profile_name}_pow{power:g}"
            output[f"modeoutcome__{suffix}"] = np.sum(
                transformed * raw_by_mode, axis=1
            )
            output[f"conditional__{suffix}"] = transformed @ conditional_success
    return output


def _score_rows(
    names: list[str],
    bank: dict[str, np.ndarray],
    target: np.ndarray,
    parent: np.ndarray,
    month: np.ndarray,
    domain: np.ndarray,
    *,
    v21: np.ndarray | None = None,
    directions: tuple[str, ...] = ("blend_to_raw",),
) -> pd.DataFrame:
    if v21 is None:
        v21 = parent
    rows = []
    for name in names:
        for direction in directions:
            for weight in WEIGHTS:
                candidate = compose(v21, parent, bank[name], weight, direction)
                result = diagnostics(target, parent, candidate, month, domain)
                rows.append(
                    {
                        "candidate": name,
                        "direction": direction,
                        "weight": weight,
                        "gain": result["gain"],
                        "positive_month_fraction": result["positive_month_fraction"],
                        "worst_month_gain": result["worst_month_gain"],
                        "minimum_domain_gain": result["minimum_domain_gain"],
                        "selection_score": min(
                            result["gain"],
                            result["worst_month_gain"],
                            result["minimum_domain_gain"],
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["selection_score", "gain"], ascending=False
    ).reset_index(drop=True)


def run(
    project: Path,
    profile_dir: Path,
    output_dir: Path,
    prefilter: int = 6,
) -> dict[str, object]:
    project = project.resolve()
    profile_dir = (project / profile_dir).resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project / "data/train.csv", low_memory=False)
    label = reconstruct_failure_mode(train)
    with np.load(
        project
        / "artifacts/latent_failure_mode_state_20260817_02"
        / "latent_failure_mode_o2022.npz",
        allow_pickle=True,
    ) as saved:
        target22 = saved["target"].astype(np.float64)
        parent22 = saved["incumbent"].astype(np.float64)
        month22 = saved["game_month"].astype(np.int16)
        domain22 = saved["domain3"].astype(str)
        raw_by_mode22 = saved["raw_by_mode"].astype(np.float64)
    bank22 = low_dof_2022_bank(train, label, raw_by_mode22)
    stage1 = _score_rows(
        list(bank22), bank22, target22, parent22, month22, domain22
    )
    stage1.to_csv(output_dir / "stage1_2022.csv", index=False)
    ranked_names = (
        stage1.drop_duplicates("candidate").head(prefilter)["candidate"].tolist()
    )

    target23, month23, names23, raw23 = _load_profiles(
        profile_dir / "failure_mode_profiles_o2023.npz"
    )
    target24, month24, names24, raw24 = _load_profiles(
        profile_dir / "failure_mode_profiles_o2024.npz"
    )
    common = [name for name in ranked_names if name in names23 and name in names24]
    if len(common) != len(ranked_names):
        raise ValueError("a prefiltered profile is missing from later artifacts")

    late23 = month23 >= 8
    axis23 = _champion_axis(project, "y2023_early_to_late")
    rows23 = train.loc[
        train["season"].eq(2023) & train["game_month"].ge(8)
    ].reset_index(drop=True)
    v21_23 = axis23["v21"].astype(np.float64)
    domain23 = axis23["domain3"].astype(str)
    v22_23 = _v22(rows23, v21_23, domain23)
    bank23 = {name: raw23[late23, names23.index(name)] for name in common}
    stage2 = _score_rows(
        common,
        bank23,
        target23[late23],
        v22_23,
        month23[late23],
        domain23,
        v21=v21_23,
        directions=DIRECTIONS,
    )
    stage2.to_csv(output_dir / "stage2_late2023.csv", index=False)
    selected = stage2.iloc[0].to_dict()
    stage2_gate = bool(
        selected["selection_score"] > 0.0
        and selected["positive_month_fraction"] >= 1.0
    )

    summary: dict[str, object] = {
        "protocol": "V25_LOW_DOF_FAILURE_THREE_STAGE_V1",
        "stage1_prefilter": prefilter,
        "stage1_ranked_profiles": ranked_names,
        "selected": selected,
        "stage2_gate_passed": stage2_gate,
        "outer_audit_run": False,
        "eligible_for_packaging": False,
    }
    if stage2_gate:
        axis24 = _champion_axis(project, "y2023_to_y2024")
        rows24 = train.loc[train["season"].eq(2024)].reset_index(drop=True)
        v21_24 = axis24["v21"].astype(np.float64)
        domain24 = axis24["domain3"].astype(str)
        v22_24 = _v22(rows24, v21_24, domain24)
        raw24_selected = raw24[:, names24.index(str(selected["candidate"]))]
        candidate24 = compose(
            v21_24,
            v22_24,
            raw24_selected,
            float(selected["weight"]),
            str(selected["direction"]),
        )
        outer = diagnostics(target24, v22_24, candidate24, month24, domain24)
        late24 = month24 >= 8
        late = diagnostics(
            target24[late24],
            v22_24[late24],
            candidate24[late24],
            month24[late24],
            domain24[late24],
        )
        eligible = bool(
            outer["gain"] >= 5.0
            and outer["positive_month_fraction"] >= 0.75
            and outer["minimum_domain_gain"] > 0.0
            and outer["worst_month_gain"] > -10.0
            and late["gain"] > 0.0
            and late["minimum_domain_gain"] > 0.0
        )
        summary.update(
            {
                "outer_audit_run": True,
                "outer_2024": outer,
                "late_2024": late,
                "eligible_for_packaging": eligible,
            }
        )
        np.savez_compressed(
            output_dir / "outer_prediction.npz",
            target=target24,
            v21=v21_24,
            v22=v22_24,
            raw=raw24_selected,
            candidate=candidate24,
            game_month=month24,
            domain3=domain24,
        )
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
        "--profile-dir",
        type=Path,
        default=Path("artifacts/v22_failure_mode_profiles_20260817_02"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/v25_low_dof_failure_three_stage_20260817_01"),
    )
    parser.add_argument("--prefilter", type=int, default=6)
    args = parser.parse_args()
    run(args.project, args.profile_dir, args.output_dir, args.prefilter)


if __name__ == "__main__":
    main()
