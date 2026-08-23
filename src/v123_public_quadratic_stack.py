"""Estimate a joint Public quadratic for the v17/v19/v20/v21/v22/v25 axes.

Official Public deltas identify the linear term of each sequential overlay once
its Brier curvature is known.  Curvatures and cross-correlations are estimated
from the strictly forward 2023 -> 2024 OOF ladder.  The v25 R_ANCHOR diagonal
and linear term use the exact quadratic identified by its three official Public
probes.  No test row, test aggregate, or evaluation label is read.

This is a research estimator, not a submission builder.  It reports sensitivity
to the uncertain transfer of OOF cross-terms before a combined package is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.train_v25_postbreak_anchor import MODEL_NAME
from src.v23_multiyear_direct_screen import _joint_domain
from src.v23_postbreak_gam_screen import SPECS, _fit_predict
from src.v23_structural_residual_screen import _derived, _load_axis


PROTOCOL = "V123_PUBLIC_QUADRATIC_STACK_RESEARCH_V1"
NAMES = (
    "v17_scale",
    "v19_scale",
    "v20_scale",
    "v21_scale",
    "v22_scale",
    "anchor_eta",
)
CURRENT = np.asarray((1.0, 1.0, 1.0, 1.0, 1.0, 0.15), dtype=np.float64)
BOUNDS = (
    (0.0, 2.0),
    (0.0, 1.5),
    (0.0, 1.5),
    (0.0, 1.5),
    (0.0, 1.5),
    (0.0, 0.60),
)
PUBLIC_SEQUENTIAL_GAINS = {
    "v17_vs_v13": 1093.3213473808 - 1068.4365711741,
    "v19_vs_v17": 1144.1518063753 - 1093.3213473808,
    "v20_vs_v19": 1151.4724287190 - 1144.1518063753,
    "v21_vs_v20": 1151.5138356157 - 1151.4724287190,
    "v22_vs_v21": 1153.0436023798 - 1151.5138356157,
}
ANCHOR_PUBLIC = {
    "q2": -57.016703333493766,
    "q1": 41.4194282233742,
    "q0": 1153.0436023803984,
}
V104_PUBLIC = 1162.6302840289


def _brier_gram(target: np.ndarray, directions: np.ndarray) -> np.ndarray:
    reference = float(target.mean() * (1.0 - target.mean()))
    return 100000.0 * (directions.T @ directions) / (len(target) * reference)


def _directions(project: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    ladder_path = project / "artifacts" / "champion_oof_20260817_01" / "y2023_to_y2024.npz"
    with np.load(ladder_path, allow_pickle=True) as saved:
        target = saved["target"].astype(np.float64)
        v17 = saved["v17"].astype(np.float64)
        v19 = saved["v19"].astype(np.float64)
        v20 = saved["v20"].astype(np.float64)
        v21 = saved["v21"].astype(np.float64)
        domain = saved["domain3"].astype(str)

    with np.load(
        project
        / "artifacts"
        / "v14_component_audit_20260815_01"
        / "v13_components_o2024.npz"
    ) as saved:
        if not np.array_equal(target, saved["target"].astype(np.float64)):
            raise ValueError("v13 OOF is not aligned with the champion ladder")
        v13 = saved["v13"].astype(np.float64)

    raw = pd.read_csv(project / "data" / "train.csv", low_memory=False)
    outer = _load_axis(project, "y2023_to_y2024", raw)
    if not np.array_equal(target, outer["target"].to_numpy(np.float64)):
        raise ValueError("v25 outer axis is not aligned with the champion ladder")
    v22 = outer["v22"].to_numpy(np.float64)

    year23 = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    year23 = _derived(year23, _joint_domain(year23))
    model_spec = next(spec for spec in SPECS if spec.name == MODEL_NAME)
    direct = _fit_predict(year23, outer, model_spec)
    anchor = domain == "R_ANCHOR"
    anchor_unit = np.zeros(len(target), dtype=np.float64)
    anchor_unit[anchor] = direct[anchor] - v22[anchor]

    directions = np.column_stack(
        (
            v17 - v13,
            v19 - v17,
            v20 - v19,
            v21 - v20,
            v22 - v21,
            anchor_unit,
        )
    )
    audit = {
        "rows": int(len(target)),
        "anchor_rows": int(anchor.sum()),
        "target_rate": float(target.mean()),
        "direction_mean_abs": {
            name: float(np.mean(np.abs(directions[:, index])))
            for index, name in enumerate(NAMES)
        },
    }
    return target, directions, audit


def _public_quadratic(local_gram: np.ndarray, cross_scale: float) -> tuple[np.ndarray, np.ndarray]:
    diagonal = np.diag(local_gram).copy()
    anchor_index = len(NAMES) - 1
    diagonal[anchor_index] = -float(ANCHOR_PUBLIC["q2"])
    correlation = local_gram / np.sqrt(
        np.maximum(np.diag(local_gram)[:, None] * np.diag(local_gram)[None, :], 1e-30)
    )
    gram = correlation * np.sqrt(diagonal[:, None] * diagonal[None, :])
    gram = np.diag(diagonal) + float(cross_scale) * (gram - np.diag(diagonal))

    sequential_keys = (
        "v17_vs_v13",
        "v19_vs_v17",
        "v20_vs_v19",
        "v21_vs_v20",
        "v22_vs_v21",
    )
    linear = np.zeros(len(NAMES), dtype=np.float64)
    for index, key in enumerate(sequential_keys):
        linear[index] = (
            PUBLIC_SEQUENTIAL_GAINS[key]
            + gram[index, index]
            + 2.0 * np.sum(gram[:index, index])
        )
    # The official anchor curve was measured with every preceding overlay active.
    linear[anchor_index] = float(ANCHOR_PUBLIC["q1"]) + 2.0 * np.sum(
        gram[:anchor_index, anchor_index]
    )
    return linear, gram


def _value(point: np.ndarray, linear: np.ndarray, gram: np.ndarray) -> float:
    return float(point @ linear - point @ gram @ point)


def _optimise(linear: np.ndarray, gram: np.ndarray) -> dict[str, object]:
    result = minimize(
        lambda point: -_value(point, linear, gram),
        CURRENT,
        method="L-BFGS-B",
        bounds=BOUNDS,
    )
    if not result.success:
        raise RuntimeError(result.message)
    selected = np.asarray(result.x, dtype=np.float64)
    current_value = _value(CURRENT, linear, gram)
    selected_value = _value(selected, linear, gram)
    return {
        "selected": {name: float(selected[index]) for index, name in enumerate(NAMES)},
        "current_surface_value": current_value,
        "selected_surface_value": selected_value,
        "projected_gain_vs_current": selected_value - current_value,
        "projected_v104_public": V104_PUBLIC + selected_value - current_value,
        "gradient_at_selected": (linear - 2.0 * gram @ selected).tolist(),
    }


def run(project: Path, output: Path) -> dict[str, object]:
    project = project.resolve()
    target, directions, audit = _directions(project)
    local_gram = _brier_gram(target, directions)
    scenarios = {}
    for cross_scale in (0.0, 0.5, 1.0):
        linear, gram = _public_quadratic(local_gram, cross_scale)
        scenarios[f"cross_{cross_scale:g}"] = {
            "cross_scale": cross_scale,
            "public_linear": linear.tolist(),
            "public_gram": gram.tolist(),
            **_optimise(linear, gram),
        }
    result = {
        "protocol": PROTOCOL,
        "method": (
            "official sequential Public gains + exact anchor Public quadratic; "
            "strict-forward 2023-to-2024 OOF Brier curvature/correlation"
        ),
        "current": {name: float(CURRENT[index]) for index, name in enumerate(NAMES)},
        "public_sequential_gains": PUBLIC_SEQUENTIAL_GAINS,
        "anchor_public_quadratic": ANCHOR_PUBLIC,
        "local_audit": audit,
        "local_gram": local_gram.tolist(),
        "local_correlation": np.corrcoef(directions, rowvar=False).tolist(),
        "scenarios": scenarios,
        "restrictions": {
            "test_csv_read": False,
            "test_aggregate_used": False,
            "external_outcomes_used": False,
            "research_estimate_only": True,
        },
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/v123_public_quadratic_stack_20260823_01/research.json"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.project / args.output
    run(args.project, output)


if __name__ == "__main__":
    main()
