"""Choose one expansion dose on sources and audit it on two OOF contracts.

The route family comes from v241 and remains post-hoc research.  Only its
single common scale is selected here, using full-2022 and late-2023 from the
runtime-faithful contract.  Full-2024 and the legacy training-transform OOF
are confirmation/transport diagnostics and cannot rescue a source rejection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.archive.v173_h1_noncore_extension_audit import _robustness
from src.archive.v241_mechanism_aware_fallback_expansion import paired_metrics
from src.core.contract import _load_contract_axis


PROTOCOL = "V244_DUAL_CONTRACT_FALLBACK_DOSE_AUDIT_V1"
AXES = ("full_2022", "late_2023", "full_2024")
SOURCE_AXES = ("full_2022", "late_2023")
FAMILIES = ("evidence_proxy", "exact_parent")
SCALES = (0.50, 1.00, 1.25, 1.50, 2.00)


def scaled_candidate(base: np.ndarray, proposal: np.ndarray, scale: float) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if base.shape != proposal.shape:
        raise ValueError("dose arrays have different shapes")
    return np.clip(base + float(scale) * (proposal - base), 0.001, 0.999)


def select_scale(results: dict[str, Any]) -> float | None:
    eligible: list[tuple[float, float, float]] = []
    for scale in SCALES:
        key = f"{scale:g}"
        source = [
            results[key][family][axis]["gain"]
            for family in FAMILIES
            for axis in SOURCE_AXES
        ]
        if all(gain > 0.0 for gain in source):
            exact_source = [
                results[key]["exact_parent"][axis]["gain"] for axis in SOURCE_AXES
            ]
            eligible.append((min(exact_source), float(np.mean(exact_source)), scale))
    return None if not eligible else max(eligible)[2]


def restrictions() -> dict[str, bool]:
    return {
        "common_route_scale_only": True,
        "scale_selected_on_full_2022_and_late_2023_only": True,
        "full_2024_not_used_for_dose_selection": True,
        "legacy_and_runtime_faithful_contracts_reported": True,
        "route_family_was_discovered_with_full_2024_visible": True,
        "clean_locked_holdout_claim": False,
        "test_csv_read": False,
        "public_score_used_for_dose_selection": False,
    }


def _load_axes(contract_dir: Path, bridge_oof: Path) -> dict[str, dict[str, np.ndarray]]:
    axes = {
        "full_2022": _load_contract_axis(contract_dir / "v84_full_2022.npz"),
        "late_2023": _load_contract_axis(contract_dir / "v84_late_2023.npz"),
        "full_2024": _load_contract_axis(bridge_oof),
    }
    return {
        name: {**axis, "exact_mask": np.ones(len(axis["target"]), dtype=bool)}
        for name, axis in axes.items()
    }


def _evaluate_bundle(
    path: Path,
    axes: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, np.ndarray]]]]:
    results: dict[str, Any] = {f"{scale:g}": {} for scale in SCALES}
    candidates: dict[str, dict[str, dict[str, np.ndarray]]] = {
        f"{scale:g}": {} for scale in SCALES
    }
    with np.load(path, allow_pickle=False) as saved:
        for scale in SCALES:
            key = f"{scale:g}"
            for family in FAMILIES:
                results[key][family] = {}
                candidates[key][family] = {}
                for axis in AXES:
                    base = saved[f"baseline_{family}_{axis}"].astype(np.float64)
                    proposal = saved[f"candidate_{family}_{axis}"].astype(np.float64)
                    active = saved[f"active_{family}_{axis}"].astype(bool)
                    candidate = scaled_candidate(base, proposal, scale)
                    candidates[key][family][axis] = candidate
                    results[key][family][axis] = paired_metrics(
                        axes[axis], base, candidate, active
                    )
    return results, candidates


def run(
    legacy_axes: Path,
    runtime_axes: Path,
    contract_dir: Path,
    bridge_oof: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    axes = _load_axes(contract_dir, bridge_oof)
    legacy_results, legacy_candidates = _evaluate_bundle(legacy_axes, axes)
    runtime_results, runtime_candidates = _evaluate_bundle(runtime_axes, axes)
    selected = select_scale(runtime_results)

    robustness: dict[str, Any] | None = None
    saved_arrays: dict[str, np.ndarray] = {}
    if selected is not None:
        key = f"{selected:g}"
        robustness = {}
        for contract_name, path, candidates in (
            ("legacy_training_transform", legacy_axes, legacy_candidates),
            ("runtime_faithful", runtime_axes, runtime_candidates),
        ):
            with np.load(path, allow_pickle=False) as saved:
                base = saved["baseline_exact_parent_full_2024"].astype(np.float64)
                active = saved["active_exact_parent_full_2024"].astype(bool)
                family = [
                    candidates[f"{scale:g}"]["exact_parent"]["full_2024"]
                    for scale in SCALES
                ] + [base.copy()]
                candidate = candidates[key]["exact_parent"]["full_2024"]
                robustness[contract_name] = _robustness(
                    axes["full_2024"], base, candidate, active, family
                )
                for parent_family in FAMILIES:
                    for axis in AXES:
                        saved_arrays[
                            f"baseline_{contract_name}_{parent_family}_{axis}"
                        ] = saved[f"baseline_{parent_family}_{axis}"].astype(np.float64)
                        saved_arrays[
                            f"candidate_{contract_name}_{parent_family}_{axis}"
                        ] = candidates[key][parent_family][axis]
                        saved_arrays[
                            f"active_{contract_name}_{parent_family}_{axis}"
                        ] = saved[f"active_{parent_family}_{axis}"].astype(bool)

    source_pass = selected is not None
    locked_positive = bool(
        source_pass
        and all(
            result[f"{selected:g}"][family]["full_2024"]["gain"] > 0.0
            for result in (legacy_results, runtime_results)
            for family in FAMILIES
        )
    )
    if saved_arrays:
        np.savez_compressed(output_dir / "selected_candidate_axes.npz", **saved_arrays)
    summary = {
        "protocol": PROTOCOL,
        "status": (
            "research_candidate_dual_contract_positive"
            if source_pass and locked_positive
            else "dose_reject"
        ),
        "scale_grid": list(SCALES),
        "selection_objective": "maximise worst exact-parent source gain after all-parent source positivity",
        "selected_scale": selected,
        "effective_route_weights": None if selected is None else {
            "pressure_boundary_agreement": 0.30 * selected,
            "nonpressure_same_hand": 0.10 * selected,
            "nonpressure_opposite_hand_high52": 0.10 * selected,
        },
        "results": {
            "legacy_training_transform": legacy_results,
            "runtime_faithful": runtime_results,
        },
        "source_gate_passed": source_pass,
        "locked_all_contract_parent_gains_positive": locked_positive,
        "locked_robustness": robustness,
        "eligible_for_release_build": bool(source_pass and locked_positive),
        "eligible_for_packaging": False,
        "restrictions": restrictions(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-axes", type=Path, required=True)
    parser.add_argument("--runtime-axes", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge-oof", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.legacy_axes,
        args.runtime_axes,
        args.contract_dir,
        args.bridge_oof,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_scale": result["selected_scale"],
                "effective_route_weights": result["effective_route_weights"],
                "locked": None if result["selected_scale"] is None else {
                    contract: {
                        family: result["results"][contract][
                            f"{result['selected_scale']:g}"
                        ][family]["full_2024"]
                        for family in FAMILIES
                    }
                    for contract in result["results"]
                },
                "robustness": result["locked_robustness"],
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
