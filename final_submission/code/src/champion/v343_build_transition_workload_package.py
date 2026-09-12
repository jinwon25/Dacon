"""Build the deployable v339 transition + workload portfolio above v335."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile


PROTOCOL = "V343_BUILD_TRANSITION_WORKLOAD_PACKAGE_V1"
FUNCTION_ANCHOR = "def _predict_player_transition(frame: pd.DataFrame) -> np.ndarray:\n"
WORKLOAD_MEMBER = "model/workload_h1/model/rf.pkl"
WORKLOAD_FUNCTIONS = '''def _minimum_common_denominator(
    success_rate: float, middle_rate: float, max_denominator: int
) -> tuple[int, bool]:
    rates = np.asarray([success_rate, middle_rate], dtype=np.float64)
    if not np.all(np.isfinite(rates)) or np.any((rates < 0.0) | (rates > 1.0)):
        return 0, False
    denominators = np.arange(1, int(max_denominator) + 1, dtype=np.float64)
    reconstructed = np.rint(denominators[:, None] * rates[None, :])
    reconstructed /= denominators[:, None]
    error = np.max(np.abs(reconstructed - rates[None, :]), axis=1)
    compatible = np.flatnonzero(error <= 0.5e-6 + 1e-12)
    index = int(compatible[0]) if compatible.size else int(np.argmin(error))
    return index + 1, bool(compatible.size)


def _attach_workload_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct v209 denominator features independently for every row."""

    output = frame.copy()
    career_success = pd.to_numeric(
        output["asof_pitcher_success_rate"], errors="coerce"
    ).fillna(0.5).to_numpy(np.float64)
    career_middle = pd.to_numeric(
        output["asof_pitcher_middle_rate"], errors="coerce"
    ).fillna(0.2).to_numpy(np.float64)
    denominators = {}
    for horizon, max_denominator, shrinkage in (
        (1, 240, 20.0), (3, 600, 60.0), (5, 1000, 100.0)
    ):
        success = pd.to_numeric(
            output[f"asof_pitcher_prev{horizon}_game_success_rate"],
            errors="coerce",
        ).to_numpy(np.float64)
        middle = pd.to_numeric(
            output[f"asof_pitcher_prev{horizon}_game_middle_rate"],
            errors="coerce",
        ).to_numpy(np.float64)
        n = np.zeros(len(output), dtype=np.float64)
        fit = np.zeros(len(output), dtype=bool)
        for index, (success_rate, middle_rate) in enumerate(zip(success, middle)):
            denominator, matched = _minimum_common_denominator(
                success_rate, middle_rate, max_denominator
            )
            n[index] = denominator
            fit[index] = matched
        valid = fit & np.isfinite(success) & np.isfinite(middle) & (n > 0.0)
        denominators[horizon] = n
        reliability = n / (n + shrinkage)
        output[f"workload_prev{horizon}_log_n"] = np.log1p(n).astype(np.float32)
        output[f"workload_prev{horizon}_missing"] = (~valid).astype(np.float32)
        output[f"workload_prev{horizon}_reliability"] = reliability.astype(np.float32)
        output[f"workload_prev{horizon}_success_reliable_delta"] = np.where(
            valid, reliability * (success - career_success), 0.0
        ).astype(np.float32)
        output[f"workload_prev{horizon}_middle_reliable_delta"] = np.where(
            valid, reliability * (middle - career_middle), 0.0
        ).astype(np.float32)
    n1, n3, n5 = (denominators[horizon] for horizon in (1, 3, 5))
    output["workload_prev3_minus_prev1_log"] = np.log1p(
        np.maximum(n3 - n1, 0.0)
    ).astype(np.float32)
    output["workload_prev5_minus_prev3_log"] = np.log1p(
        np.maximum(n5 - n3, 0.0)
    ).astype(np.float32)
    output["workload_prev1_over_prev3"] = np.divide(
        n1, n3, out=np.zeros(len(output)), where=n3 > 0.0
    ).astype(np.float32)
    output["workload_prev3_over_prev5"] = np.divide(
        n3, n5, out=np.zeros(len(output)), where=n5 > 0.0
    ).astype(np.float32)
    return output


def _predict_workload_h1(frame: pd.DataFrame) -> np.ndarray:
    root = MODEL_DIR / "workload_h1"
    module = _load_module("v343_workload_h1_component", MODEL_DIR / "h1" / "script.py")
    bundle = joblib.load(root / "model" / "rf.pkl")
    prepared = module.attach_ctx(frame.copy(), bundle)
    if any(column in (bundle.get("features") or []) for column in module.CAAFE_COLS):
        prepared = module.attach_caafe(prepared)
    if any(column in (bundle.get("features") or []) for column in module.ASOF_COLS):
        prepared = module.attach_asof_state(prepared, bundle)
    prepared = _attach_workload_features(prepared)
    features = module.build_features(prepared, bundle)
    return np.asarray(module.predict_proba(bundle, features), dtype=np.float64)


'''

OLD_TAIL = '''    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
NEW_TAIL = '''    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
            0.001,
            0.999,
        )
    workload_active = active & rcore
    if np.any(workload_active):
        workload_h1 = _predict_workload_h1(
            frame.loc[workload_active].reset_index(drop=True)
        )
        workload_proposal = np.clip(
            0.82 * effective_bridge[workload_active]
            + 0.18 * workload_h1
            + C3_WEIGHT * c3[workload_active],
            0.001,
            0.999,
        )
        workload_delta = workload_proposal - jy_probability[workload_active]
        output[workload_active] = np.clip(
            output[workload_active] + workload_delta,
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def patch_script(source: str) -> str:
    source = source.replace("\r\n", "\n")
    if source.count(FUNCTION_ANCHOR) != 1:
        raise ValueError("workload function insertion anchor is not unique")
    if source.count(OLD_TAIL) != 1:
        raise ValueError("v334 route tail is not unique")
    output = source.replace(FUNCTION_ANCHOR, WORKLOAD_FUNCTIONS + FUNCTION_ANCHOR)
    output = output.replace(OLD_TAIL, NEW_TAIL)
    if output.count("def _predict_workload_h1") != 1:
        raise ValueError("workload runtime insertion failed")
    return output


def run(
    source_zip: Path,
    workload_bundle: Path,
    workload_summary: Path,
    v339_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    finalized = json.loads(workload_summary.read_text(encoding="utf-8"))
    audit = json.loads(v339_summary.read_text(encoding="utf-8"))
    if finalized.get("protocol") != "V342_FINALIZE_WORKLOAD_H1_V1":
        raise ValueError("unexpected workload full-fit protocol")
    if finalized.get("status") != "full_fit_complete":
        raise ValueError("workload H1 full fit is incomplete")
    if audit.get("protocol") != "V339_TRANSITION_WORKLOAD_PORTFOLIO_V335_V1":
        raise ValueError("unexpected v339 audit protocol")
    if not audit.get("candidate_gate_passed"):
        raise ValueError("v339 candidate gate did not pass")
    if sha256(workload_bundle) != str(finalized["output_sha256"]):
        raise ValueError("workload H1 bundle hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v343_transition_workload.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        if WORKLOAD_MEMBER in names:
            raise ValueError("source ZIP already contains workload H1")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as target:
            for item in source.infolist():
                payload = (
                    patched.encode("utf-8")
                    if item.filename == "script.py"
                    else source.read(item.filename)
                )
                target.writestr(item, payload)
            target.write(workload_bundle, WORKLOAD_MEMBER)
    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        members = built.namelist()
    metrics = audit["metrics"]
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": len(members),
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_zip_sha256": sha256(source_zip),
        "workload_bundle_sha256": sha256(workload_bundle),
        "recipe": {
            "base": "v334 = v335 + player transition",
            "new_component": "v209 three-seed workload H1",
            "route": "R_CORE runners-or-high-LI only",
            "absolute_h1_weight": 0.18,
            "increment_formula": "v334 + (v209 candidate - v209 parent)",
        },
        "local_metrics_vs_v335": metrics["portfolio"],
        "component_metrics_vs_v335": {
            "transition": metrics["v338_transition"],
            "workload": metrics["v209_three_seed"],
        },
        "full_2024_robustness": audit["locked_robustness"],
        "selection_warning": audit["selection_warning"],
        "restrictions": {
            **audit["restrictions"],
            "full_fit_used_all_train_through_2024": True,
            "runtime_audit_pending": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--workload-bundle", type=Path, required=True)
    parser.add_argument("--workload-summary", type=Path, required=True)
    parser.add_argument("--v339-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.source_zip,
        args.workload_bundle,
        args.workload_summary,
        args.v339_summary,
        args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
