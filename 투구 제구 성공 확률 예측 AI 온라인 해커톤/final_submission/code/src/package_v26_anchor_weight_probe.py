"""Create the 0819 row-region TrackMan-ASOF gate child of the v26 weight probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from src.package import verify_package
from src.package_v25_postbreak_anchor import _safe_extract


PARENT_SHA256 = "60CF13B28AF01B24A49E2B03E13161F2E04A1C09647D8DA2CFD95407ED9335B1"
SOURCE_ETA = 0.075
PROBE_ETA = 0.150
TMGATE_ETA = 0.030
TMGATE_CAP = 0.030


TMGATE_FUNCTION = r'''

def apply_trackman_asof_gate_overlay(
    probability: np.ndarray,
    frame: pd.DataFrame,
    global_rate: float,
) -> np.ndarray:
    """Tiny R_ANCHOR-only shrink toward ASOF prior gated by TrackMan coverage."""
    profile_path = MODEL_DIR / "trackman_pitcher_profiles.csv"
    if not profile_path.exists():
        return probability
    profile_columns = ["season", "pitcher_id", "tm_linked", "tm_pitcher_n"]
    profile = pd.read_csv(profile_path, usecols=profile_columns)
    source = frame[["season", "pitcher_id"]].copy()
    source["__row_order"] = np.arange(len(source))
    linked = (
        source.merge(
            profile,
            on=["season", "pitcher_id"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
        .sort_values("__row_order", kind="stable")
        .reset_index(drop=True)
    )
    tm_linked = pd.to_numeric(linked["tm_linked"], errors="coerce").fillna(0.0).to_numpy(float)
    tm_n = pd.to_numeric(linked["tm_pitcher_n"], errors="coerce").fillna(0.0).to_numpy(float)
    tm_conf = np.where(tm_linked > 0.0, tm_n / (tm_n + 500.0), 0.0)

    regular = frame["game_type"].astype("string").fillna("__MISSING__").eq("R").to_numpy()
    anchor = (
        frame["pitcher_team_id"].eq(13) | frame["batter_team_id"].eq(13)
    ).to_numpy()
    pressure = np.where(
        (pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).to_numpy() == 3)
        | (pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).to_numpy() == 2),
        1.25,
        0.75,
    )
    gate = np.where(regular & anchor, 0.03 * tm_conf * pressure, 0.0)
    gate = np.clip(gate, 0.0, 0.03)
    if not np.any(gate > 0.0):
        return probability

    prior = hierarchical_prior(frame, global_rate, 200.0, 0.25)
    result = np.asarray(probability, dtype=np.float64).copy()
    result = result * (1.0 - gate) + prior * gate
    return np.clip(result, 1e-6, 1.0 - 1e-6)
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _patch_script_for_tmgate(script_path: Path) -> None:
    script = script_path.read_text(encoding="utf-8")
    if "def apply_trackman_asof_gate_overlay(" not in script:
        marker = "\ndef resolve_trackman_weights(frame: pd.DataFrame, hybrid: dict) -> np.ndarray:"
        if marker not in script:
            raise ValueError("cannot find resolve_trackman_weights insertion marker")
        script = script.replace(marker, TMGATE_FUNCTION + marker, 1)

    call = (
        "    prediction = apply_trackman_asof_gate_overlay(\n"
        "        prediction,\n"
        "        frame,\n"
        "        float(ensemble[\"global_rate\"]),\n"
        "    )\n"
    )
    if call not in script:
        marker = "    prediction = apply_v25_postbreak_anchor_overlay(prediction, frame)\n"
        if marker not in script:
            raise ValueError("cannot find v25 postbreak anchor overlay marker")
        script = script.replace(marker, marker + call, 1)
    script_path.write_text(script, encoding="utf-8")


def build(
    project: Path,
    parent: Path,
    output: Path,
    probe_eta: float = PROBE_ETA,
    expected_parent_sha256: str = PARENT_SHA256,
) -> dict[str, object]:
    project = project.resolve()
    parent = parent.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not SOURCE_ETA < float(probe_eta) <= 0.25:
        raise ValueError("probe eta must be in (0.075, 0.25]")
    if _sha256(parent) != expected_parent_sha256.upper():
        raise ValueError("unexpected v25 parent SHA-256")
    verify_package(parent)

    with tempfile.TemporaryDirectory(prefix="v26_anchor_weight_probe_") as name:
        stage = Path(name)
        with zipfile.ZipFile(parent) as archive:
            _safe_extract(archive, stage)
        spec_path = stage / "model" / "v25_postbreak_anchor_spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if float(spec["blend_eta"]) != SOURCE_ETA:
            raise ValueError("v25 source eta differs from the frozen 7.5% parent")
        spec.update(
            {
                "protocol": "R_ANCHOR_WEIGHT_PROBE_2025_V1",
                "leaderboard_probe_parent": parent.name,
                "leaderboard_probe_parent_sha256": _sha256(parent),
                "blend_eta": float(probe_eta),
                "local_evidence_reference": (
                    "artifacts/v25_postbreak_anchor_audit_20260817_01 and "
                    "the exact quadratic eta sweep recorded in the research report"
                ),
                "selection_note": (
                    f"same row-local v25 signal at eta={float(probe_eta):.6f}; "
                    "Public weight probe explicitly permitted by DACON FAQ "
                    "reply 320256; no test-row aggregation"
                ),
            }
        )
        spec_path.write_text(
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _patch_script_for_tmgate(stage / "script.py")
        hybrid_path = stage / "model" / "hybrid.json"
        hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))
        hybrid.update(
            {
                "candidate": output.stem,
                "variant_parent": parent.name,
                "leaderboard_weight_probe": True,
                "trackman_asof_gate": {
                    "protocol": "0819_JY_TRACKMAN_ASOF_GATE_V1",
                    "eta": TMGATE_ETA,
                    "cap": TMGATE_CAP,
                    "apply_domain": "R_ANCHOR",
                    "confidence_source": "trackman_pitcher_profiles.csv: tm_linked, tm_pitcher_n",
                    "asof_prior": {
                        "alpha": 200.0,
                        "batter_weight": 0.25,
                    },
                    "pressure_scale": {
                        "three_ball_or_two_strike": 1.25,
                        "other": 0.75,
                    },
                    "public_score": 1158.0745556751,
                },
            }
        )
        hybrid_path.write_text(
            json.dumps(hybrid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(
            output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())

    verify_package(output)
    result = {
        "candidate": output.stem,
        "parent": parent.name,
        "parent_sha256": _sha256(parent),
        "source_eta": SOURCE_ETA,
        "probe_eta": float(probe_eta),
        "trackman_asof_gate_eta": TMGATE_ETA,
        "trackman_asof_gate_cap": TMGATE_CAP,
        "public_score": 1158.0745556751,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "official_faq": "https://dacon.io/competitions/official/236743/talkboard/417082#comment_320256",
    }
    manifest_dir = project / "artifacts" / "candidates" / output.stem
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--parent", type=Path, default=Path("submit_v25.zip"))
    parser.add_argument("--output", type=Path, default=Path("0819 row-region_tmgate03.zip"))
    parser.add_argument("--probe-eta", type=float, default=PROBE_ETA)
    parser.add_argument("--expected-parent-sha256", default=PARENT_SHA256)
    args = parser.parse_args()
    project = args.project.resolve()
    parent = args.parent if args.parent.is_absolute() else project / args.parent
    output = args.output if args.output.is_absolute() else project / args.output
    build(project, parent, output, args.probe_eta, args.expected_parent_sha256)


if __name__ == "__main__":
    main()
