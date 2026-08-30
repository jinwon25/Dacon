"""Build a standalone aggressive R-route MoE backup above v320."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PROTOCOL = "V334_BUILD_AGGRESSIVE_MOE_PACKAGE_V1"
FUNCTION_ANCHOR = "def _window_adjustment(\n"
TRANSITION_FUNCTION = '''def _predict_player_transition(frame: pd.DataFrame) -> np.ndarray:
    """Frozen 2025 pitcher transition/count correction; each row maps alone."""

    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    current_team = pd.to_numeric(
        frame["pitcher_team_id"], errors="raise"
    ).to_numpy(np.int64)
    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int16)
    game_type = frame["game_type"].astype(str).to_numpy()
    with np.load(
        MODEL_DIR / "player_transition" / "lookup.npz", allow_pickle=False
    ) as saved:
        known_pitchers = saved["pitcher_ids"].astype(np.int64)
        index = pd.Index(known_pitchers).get_indexer(pitcher)
        seen = index >= 0
        last_year = np.full(len(frame), -1, dtype=np.int16)
        previous_team = np.full(len(frame), -1, dtype=np.int64)
        last_year[seen] = saved["last_year"].astype(np.int16)[index[seen]]
        previous_team[seen] = saved["previous_team"].astype(np.int64)[index[seen]]
        status = np.full(len(frame), "SWITCH", dtype="<U6")
        status[~seen] = "NEW"
        status[seen & (last_year < 2024)] = "RETURN"
        status[seen & (last_year == 2024) & (previous_team == current_team)] = "SAME"
        count = np.char.add(
            np.char.add(balls.astype(str), "-"), strikes.astype(str)
        )
        query_key = np.char.add(
            np.char.add(np.char.add(np.char.add(status, "|"), count), "|"),
            game_type,
        )
        correction = pd.Series(
            saved["corrections"].astype(np.float64),
            index=saved["correction_keys"].astype(str),
        ).reindex(query_key).fillna(0.0).to_numpy(np.float64)
    return correction


'''
OLD_ROUTE_BLOCK = '''    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
    return parent, h1, c3, active, output
'''
NEW_ROUTE_BLOCK = '''    futures = frame["game_type"].astype(str).eq("F").to_numpy()
    lowrank_probability_delta = None
    if np.any(futures):
        futures_probability = _predict_recent_futures(frame)
        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
    regular = frame["game_type"].astype(str).eq("R").to_numpy()
    anchor = regular & (
        frame["pitcher_team_id"].astype("int64").eq(13).to_numpy()
        | frame["batter_team_id"].astype("int64").eq(13).to_numpy()
    )
    if np.any(anchor):
        if lowrank_probability_delta is None:
            lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[anchor] = np.clip(
            output[anchor] + 0.50 * lowrank_probability_delta[anchor],
            0.001,
            0.999,
        )
    rcore = regular & ~anchor
    if np.any(rcore):
        transition_delta = _predict_player_transition(frame)
        output[rcore] = np.clip(
            output[rcore] + 0.25 * transition_delta[rcore],
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
        raise ValueError("transition function insertion anchor is not unique")
    if source.count(OLD_ROUTE_BLOCK) != 1:
        raise ValueError("v320 route block is not unique")
    output = source.replace(FUNCTION_ANCHOR, TRANSITION_FUNCTION + FUNCTION_ANCHOR)
    output = output.replace(OLD_ROUTE_BLOCK, NEW_ROUTE_BLOCK)
    if output.count("def _predict_player_transition") != 1:
        raise ValueError("transition runtime insertion failed")
    return output


def run(
    source_zip: Path,
    transition_lookup: Path,
    transition_summary: Path,
    v330_summary: Path,
    output_dir: Path,
) -> dict[str, object]:
    transition_audit = json.loads(transition_summary.read_text(encoding="utf-8"))
    router_audit = json.loads(v330_summary.read_text(encoding="utf-8"))
    if transition_audit.get("protocol") != "V334_FINALIZE_PLAYER_TRANSITION_V1":
        raise ValueError("unexpected transition summary protocol")
    if router_audit.get("protocol") != "V330_LEAVE_ONE_ORIGIN_BASEBALL_MOE_V1":
        raise ValueError("unexpected router summary protocol")
    expected_router = {
        "R_ANCHOR": "lowrank_interaction", "R_CORE": "player_transition"
    }
    if router_audit.get("final_router") != expected_router:
        raise ValueError("v330 router no longer matches the frozen runtime recipe")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v334_aggressive_training_moe.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        member = "model/player_transition/lookup.npz"
        if member in names:
            raise ValueError("source ZIP already contains a transition lookup")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                payload = patched.encode("utf-8") if item.filename == "script.py" else source.read(item.filename)
                target.writestr(item, payload)
            target.write(transition_lookup, member)
    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        members = built.namelist()
    metrics = router_audit["metrics"]
    robustness = router_audit["full_2024_training_axis_robustness"]
    summary = {
        "protocol": PROTOCOL,
        "status": "built_aggressive_training_axis_backup_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": len(members),
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_sha256": sha256(source_zip),
        "transition_lookup_sha256": sha256(transition_lookup),
        "recipe": {
            "base": "v320",
            "F": "unchanged v320 direct+lowrank portfolio",
            "R_ANCHOR": "0.50 * finalized lowrank interaction",
            "R_CORE": "0.25 * pitcher_status_count(alpha=1000) residual",
        },
        "training_axis_gains_vs_v320": {
            name: metrics[name]["gain"] for name in ("full_2022", "late_2023", "full_2024")
        },
        "training_axis_positive_month_fraction": {
            name: metrics[name]["positive_month_fraction"]
            for name in ("full_2022", "late_2023", "full_2024")
        },
        "full_2024_robustness": robustness,
        "risk": {
            "leave_one_origin_2024_gain": router_audit["selected_schema"]["loo_metrics"]["full_2024"]["gain"],
            "cross_origin_gate_passed": router_audit["cross_origin_gate_passed"],
            "promotion_gate_passed": router_audit["promotion_gate_passed"],
            "classification": "aggressive_diversity_backup_not_strict_champion",
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
    parser.add_argument("--transition-lookup", type=Path, required=True)
    parser.add_argument("--transition-summary", type=Path, required=True)
    parser.add_argument("--v330-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.source_zip, args.transition_lookup, args.transition_summary,
        args.v330_summary, args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
