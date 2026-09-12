"""Build the conservative R_ANCHOR low-rank complement above v320."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PROTOCOL = "V335_BUILD_ANCHOR_LOWRANK_PACKAGE_V1"
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
    if source.count(OLD_ROUTE_BLOCK) != 1:
        raise ValueError("v320 route block is not unique")
    output = source.replace(OLD_ROUTE_BLOCK, NEW_ROUTE_BLOCK)
    if output.count("lowrank_probability_delta[anchor]") != 1:
        raise ValueError("anchor runtime insertion failed")
    return output


def run(source_zip: Path, audit_summary: Path, output_dir: Path) -> dict[str, object]:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V335_ANCHOR_LOWRANK_COMPLEMENT_AUDIT_V1":
        raise ValueError("unexpected v335 audit protocol")
    if audit.get("status") != "candidate":
        raise ValueError("v335 audit did not pass")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v335_anchor_lowrank_complement.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        if source.namelist().count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                payload = patched.encode("utf-8") if item.filename == "script.py" else source.read(item.filename)
                target.writestr(item, payload)
    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        member_count = len(built.namelist())
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_runtime_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": member_count,
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_sha256": sha256(source_zip),
        "recipe": audit["recipe"],
        "local_metrics": audit["metrics"],
        "full_2024_robustness": audit["full_2024_robustness"],
        "selection_risk": audit["selection_risk"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.source_zip, args.audit_summary, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
