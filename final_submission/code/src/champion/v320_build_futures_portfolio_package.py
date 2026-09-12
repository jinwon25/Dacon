"""Build a standalone v318/v320 package by extending the preserved v290 ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


PROTOCOL = "V320_BUILD_FUTURES_PORTFOLIO_PACKAGE_V1"
FUNCTION_ANCHOR = '''def _window_adjustment(
'''
LOWRANK_FUNCTION = '''def _predict_futures_lowrank(frame: pd.DataFrame) -> np.ndarray:
    """Map frozen prior-season OOF matrices using only fields in this row."""

    balls = pd.to_numeric(frame["balls_before"], errors="raise").to_numpy(np.int16)
    strikes = pd.to_numeric(frame["strikes_before"], errors="raise").to_numpy(np.int16)
    hand = pd.to_numeric(frame["batter_hand"], errors="raise").to_numpy(np.int16)
    if not (
        np.isin(balls, np.arange(4)).all()
        and np.isin(strikes, np.arange(3)).all()
        and np.isin(hand, (1, 2)).all()
    ):
        raise ValueError("unexpected count or batter-hand value")
    context = ((balls * 3 + strikes) * 2 + (hand - 1)).astype(np.int16)
    pitcher = pd.to_numeric(frame["pitcher_id"], errors="raise").to_numpy(np.int64)
    output = np.zeros(len(frame), dtype=np.float64)
    with np.load(MODEL_DIR / "futures_lowrank" / "lookup.npz", allow_pickle=False) as saved:
        years = saved["source_years"].astype(np.int16)
        for year in years:
            pitcher_ids = saved[f"pitcher_ids_{int(year)}"].astype(np.int64)
            matrix = saved[f"matrix_{int(year)}"].astype(np.float64)
            index = pd.Index(pitcher_ids).get_indexer(pitcher)
            seen = index >= 0
            output[seen] += matrix[index[seen], context[seen]]
    return output / float(len(years))


'''
OLD_FUTURES_BLOCK = '''        output[futures] = np.clip(
            output[futures]
            + 0.10 * (futures_probability[futures] - output[futures]),
            0.001,
            0.999,
        )
'''
NEW_FUTURES_BLOCK = '''        lowrank_probability_delta = _predict_futures_lowrank(frame)
        output[futures] = np.clip(
            output[futures]
            + 0.20 * (futures_probability[futures] - output[futures])
            + 0.50 * lowrank_probability_delta[futures],
            0.001,
            0.999,
        )
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
        raise ValueError("low-rank function insertion anchor is not unique")
    if source.count(OLD_FUTURES_BLOCK) != 1:
        raise ValueError("v290 futures block is not unique")
    output = source.replace(FUNCTION_ANCHOR, LOWRANK_FUNCTION + FUNCTION_ANCHOR)
    output = output.replace(OLD_FUTURES_BLOCK, NEW_FUTURES_BLOCK)
    if output.count("def _predict_futures_lowrank") != 1:
        raise ValueError("low-rank runtime insertion failed")
    return output


def run(
    source_zip: Path,
    lookup_npz: Path,
    v318_summary: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / "submit_v320_futures_portfolio.zip"
    with zipfile.ZipFile(source_zip, "r") as source:
        names = source.namelist()
        if names.count("script.py") != 1:
            raise ValueError("source ZIP must contain one root script.py")
        if "model/futures_lowrank/lookup.npz" in names:
            raise ValueError("source ZIP already contains a low-rank lookup")
        patched = patch_script(source.read("script.py").decode("utf-8"))
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                payload = patched.encode("utf-8") if item.filename == "script.py" else source.read(item.filename)
                target.writestr(item, payload)
            target.write(lookup_npz, "model/futures_lowrank/lookup.npz")
    with zipfile.ZipFile(output_zip, "r") as built:
        bad_crc = built.testzip()
        members = built.namelist()
    audit = json.loads(v318_summary.read_text(encoding="utf-8"))
    if audit.get("protocol") != "V318_FUTURES_DIRECT_LOWRANK_PORTFOLIO_V1":
        raise ValueError("unexpected v318 summary protocol")
    if not audit.get("eligible_for_exploratory_packaging"):
        raise ValueError("v318 is not eligible for exploratory packaging")
    summary = {
        "protocol": PROTOCOL,
        "status": "built_pending_standalone_audit",
        "output_zip": str(output_zip),
        "output_sha256": sha256(output_zip),
        "output_bytes": output_zip.stat().st_size,
        "member_count": len(members),
        "crc_passed": bad_crc is None,
        "source_zip": str(source_zip),
        "source_sha256": sha256(source_zip),
        "lookup_sha256": sha256(lookup_npz),
        "recipe": {
            "recent_futures_probability_weight": 0.20,
            "lowrank_s300_r2_probability_delta_weight": 0.50,
            "route": "F",
        },
        "local_locked_gain_vs_v290": audit["metrics"]["full_2024"]["portfolio"]["gain"],
        "local_locked_full_row_rms_shift": audit["locked_full_row_rms_shift"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--lookup-npz", type=Path, required=True)
    parser.add_argument("--v318-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(
        args.source_zip, args.lookup_npz, args.v318_summary, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
