"""Target-free pitch alignment for training-only TrackMan information.

The official tables do not share identifiers, but the training table is in
game/pitch order and both tables contain the pre-pitch count, inning, half and
handedness.  These fields form a strong structural fingerprint without using
``control_success`` or any other outcome-derived value.

The resulting row alignment is a *training artifact*.  Current-pitch TrackMan
measurements must never be used by the inference model.  They may be used for
training-time privileged-information experiments, while prior-season player
profiles derived from the official TrackMan history remain inference-safe.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_COLUMNS = [
    "row_id",
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "game_type",
]
TRACKMAN_COLUMNS = [
    "trackman_id",
    "season",
    "game_date",
    "game_month",
    "game_dayofweek",
    "trackman_game_id",
    "pitch_no",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitcher_trackman_id",
    "batter_trackman_id",
]
FORBIDDEN_ALIGNMENT_COLUMNS = {
    "control_success",
    "target",
    "label",
    "prediction",
    "residual",
}


def detect_main_game_ids(frame: pd.DataFrame) -> np.ndarray:
    """Recover game blocks from the ordered main table.

    A game starts when the stream enters inning 1/top after not being in that
    half-inning.  Looking for every 0-0, 0-out row is incorrect because each
    new plate appearance in the first inning can have that same state.
    """

    inning = pd.to_numeric(frame["inning"], errors="coerce")
    half = frame["top_bottom"].astype(str)
    in_first_top = inning.eq(1) & half.eq("T")
    previous_first_top = inning.shift().eq(1) & half.shift().astype(str).eq("T")
    starts = in_first_top & ~previous_first_top
    starts |= frame["season"].ne(frame["season"].shift())
    if len(starts):
        starts.iloc[0] = True
    return starts.cumsum().to_numpy(np.int32)


def structural_tokens(frame: pd.DataFrame) -> np.ndarray:
    """Encode only shared, pre-pitch structural fields into integer tokens."""

    forbidden = FORBIDDEN_ALIGNMENT_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise ValueError(f"outcome-derived columns reached alignment: {sorted(forbidden)}")
    half = frame["top_bottom"].map({"T": 0, "B": 1, "Top": 0, "Bottom": 1})
    hand = {1: 0, 2: 1, "1": 0, "2": 1, "Left": 0, "Right": 1}
    pitcher_hand = frame["pitcher_hand"].map(hand)
    batter_hand = frame["batter_hand"].map(hand)
    if half.isna().any() or pitcher_hand.isna().any() or batter_hand.isna().any():
        raise ValueError("unknown half-inning or handedness code")
    token = pd.to_numeric(frame["inning"], errors="raise").astype(np.int16) * 2
    token = token + half.astype(np.int16)
    token = token * 4 + pd.to_numeric(frame["outs_before"], errors="raise").astype(np.int16)
    token = token * 4 + pd.to_numeric(frame["balls_before"], errors="raise").astype(np.int16)
    token = token * 3 + pd.to_numeric(frame["strikes_before"], errors="raise").astype(np.int16)
    token = token * 3 + pitcher_hand.astype(np.int16)
    token = token * 3 + batter_hand.astype(np.int16)
    return token.to_numpy(np.int32)


def _ngrams(tokens: np.ndarray, order: int = 5) -> set[tuple[int, ...]]:
    if len(tokens) < order:
        return set()
    return set(zip(*(tokens[i : len(tokens) - order + i + 1] for i in range(order))))


def _dice(left: set[tuple[int, ...]], right: set[tuple[int, ...]]) -> float:
    return 2.0 * len(left.intersection(right)) / max(1, len(left) + len(right))


def match_game_sequences(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    *,
    minimum_dice: float = 0.95,
    minimum_margin: float = 0.90,
    ngram_order: int = 5,
) -> pd.DataFrame:
    """Match games within the exact official season/month/day-of-week bucket."""

    main = main.copy()
    trackman = trackman.copy()
    main["__game_id"] = detect_main_game_ids(main)
    main["__token"] = structural_tokens(main)
    trackman["__token"] = structural_tokens(trackman)
    trackman = trackman.sort_values(
        ["season", "trackman_game_id", "pitch_no"], kind="stable"
    )

    main_games: list[dict[str, object]] = []
    for (season, game_id), group in main.groupby(
        ["season", "__game_id"], sort=False, observed=True
    ):
        tokens = group["__token"].to_numpy(np.int32)
        main_games.append(
            {
                "season": int(season),
                "main_game_id": int(game_id),
                "month": int(group["game_month"].iloc[0]),
                "dow": int(group["game_dayofweek"].iloc[0]),
                "main_rows": int(len(group)),
                "grams": _ngrams(tokens, ngram_order),
            }
        )
    trackman_by_bucket: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
    for (season, game_id), group in trackman.groupby(
        ["season", "trackman_game_id"], sort=False, observed=True
    ):
        tokens = group["__token"].to_numpy(np.int32)
        item = {
            "season": int(season),
            "trackman_game_id": str(game_id),
            "game_date": str(group["game_date"].iloc[0]),
            "trackman_rows": int(len(group)),
            "grams": _ngrams(tokens, ngram_order),
        }
        key = (
            int(season),
            int(group["game_month"].iloc[0]),
            int(group["game_dayofweek"].iloc[0]),
        )
        trackman_by_bucket[key].append(item)

    output: list[dict[str, object]] = []
    for game in main_games:
        key = (int(game["season"]), int(game["month"]), int(game["dow"]))
        scored = sorted(
            (
                (_dice(game["grams"], candidate["grams"]), candidate)
                for candidate in trackman_by_bucket.get(key, [])
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if not scored:
            continue
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        output.append(
            {
                "season": int(game["season"]),
                "main_game_id": int(game["main_game_id"]),
                "trackman_game_id": str(best["trackman_game_id"]),
                "game_date": str(best["game_date"]),
                "main_rows": int(game["main_rows"]),
                "trackman_rows": int(best["trackman_rows"]),
                "dice": float(best_score),
                "margin": float(best_score - second_score),
            }
        )
    matches = pd.DataFrame(output)
    if matches.empty:
        return matches
    matches["accepted"] = matches["dice"].ge(minimum_dice) & matches["margin"].ge(
        minimum_margin
    )
    duplicate = matches.loc[matches["accepted"]].duplicated(
        ["season", "trackman_game_id"], keep=False
    )
    if duplicate.any():
        duplicate_index = matches.loc[matches["accepted"]].index[duplicate]
        matches.loc[duplicate_index, "accepted"] = False
    return matches


def align_pitch_rows(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    minimum_aligned_fraction: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align accepted game sequences and keep only structurally equal rows."""

    main = main.copy()
    trackman_original = trackman.copy()
    trackman = trackman.copy()
    main["__main_index"] = np.arange(len(main), dtype=np.int64)
    trackman["__trackman_index"] = np.arange(len(trackman), dtype=np.int64)
    main["__game_id"] = detect_main_game_ids(main)
    main["__token"] = structural_tokens(main)
    trackman["__token"] = structural_tokens(trackman)
    trackman = trackman.sort_values(
        ["season", "trackman_game_id", "pitch_no"], kind="stable"
    )
    main_groups = {
        (int(key[0]), int(key[1])): group
        for key, group in main.groupby(["season", "__game_id"], sort=False, observed=True)
    }
    trackman_groups = {
        (int(key[0]), str(key[1])): group
        for key, group in trackman.groupby(
            ["season", "trackman_game_id"], sort=False, observed=True
        )
    }
    row_chunks: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for match in matches.loc[matches["accepted"]].itertuples(index=False):
        left = main_groups[(int(match.season), int(match.main_game_id))]
        right = trackman_groups[(int(match.season), str(match.trackman_game_id))]
        left_tokens = left["__token"].to_numpy(np.int32)
        right_tokens = right["__token"].to_numpy(np.int32)
        blocks = SequenceMatcher(
            None, left_tokens.tolist(), right_tokens.tolist(), autojunk=False
        ).get_matching_blocks()
        pairs: list[tuple[int, int]] = []
        for block in blocks:
            pairs.extend(
                (int(left.iloc[block.a + offset]["__main_index"]),
                 int(right.iloc[block.b + offset]["__trackman_index"]))
                for offset in range(block.size)
            )
        aligned_fraction = len(pairs) / max(1, min(len(left), len(right)))
        audits.append(
            {
                "season": int(match.season),
                "main_game_id": int(match.main_game_id),
                "trackman_game_id": str(match.trackman_game_id),
                "aligned_rows": int(len(pairs)),
                "aligned_fraction": float(aligned_fraction),
            }
        )
        if aligned_fraction < minimum_aligned_fraction:
            continue
        chunk = pd.DataFrame(pairs, columns=["main_index", "trackman_index"])
        chunk["season"] = int(match.season)
        chunk["main_game_id"] = int(match.main_game_id)
        chunk["trackman_game_id"] = str(match.trackman_game_id)
        chunk["game_dice"] = float(match.dice)
        chunk["game_margin"] = float(match.margin)
        row_chunks.append(chunk)
    rows = pd.concat(row_chunks, ignore_index=True) if row_chunks else pd.DataFrame()
    audit = pd.DataFrame(audits)
    if not rows.empty:
        if rows["main_index"].duplicated().any() or rows["trackman_index"].duplicated().any():
            raise AssertionError("pitch alignment is not one-to-one")
        left_token = structural_tokens(main.iloc[rows["main_index"].to_numpy()])
        right_token = structural_tokens(
            trackman_original.iloc[rows["trackman_index"].to_numpy()]
        )
        if not np.array_equal(left_token, right_token):
            raise AssertionError("aligned rows do not share structural tokens")
    return rows, audit


def derive_entity_map(
    rows: pd.DataFrame,
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    main_column: str,
    trackman_column: str,
) -> pd.DataFrame:
    paired = pd.DataFrame(
        {
            main_column: main.iloc[rows["main_index"].to_numpy()][main_column].to_numpy(),
            trackman_column: trackman.iloc[rows["trackman_index"].to_numpy()][trackman_column].to_numpy(),
        }
    )
    counts = paired.groupby([main_column, trackman_column], observed=True).size().rename("support").reset_index()
    totals = counts.groupby(main_column, observed=True)["support"].sum().rename("total_support")
    best = counts.sort_values("support").groupby(main_column, observed=True).tail(1).set_index(main_column)
    best = best.join(totals)
    best["dominance"] = best["support"] / best["total_support"]
    return best.reset_index().sort_values(main_column).reset_index(drop=True)


def run(project: Path, output_dir: Path) -> dict[str, object]:
    project = project.resolve()
    output_dir = (project / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(project / "data" / "train.csv", usecols=MAIN_COLUMNS, low_memory=False)
    trackman = pd.read_csv(
        project / "data" / "trackman_history.csv",
        usecols=TRACKMAN_COLUMNS,
        low_memory=False,
    )
    matches = match_game_sequences(main, trackman)
    rows, row_audit = align_pitch_rows(main, trackman, matches)
    pitcher_map = derive_entity_map(
        rows, main, trackman, "pitcher_id", "pitcher_trackman_id"
    )
    batter_map = derive_entity_map(
        rows, main, trackman, "batter_id", "batter_trackman_id"
    )
    np.savez_compressed(
        output_dir / "pitch_alignment.npz",
        main_index=rows["main_index"].to_numpy(np.int64),
        trackman_index=rows["trackman_index"].to_numpy(np.int64),
        season=rows["season"].to_numpy(np.int16),
        main_game_id=rows["main_game_id"].to_numpy(np.int32),
        game_dice=rows["game_dice"].to_numpy(np.float32),
        game_margin=rows["game_margin"].to_numpy(np.float32),
    )
    matches.to_csv(output_dir / "game_matches.csv", index=False)
    row_audit.to_csv(output_dir / "pitch_alignment_audit.csv", index=False)
    pitcher_map.to_csv(output_dir / "pitcher_map.csv", index=False)
    batter_map.to_csv(output_dir / "batter_map.csv", index=False)
    by_season = (
        rows.groupby("season", observed=True).size().rename("aligned_rows").to_frame()
        .join(main.groupby("season", observed=True).size().rename("all_rows"))
    )
    by_season["coverage"] = by_season["aligned_rows"] / by_season["all_rows"]
    result: dict[str, object] = {
        "protocol": "TARGET_FREE_TRACKMAN_SEQUENCE_ALIGNMENT_V1",
        "main_games": int(detect_main_game_ids(main).max()),
        "trackman_games": int(trackman["trackman_game_id"].nunique()),
        "accepted_games": int(matches["accepted"].sum()),
        "aligned_games": int(row_audit["aligned_fraction"].ge(0.95).sum()),
        "aligned_rows": int(len(rows)),
        "row_coverage": float(len(rows) / len(main)),
        "by_season": by_season.reset_index().to_dict(orient="records"),
        "pitcher_entities": int(len(pitcher_map)),
        "pitcher_dominance_ge_099": int(pitcher_map["dominance"].ge(0.99).sum()),
        "batter_entities": int(len(batter_map)),
        "batter_dominance_ge_099": int(batter_map["dominance"].ge(0.99).sum()),
        "alignment_columns": sorted(set(MAIN_COLUMNS + TRACKMAN_COLUMNS)),
        "forbidden_columns_used": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Training-only TrackMan sequence alignment",
        "",
        "- Linkage uses only shared pre-pitch structure: season/month/day-of-week, inning/half, count/outs, and handedness.",
        "- `control_success`, predictions, residuals, and target-derived statistics are excluded by contract.",
        f"- Accepted/aligned games: **{result['accepted_games']:,} / {result['aligned_games']:,}**.",
        f"- Aligned training rows: **{result['aligned_rows']:,} ({result['row_coverage']:.2%})**.",
        f"- High-dominance (>=99%) pitcher/batter maps: **{result['pitcher_dominance_ge_099']:,} / {result['batter_dominance_ge_099']:,}**.",
        "- This artifact does not authorize current-pitch TrackMan inputs at inference; it is for privileged-information training and historical profile construction only.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/trackman_privileged_20260816")
    )
    args = parser.parse_args()
    run(args.project, args.output_dir)


if __name__ == "__main__":
    main()
