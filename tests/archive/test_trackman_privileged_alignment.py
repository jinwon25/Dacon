import numpy as np
import pandas as pd

from src.archive.trackman_privileged_alignment import (
    align_pitch_rows,
    detect_main_game_ids,
    match_game_sequences,
    structural_tokens,
)


def _main_game(season: int, game: int, dow: int = 5) -> pd.DataFrame:
    states = [
        (1, "T", 0, 0, 0),
        (1, "T", 1, 0, 0),
        (1, "T", 0, 0, 0),  # another PA, not another game
        (1, "T", 0, 1, 0),
        (1, "B", 0, 0, 0),
        (2, "T", 0, 0, 0),
        (2, "B", 0, 0, 0),
    ]
    rows = []
    for inning, half, balls, strikes, outs in states:
        rows.append(
            {
                "season": season,
                "game_month": 3,
                "game_dayofweek": dow,
                "inning": inning,
                "top_bottom": half,
                "balls_before": balls,
                "strikes_before": strikes,
                "outs_before": outs,
                "pitcher_hand": 1,
                "batter_hand": 2,
                "pitcher_id": 10 + game,
                "batter_id": 20 + game,
                "pitcher_team_id": 1,
                "batter_team_id": 2,
                "row_id": f"R{game}_{len(rows)}",
                "game_type": "R",
            }
        )
    return pd.DataFrame(rows)


def _trackman_game(main: pd.DataFrame, game_id: str) -> pd.DataFrame:
    out = main.copy()
    out["top_bottom"] = out["top_bottom"].map({"T": "Top", "B": "Bottom"})
    out["pitcher_hand"] = out["pitcher_hand"].map({1: "Left", 2: "Right"})
    out["batter_hand"] = out["batter_hand"].map({1: "Left", 2: "Right"})
    out["trackman_game_id"] = game_id
    out["game_date"] = "03/23/2019"
    out["pitch_no"] = np.arange(1, len(out) + 1)
    out["trackman_id"] = np.arange(100, 100 + len(out))
    out["pitcher_trackman_id"] = 1000
    out["batter_trackman_id"] = 2000
    return out


def test_game_detection_does_not_split_new_plate_appearance():
    two_games = pd.concat([_main_game(2019, 1), _main_game(2019, 2)], ignore_index=True)
    game_ids = detect_main_game_ids(two_games)
    assert game_ids.max() == 2
    assert np.unique(game_ids[:7]).tolist() == [1]
    assert np.unique(game_ids[7:]).tolist() == [2]


def test_tokens_reject_target_columns():
    frame = _main_game(2019, 1)
    frame["control_success"] = 1
    try:
        structural_tokens(frame)
    except ValueError as error:
        assert "outcome-derived" in str(error)
    else:
        raise AssertionError("target column should have been rejected")


def test_exact_weekday_sequence_matches_without_shift():
    main = _main_game(2019, 1)
    trackman = _trackman_game(main, "G1")
    matches = match_game_sequences(
        main, trackman, minimum_dice=0.9, minimum_margin=0.8, ngram_order=3
    )
    assert len(matches) == 1
    assert bool(matches.iloc[0]["accepted"])
    shifted = trackman.copy()
    shifted["game_dayofweek"] = 6
    no_match = match_game_sequences(
        main, shifted, minimum_dice=0.9, minimum_margin=0.8, ngram_order=3
    )
    assert no_match.empty


def test_sequence_alignment_handles_one_missing_trackman_pitch():
    main = _main_game(2019, 1)
    trackman = _trackman_game(main, "G1").drop(index=[1]).reset_index(drop=True)
    matches = pd.DataFrame(
        [
            {
                "season": 2019,
                "main_game_id": 1,
                "trackman_game_id": "G1",
                "dice": 1.0,
                "margin": 1.0,
                "accepted": True,
            }
        ]
    )
    rows, audit = align_pitch_rows(
        main, trackman, matches, minimum_aligned_fraction=0.8
    )
    assert len(rows) == len(trackman)
    assert audit.iloc[0]["aligned_fraction"] == 1.0


def test_alignment_indices_refer_to_original_trackman_order():
    main = _main_game(2019, 1)
    trackman = _trackman_game(main, "G1").sample(frac=1.0, random_state=7).reset_index(drop=True)
    matches = pd.DataFrame(
        [
            {
                "season": 2019,
                "main_game_id": 1,
                "trackman_game_id": "G1",
                "dice": 1.0,
                "margin": 1.0,
                "accepted": True,
            }
        ]
    )
    rows, _ = align_pitch_rows(main, trackman, matches)
    recovered_pitch_numbers = trackman.iloc[rows["trackman_index"]]["pitch_no"].to_numpy()
    assert recovered_pitch_numbers.tolist() == list(range(1, len(main) + 1))
