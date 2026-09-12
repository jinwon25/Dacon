from __future__ import annotations

import pandas as pd

from src.archive.v66_direct_trackman_gate import (
    attach_legacy_id,
    build_direct_profiles,
    combine_profiles,
)


def test_build_direct_profiles_excludes_origin_season() -> None:
    pairs = pd.DataFrame(
        {
            "season": [2021, 2021, 2022, 2022],
            "pitcher_id": [10, 10, 10, 20],
            "pitcher_trackman_id": [100, 100, 999, 200],
        }
    )
    trackman = pd.DataFrame(
        {
            "season": [2020, 2021, 2022, 2022],
            "pitcher_trackman_id": [100, 100, 999, 200],
        }
    )
    profile, _ = build_direct_profiles(
        pairs,
        trackman,
        (2022,),
        minimum_support=2,
        minimum_purity=0.99,
    )
    assert profile["pitcher_id"].tolist() == [10]
    assert profile["pitcher_trackman_id"].tolist() == [100]
    assert profile["tm_pitcher_n"].tolist() == [2]


def test_direct_first_overrides_disagreeing_legacy_and_falls_back() -> None:
    direct = pd.DataFrame(
        {
            "season": [2025],
            "pitcher_id": [1],
            "pitcher_trackman_id": [101],
            "tm_linked": [1],
            "tm_pitcher_n": [500],
        }
    )
    legacy = pd.DataFrame(
        {
            "season": [2025, 2025],
            "pitcher_id": [1, 2],
            "pitcher_trackman_id": [999, 202],
            "tm_linked": [1, 1],
            "tm_pitcher_n": [900, 300],
        }
    )
    output = combine_profiles(direct, legacy, "direct_first").set_index("pitcher_id")
    assert output.loc[1, "pitcher_trackman_id"] == 101
    assert output.loc[1, "tm_pitcher_n"] == 500
    assert output.loc[2, "pitcher_trackman_id"] == 202
    assert output.loc[2, "tm_pitcher_n"] == 300


def test_agreement_only_rejects_disagreement() -> None:
    direct = pd.DataFrame(
        {
            "season": [2025, 2025],
            "pitcher_id": [1, 2],
            "pitcher_trackman_id": [101, 202],
            "tm_linked": [1, 1],
            "tm_pitcher_n": [500, 300],
        }
    )
    legacy = pd.DataFrame(
        {
            "season": [2025, 2025],
            "pitcher_id": [1, 2],
            "pitcher_trackman_id": [999, 202],
            "tm_linked": [1, 1],
            "tm_pitcher_n": [900, 300],
        }
    )
    output = combine_profiles(direct, legacy, "agreement_only")
    assert output["pitcher_id"].tolist() == [2]
    assert output["pitcher_trackman_id"].tolist() == [202]


def test_attach_legacy_id_preserves_compact_gate_fields() -> None:
    profile = pd.DataFrame(
        {
            "season": [2025],
            "pitcher_id": [1],
            "tm_linked": [1],
            "tm_pitcher_n": [700],
        }
    )
    linkage = pd.DataFrame(
        {
            "season": [2025],
            "pitcher_id": [1],
            "pitcher_trackman_id": [101],
        }
    )
    output = attach_legacy_id(profile, linkage)
    assert output.loc[0, "pitcher_trackman_id"] == 101
    assert output.loc[0, "tm_pitcher_n"] == 700
