import pandas as pd

from src.archive.trackman_linkage import (
    link_pitchers,
    main_annual_fingerprints,
    trackman_annual_fingerprints,
)


SPEC = {
    "annual_weight": 0.7,
    "cumulative_weight": 0.3,
    "log_count_weight": 0.15,
    "activity_penalty": 0.06,
    "maximum_distance": 0.2,
    "maximum_assignment_rank": 2,
    "high_confidence_distance": 0.06,
    "medium_confidence_distance": 0.12,
}


def _main_rows() -> pd.DataFrame:
    rows = []
    specifications = {
        10: {2019: (100, 0.60, 0.25, 0.15), 2020: (220, 0.55, 0.25, 0.20)},
        20: {2019: (80, 0.30, 0.60, 0.10), 2020: (170, 0.25, 0.60, 0.15)},
    }
    for pitcher_id, seasons in specifications.items():
        for season, (count, fastball, breaking, offspeed) in seasons.items():
            rows.append(
                {
                    "season": season,
                    "pitcher_id": pitcher_id,
                    "pitcher_hand": 2,
                    "asof_pitcher_pitchmix_n": count,
                    "asof_pitcher_fastball_rate": fastball,
                    "asof_pitcher_breaking_rate": breaking,
                    "asof_pitcher_offspeed_rate": offspeed,
                }
            )
    return pd.DataFrame(rows)


def _trackman_rows() -> pd.DataFrame:
    rows = []
    annual_counts = {
        1000: {2019: (60, 25, 15), 2020: (54, 30, 36)},
        2000: {2019: (24, 48, 8), 2020: (20, 54, 16)},
    }
    for pitcher_id, seasons in annual_counts.items():
        for season, counts in seasons.items():
            for group, count in zip(("fastball", "breaking", "offspeed"), counts):
                rows.extend(
                    {
                        "season": season,
                        "pitcher_trackman_id": pitcher_id,
                        "pitcher_hand": "Right",
                        "pitch_type_group": group,
                    }
                    for _ in range(count)
                )
    return pd.DataFrame(rows)


def test_longitudinal_linkage_recovers_known_pairs():
    main_annual = main_annual_fingerprints(_main_rows())
    trackman_annual, hands = trackman_annual_fingerprints(_trackman_rows())
    linkage = link_pitchers(main_annual, trackman_annual, hands, 2021, SPEC)
    actual = linkage.set_index("pitcher_id")["pitcher_trackman_id"].to_dict()
    assert actual == {10: 1000, 20: 2000}
    assert linkage["tm_linked"].eq(1).all()


def test_linkage_excludes_forecast_season_rows():
    main = _main_rows()
    trackman = _trackman_rows()
    extra = trackman.iloc[[0]].copy()
    extra["season"] = 2021
    extra["pitcher_trackman_id"] = 9999
    trackman = pd.concat([trackman, extra], ignore_index=True)
    main_annual = main_annual_fingerprints(main)
    trackman_annual, hands = trackman_annual_fingerprints(trackman)
    linkage = link_pitchers(main_annual, trackman_annual, hands, 2021, SPEC)
    assert 9999 not in set(linkage["pitcher_trackman_id"])


def test_linkage_resolves_inconsistent_hands_without_duplicate_pitchers():
    main = _main_rows()
    switched = main.iloc[[0]].copy()
    switched["season"] = 2018
    switched["pitcher_hand"] = 1
    main = pd.concat([switched, main], ignore_index=True)
    main_annual = main_annual_fingerprints(main)
    trackman_annual, hands = trackman_annual_fingerprints(_trackman_rows())
    linkage = link_pitchers(main_annual, trackman_annual, hands, 2021, SPEC)
    assert linkage["pitcher_id"].is_unique
    assert linkage.set_index("pitcher_id").loc[10, "pitcher_trackman_id"] == 1000
