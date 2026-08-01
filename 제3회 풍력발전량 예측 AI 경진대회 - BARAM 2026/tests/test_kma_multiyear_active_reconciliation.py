from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.kma_multiyear_active_reconciliation import (
    TARGETS,
    compose_replacement,
    load_pooled_cache,
    period_rows,
)


def _write_cache(path: Path, *, offset: float = 0.0) -> None:
    index = pd.date_range("2024-01-01 01:00", periods=8, freq="91D")
    issue = index - pd.Timedelta(hours=25)
    payload: dict[str, np.ndarray] = {}
    for position, target in enumerate(TARGETS):
        prefix = f"{target}__"
        reference = np.full(len(index), 100.0 + position + offset)
        payload[prefix + "index_ns"] = index.to_numpy(dtype="datetime64[ns]")
        payload[prefix + "issue_ns"] = issue.to_numpy(dtype="datetime64[ns]")
        payload[prefix + "truth"] = reference + 10.0
        payload[prefix + "reference"] = reference
        payload[prefix + "expert"] = reference + 2.0
        payload[prefix + "candidate"] = reference + 1.0
    np.savez_compressed(path, **payload)


def test_load_pooled_cache_preserves_aligned_surfaces(tmp_path: Path) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path)

    loaded = load_pooled_cache(path)

    assert len(loaded["index"]) == 8
    assert loaded["issues"].equals(
        loaded["index"] - pd.Timedelta(hours=25)
    )
    assert np.allclose(
        loaded["targets"]["kpx_group_2"]["candidate"], 102.0
    )


def test_compose_replacement_changes_only_requested_target(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cache.npz"
    _write_cache(path)
    cache = load_pooled_cache(path)
    active = {
        target: np.full(8, 50.0 + position)
        for position, target in enumerate(TARGETS)
    }

    candidate = compose_replacement(
        active, cache, ("kpx_group_3",)
    )

    assert np.array_equal(candidate["kpx_group_1"], active["kpx_group_1"])
    assert np.array_equal(candidate["kpx_group_2"], active["kpx_group_2"])
    assert np.allclose(candidate["kpx_group_3"], 103.0)
    candidate["kpx_group_1"][0] = -1.0
    assert active["kpx_group_1"][0] == 50.0


def test_period_rows_uses_interval_ending_month() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-04-01 00:00",
            "2024-04-01 01:00",
            "2024-07-01 00:00",
            "2024-07-01 01:00",
        ]
    )

    rows = period_rows(index)

    assert rows["q1"].tolist() == [True, False, False, False]
    assert rows["q2"].tolist() == [False, True, True, False]
    assert rows["h2"].tolist() == [False, False, False, True]
    assert rows["full"].all()
