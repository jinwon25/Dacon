from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features import build_features


def load_or_build_features(
    data_dir: str | Path,
    split: str,
    cache_dir: str | Path = "artifacts_feature_cache",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load a reusable feature frame or build it atomically on first use."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"features_{split}.pkl"
    if cache_path.exists() and not rebuild:
        frame = pd.read_pickle(cache_path)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"Invalid cached index in {cache_path}")
        return frame

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = build_features(data_dir, split)
    temporary_path = cache_path.with_suffix(".tmp.pkl")
    frame.to_pickle(temporary_path)
    temporary_path.replace(cache_path)
    return frame
