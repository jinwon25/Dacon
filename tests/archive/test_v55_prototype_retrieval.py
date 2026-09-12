from __future__ import annotations

import numpy as np
import pandas as pd

from src.archive.v55_prototype_retrieval import (
    NUMERIC_COLUMNS,
    PrototypeBucket,
    PrototypeRetriever,
    RobustScaler,
)


def _frame(rows: int, domain: str = "R_CORE") -> pd.DataFrame:
    data: dict[str, object] = {
        "game_type": ["R"] * rows,
        "pitcher_team_id": [1] * rows,
        "batter_team_id": [2] * rows,
        "balls_before": [0] * rows,
        "strikes_before": [1] * rows,
        "pitcher_hand": [1] * rows,
        "batter_hand": [2] * rows,
    }
    for index, column in enumerate(NUMERIC_COLUMNS):
        data[column] = np.linspace(index, index + 1, rows)
    frame = pd.DataFrame(data)
    if domain == "F":
        frame["game_type"] = "F"
    return frame


def test_retriever_maps_unknown_bucket_to_zero() -> None:
    source = _frame(4)
    scaler = RobustScaler.fit(source)
    features = scaler.transform(source)
    key = "R_CORE\x1f0\x1f1\x1f1\x1f2"
    model = PrototypeRetriever(
        scaler=scaler,
        buckets={
            key: PrototypeBucket(
                centroids=features[:1],
                effects=np.array([0.03]),
                counts=np.array([100]),
            )
        },
        diagnostics={},
    )
    known, seen_known = model.predict(source.iloc[:2])
    unknown, seen_unknown = model.predict(_frame(2, domain="F"))
    assert np.allclose(known, 0.03)
    assert seen_known.all()
    assert np.allclose(unknown, 0.0)
    assert not seen_unknown.any()


def test_retrieval_is_batch_and_order_invariant() -> None:
    source = _frame(6)
    scaler = RobustScaler.fit(source)
    features = scaler.transform(source)
    key = "R_CORE\x1f0\x1f1\x1f1\x1f2"
    model = PrototypeRetriever(
        scaler=scaler,
        buckets={
            key: PrototypeBucket(
                centroids=np.vstack([features[0], features[-1]]),
                effects=np.array([-0.02, 0.04]),
                counts=np.array([50, 50]),
            )
        },
        diagnostics={},
    )
    whole, _ = model.predict(source)
    pieces = np.concatenate(
        [model.predict(source.iloc[:2])[0], model.predict(source.iloc[2:])[0]]
    )
    permutation = np.array([4, 1, 5, 0, 3, 2])
    shuffled, _ = model.predict(source.iloc[permutation])
    assert np.allclose(whole, pieces)
    assert np.allclose(shuffled, whole[permutation])
