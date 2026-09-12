from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.champion.v53_factorization_offset import (
    FIELDS,
    FieldEncoder,
    PairwiseFM,
    _predict_raw,
    apply_offset,
    centre_by_domain,
    domain_centres,
)


def _categorical_frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            field: [f"{field}_{index % 2}" for index in range(rows)]
            for field in FIELDS
        }
    )


def test_encoder_uses_source_vocabulary_and_maps_unknown_to_zero() -> None:
    source = _categorical_frame()
    encoder = FieldEncoder.fit(source)
    audit = source.iloc[:2].copy()
    audit.loc[1, "pitcher_id"] = "new_pitcher"
    code = encoder.transform(audit)
    assert code[0, 0] != 0
    assert code[1, 0] == 0
    assert len(encoder.vocabularies[0]) == 2


def test_fm_prediction_is_batch_invariant() -> None:
    torch.manual_seed(7)
    model = PairwiseFM([5] * len(FIELDS), rank=4, dropout=0.1)
    code = np.tile(np.arange(len(FIELDS)) % 4 + 1, (13, 1)).astype(np.int64)
    whole = _predict_raw(model, code, batch_size=13)
    split = _predict_raw(model, code, batch_size=4)
    assert np.allclose(whole, split, atol=1e-12)


def test_domain_centring_removes_source_means() -> None:
    values = np.array([1.0, 3.0, -2.0, 2.0])
    domains = np.array(["R", "R", "F", "F"])
    centres = domain_centres(values, domains)
    centred = centre_by_domain(values, domains, centres)
    assert np.isclose(centred[domains == "R"].mean(), 0.0)
    assert np.isclose(centred[domains == "F"].mean(), 0.0)


def test_apply_offset_changes_only_selected_route() -> None:
    frame = pd.DataFrame({"domain3": ["R_CORE", "F", "R_CORE"]})
    parent = np.array([0.4, 0.5, 0.6])
    correction = np.array([0.2, -0.2, 0.1])
    candidate, active = apply_offset(
        frame, parent, correction, "R_CORE", 0.5
    )
    assert np.array_equal(active, np.array([True, False, True]))
    assert candidate[1] == parent[1]
    assert np.all((candidate > 0.0) & (candidate < 1.0))
    assert candidate[0] > parent[0]
