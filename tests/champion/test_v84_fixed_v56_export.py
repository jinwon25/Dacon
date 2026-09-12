import numpy as np
import torch

from src.champion.v53_factorization_offset import DROPOUT, FIELDS, PairwiseFM, _predict_raw
from src.champion.v84_fixed_v56_export import numpy_raw


def test_numpy_raw_matches_torch_eval() -> None:
    torch.manual_seed(84)
    cardinalities = [index + 4 for index in range(len(FIELDS))]
    model = PairwiseFM(cardinalities, rank=5, dropout=DROPOUT)
    rng = np.random.default_rng(84)
    code = np.column_stack(
        [rng.integers(0, cardinality, size=37) for cardinality in cardinalities]
    ).astype(np.int64)
    embeddings = [
        layer.weight.detach().cpu().numpy().astype(np.float32)
        for layer in model.embeddings
    ]
    expected = _predict_raw(model, code, batch_size=11)
    actual = numpy_raw(code, embeddings)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-9)


def test_numpy_raw_rejects_invalid_shape() -> None:
    embeddings = [np.zeros((2, 2), dtype=np.float32) for _ in FIELDS]
    try:
        numpy_raw(np.zeros((3, len(FIELDS) - 1), dtype=np.int64), embeddings)
    except ValueError as error:
        assert "invalid shape" in str(error)
    else:
        raise AssertionError("invalid field count must fail")
