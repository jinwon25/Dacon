import numpy as np
import torch

from experiments.spatiotemporal_multitask import (
    SpatialTemporalMultiTask,
    competition_loss,
    graph_adjacency,
    group_pooling_weights,
)
from experiments.spatiotemporal_final import hybrid_group3_prediction


def test_graph_and_pooling_weights_are_normalized() -> None:
    coordinates = np.asarray([[37.0, 128.9], [37.1, 129.0], [37.2, 129.1]], dtype=np.float32)
    adjacency = graph_adjacency(coordinates)
    pooling = group_pooling_weights(coordinates)

    np.testing.assert_allclose(adjacency.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(pooling.sum(axis=1), 1.0, atol=1e-6)


def test_spatiotemporal_model_output_shape_and_bounds() -> None:
    ldaps_coordinates = np.asarray([[37.0, 128.9], [37.1, 129.0]], dtype=np.float32)
    gfs_coordinates = np.asarray([[37.0, 128.8], [37.2, 129.1]], dtype=np.float32)
    model = SpatialTemporalMultiTask(
        ldaps_shape=(2, 4),
        gfs_shape=(2, 3),
        hidden=4,
        ldaps_adjacency=graph_adjacency(ldaps_coordinates),
        gfs_adjacency=graph_adjacency(gfs_coordinates),
        ldaps_pooling=group_pooling_weights(ldaps_coordinates),
        gfs_pooling=group_pooling_weights(gfs_coordinates),
        ldaps_mean=np.zeros(4, dtype=np.float32),
        ldaps_std=np.ones(4, dtype=np.float32),
        gfs_mean=np.zeros(3, dtype=np.float32),
        gfs_std=np.ones(3, dtype=np.float32),
    )
    output = model(
        torch.randn(2, 24, 2, 4),
        torch.randn(2, 24, 2, 3),
        torch.randn(2, 24, 5),
    )

    assert output.shape == (2, 24, 3)
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.05)


def test_hybrid_group3_prediction_respects_gate() -> None:
    prediction, member, mask = hybrid_group3_prediction(
        base=np.asarray([5000.0, 5000.0]),
        cross_member=np.asarray([6000.0, 9000.0]),
        spatial_member=np.asarray([5500.0, 8500.0]),
        group_1_ratio=np.asarray([0.3, 0.1]),
        group_2_ratio=np.asarray([0.32, 0.4]),
    )

    assert mask.tolist() == [True, False]
    assert prediction[0] != 5000.0
    assert prediction[1] == 5000.0
    assert member.shape == (2,)


def test_competition_loss_macro_averages_available_groups() -> None:
    # Duplicating eligible rows in group 1 must not change its weight relative
    # to group 2.  Group 3 is absent and is intentionally excluded.
    target = torch.tensor([[[0.50, 0.50, float("nan")], [0.50, float("nan"), float("nan")]]])
    prediction = torch.tensor([[[0.60, 0.70, 0.40], [0.60, 0.20, 0.40]]])

    loss = competition_loss(prediction, target, reward_strength=0.0)

    # Group 1 MAE is 0.10 and group 2 MAE is 0.20, so the macro mean is 0.15.
    torch.testing.assert_close(loss, torch.tensor(0.15), atol=1e-6, rtol=0.0)
