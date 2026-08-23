from __future__ import annotations

import numpy as np

from src.archive.v73_trackman_label_pitch_type_student import trackman_training_labels


def test_trackman_training_labels_preserve_unaligned_and_other_as_missing() -> None:
    label = trackman_training_labels(
        5,
        np.array([0, 2, 4]),
        np.array(["fastball", "other", "offspeed"]),
    )
    assert label.tolist() == [0, -1, -1, -1, 2]
