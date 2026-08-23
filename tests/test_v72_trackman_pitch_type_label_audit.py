from __future__ import annotations

import numpy as np

from src.archive.v72_trackman_pitch_type_label_audit import label_agreement_rows


def test_label_agreement_separates_other_and_additional_labels() -> None:
    agreement, confusion = label_agreement_rows(
        np.array([2023, 2023, 2024, 2024]),
        np.array([0, -1, 1, 2]),
        np.array(["fastball", "breaking", "breaking", "other"]),
    )
    overall = agreement.loc[agreement["season"].eq("ALL")].iloc[0]
    assert overall["three_class_trackman_rows"] == 3
    assert overall["trackman_additional_labels"] == 1
    assert overall["trackman_other_rows"] == 1
    assert overall["agreement"] == 1.0
    assert confusion.loc[0, "fastball"] == 1
    assert confusion.loc[1, "breaking"] == 1
