from __future__ import annotations

import json

import numpy as np

from src.archive.v178_row_region_signed_stack_rebase import apply_direction, load_weights


def test_load_weights_reads_frozen_net_weights(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "V165_SIGNED_GROUP_CONSTRAINED_STACK_V1",
                "selected_on_sources_only": {
                    "net_weights": json.dumps({"a": 0.2, "b": -0.1})
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_weights(path) == {"a": 0.2, "b": -0.1}


def test_apply_direction_is_additive_and_clipped() -> None:
    output = apply_direction(
        np.asarray([0.5, 0.99]), np.asarray([0.2, 0.2]), 0.5
    )
    assert np.allclose(output, [0.6, 0.999])
