import numpy as np
import pandas as pd

from src.v28_domain_specialist_screen import _v27_parent


def test_v27_parent_only_rescales_existing_anchor_delta() -> None:
    frame = pd.DataFrame(
        {"v22": [0.4, 0.5], "v25": [0.4, 0.515]}
    )
    result = _v27_parent(frame)
    assert np.allclose(result, [0.4, 0.52])
