import numpy as np

from src.archive import v338_v330_rebase_v335 as v338


def test_rebase_changes_only_player_transition_assignment() -> None:
    v335 = np.array([0.2, 0.4, 0.6])
    parent = np.array([0.1, 0.3, 0.5])
    v330 = np.array([0.1, 0.35, 0.5])
    assignment = np.array(["PARENT", "player_transition", "lowrank_interaction"])
    candidate, active, outside_parity = v338.rebase_candidate(
        v335, parent, v330, assignment
    )
    assert active.tolist() == [False, True, False]
    assert np.allclose(candidate, [0.2, 0.45, 0.6])
    # Protected-route parity is a real-package invariant; this synthetic
    # example deliberately violates it on rows 0 and 2.
    assert outside_parity == 0.1
