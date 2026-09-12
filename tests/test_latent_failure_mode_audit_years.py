import inspect

from src.archive.latent_failure_mode_state_model import run


def test_latent_failure_mode_run_keeps_backward_compatible_default_years():
    default = inspect.signature(run).parameters["audit_years"].default
    assert default == (2023, 2024)
