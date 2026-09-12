from src.champion.v353_build_refit_trackman_pfd_package import PROTOCOL


def test_v353_builder_protocol_is_versioned():
    assert PROTOCOL == "V353_BUILD_REFIT_TRACKMAN_PFD_PACKAGE_V1"
