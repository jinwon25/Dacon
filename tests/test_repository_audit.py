from scripts.audit_repository import forbidden_path_reason, secret_findings


def test_repository_path_policy_allows_documentation_only():
    assert forbidden_path_reason("data/README.md") is None
    assert forbidden_path_reason("artifacts/README.md") is None
    assert forbidden_path_reason(".env.example") is None
    assert forbidden_path_reason("src/model.py") is None


def test_repository_path_policy_rejects_data_models_and_archives():
    assert forbidden_path_reason("data/train.csv") is not None
    assert forbidden_path_reason("model/lgb_model.txt") is not None
    assert forbidden_path_reason("artifacts/oof/prediction.npz") is not None
    assert forbidden_path_reason("submit_candidate.zip") is not None
    assert forbidden_path_reason(".env") is not None


def test_repository_path_policy_allows_explicit_private_lfs_releases():
    assert (
        forbidden_path_reason(
            "artifacts/standalone_champion_1161/standalone_champion_1161.zip"
        )
        is None
    )
    assert (
        forbidden_path_reason("artifacts/oof_champion_1161/v84_full_2024.npz")
        is None
    )
    assert (
        forbidden_path_reason(
            "artifacts/standalone_champion_1162/standalone_champion_1162.zip"
        )
        is None
    )
    assert (
        forbidden_path_reason(
            "artifacts/standalone_champion_1162/standalone_manifest.json"
        )
        is None
    )
    assert forbidden_path_reason("artifacts/oof_champion_1159/oof.npz") is not None


def test_repository_path_policy_allows_only_pinned_delivery_zips():
    assert (
        forbidden_path_reason("submissions/releases/v167/submit_v167.zip") is None
    )
    assert (
        forbidden_path_reason(
            "submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip"
        )
        is None
    )
    assert (
        forbidden_path_reason("submissions/releases/v167/submit_v167_rebuild.zip")
        is not None
    )
    assert (
        forbidden_path_reason("submissions/releases/v168/submit_v168.zip") is not None
    )


def test_repository_path_policy_allows_existing_pinned_v148_oof_bundle():
    assert forbidden_path_reason("artifacts/oof_champion_1170/README.md") is None
    assert forbidden_path_reason("artifacts/oof_champion_1170/manifest.json") is None
    assert (
        forbidden_path_reason("artifacts/oof_champion_1170/v148_full_2024.npz")
        is None
    )
    assert (
        forbidden_path_reason("artifacts/oof_champion_1170/extra_oof.npz")
        is not None
    )


def test_secret_scanner_accepts_blank_template_and_rejects_tokens():
    assert secret_findings(".env.example", b"DACON_API_TOKEN=\n") == []
    github_token = b"gho_" + b"A" * 30
    assert secret_findings("bad.txt", github_token)
    dacon_token = b"DACON_API_TOKEN=" + b"x" * 32
    assert secret_findings("bad.env", dacon_token)
