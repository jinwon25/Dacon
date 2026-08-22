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


def test_repository_path_policy_allows_only_current_private_lfs_release():
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
    assert forbidden_path_reason("artifacts/oof_champion_1159/oof.npz") is not None


def test_secret_scanner_accepts_blank_template_and_rejects_tokens():
    assert secret_findings(".env.example", b"DACON_API_TOKEN=\n") == []
    github_token = b"gho_" + b"A" * 30
    assert secret_findings("bad.txt", github_token)
    dacon_token = b"DACON_API_TOKEN=" + b"x" * 32
    assert secret_findings("bad.env", dacon_token)
