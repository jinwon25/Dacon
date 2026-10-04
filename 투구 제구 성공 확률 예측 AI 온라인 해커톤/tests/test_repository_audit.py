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


def test_repository_path_policy_rejects_dacon_derived_artifacts():
    """The repository is public, so nothing derived from the provided data is tracked.

    OOF bundles carry the train target and player ids, the delivery archives
    embed per-player count/success priors, and the linkage tables reconstruct
    the pitcher and game id mapping.
    """
    assert (
        forbidden_path_reason("artifacts/oof_champion_1161/v84_full_2024.npz")
        is not None
    )
    assert (
        forbidden_path_reason("artifacts/oof_champion_1170/v148_full_2024.npz")
        is not None
    )
    assert (
        forbidden_path_reason(
            "artifacts/standalone_champion_1161/standalone_champion_1161.zip"
        )
        is not None
    )
    assert (
        forbidden_path_reason("submissions/releases/v167/submit_v167.zip") is not None
    )
    assert (
        forbidden_path_reason("submissions/releases/v345/submit_v345.zip") is not None
    )
    assert (
        forbidden_path_reason(
            "src/team_assets/JY_fallback_XGB_active50_w030/fallback_lookups.joblib"
        )
        is not None
    )


def test_repository_path_policy_still_allows_artifact_manifests():
    """Manifests record what the private artifacts are; they hold no data."""
    assert forbidden_path_reason("artifacts/oof_champion_1161/README.md") is None
    assert forbidden_path_reason("artifacts/oof_champion_1161/manifest.json") is None
    assert forbidden_path_reason("artifacts/oof_champion_1170/manifest.json") is None
    assert (
        forbidden_path_reason(
            "artifacts/standalone_champion_1162/standalone_manifest.json"
        )
        is None
    )


def test_repository_audit_catches_json_model_and_linkage_tables():
    assert forbidden_path_reason("src/team_assets/fallback_xgb.json") is not None
    assert forbidden_path_reason("research/reports/player_mapping_by_origin.csv") is not None
    assert forbidden_path_reason("research/reports/alignment_game_candidates.csv") is not None
    assert secret_findings("unknown.json", b'{"learner":{"gradient_booster":{}}}')


def test_secret_scanner_accepts_blank_template_and_rejects_tokens():
    assert secret_findings(".env.example", b"DACON_API_TOKEN=\n") == []
    github_token = b"gho_" + b"A" * 30
    assert secret_findings("bad.txt", github_token)
    dacon_token = b"DACON_API_TOKEN=" + b"x" * 32
    assert secret_findings("bad.env", dacon_token)
