"""Check the recovered H1 recipe against the immutable shipped bundle, without fitting."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import joblib

ROOT = Path(__file__).resolve().parent


def audit():
    recipe = json.loads((ROOT / "h1/training_recipe.json").read_text(encoding="utf-8"))
    archive = ROOT / "lineage_inputs/cand_asof_xl.zip"
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != recipe["archive_sha256"]:
        raise ValueError("Unrecognized H1 archive; refusing pickle load")
    with zipfile.ZipFile(archive) as source:
        bundle = joblib.load(io.BytesIO(source.read("model/rf.pkl")))
    assert len(bundle["features"]) == recipe["features"]
    assert bundle["alpha"] == recipe["alpha"]
    assert bundle["center"] == recipe["center"]
    assert len(bundle["models"]) == len(recipe["seeds"])
    reports = []
    for seed, pipeline in zip(recipe["seeds"], bundle["models"]):
        model = pipeline.steps[-1][1]
        params = model.get_params()
        assert type(model).__name__ == "CatBoostClassifier"
        assert params["random_seed"] == seed
        for key in ("iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count", "thread_count"):
            assert params[key] == recipe[key], (seed, key, params[key])
        assert "tags/v1.2.10" in model.get_metadata()["catboost_version_info"]
        reports.append({"seed": seed, "params": params, "training_version": "1.2.10"})
    return {"status": "pass", "scope": "recipe and shipped bundle, NOT fresh model fit",
            "archive_sha256": digest,
            "features": len(bundle["features"]), "models": reports}


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
