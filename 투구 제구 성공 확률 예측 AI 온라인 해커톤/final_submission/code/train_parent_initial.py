"""Fit initial parent models from official data and hash-checked fresh V2 OOF."""
import argparse
import json
from pathlib import Path
import importlib.metadata
import numpy as np
from train import sha256, DATA_HASHES
from src.data import read_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component",choices=("legacy","advanced","corrected","futures_initial"),required=True)
    parser.add_argument("--data-project",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--oof-project",type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new model training directory")
    train_path = args.data_project/"data/train.csv"
    if sha256(train_path) != DATA_HASHES["train.csv"]:
        raise ValueError("Official training data mismatch")
    versions = {k:importlib.metadata.version(k) for k in ("numpy","pandas","lightgbm","catboost")}
    if versions != {"numpy":"2.2.6","pandas":"2.2.3","lightgbm":"4.6.0","catboost":"1.2.8"}:
        raise ValueError("Use the pinned parent-model training environment")
    source_oof_hashes = {}
    if args.component in ("advanced","corrected"):
        if not args.oof_project:
            parser.error("advanced/corrected require --oof-project from train_initial_oof_fresh.py")
        record = json.loads((args.oof_project/"initial_oof_fresh_fit.json").read_text(encoding="utf-8"))
        if (record.get("fresh_training_complete") is not True
                or record.get("train_sha256") != DATA_HASHES["train.csv"]
                or record.get("cached_fits_from_other_runs_used") is not False):
            raise ValueError("Missing verified fresh OOF generation record")
        for year in (2021,2022,2023,2024):
            path = args.oof_project/"artifacts/top1100/v2_r1_oof/fresh"/f"v2_r1_o{year}.npz"
            actual = sha256(path)
            if actual != record["folds"][str(year)]["sha256"]:
                raise ValueError("Fresh OOF file hash mismatch")
            source_oof_hashes[str(year)] = actual
    args.output_dir.mkdir(parents=True)
    model_dir = args.output_dir/"model"
    if args.component == "futures_initial":
        from src.f_regime_catboost import train_final
        config = json.loads((Path(__file__).resolve().parent/"configs/f_regime_catboost.json").read_text(encoding="utf-8"))
        train_final(args.output_dir,read_main(train_path),config)
        model_dir = args.output_dir/"artifacts/f_regime/final"
    elif args.component == "legacy":
        from src.finalize_legacy_cb_axis import run
        run(args.data_project,Path(__file__).resolve().parent/"legacy_state",model_dir)
    else:
        frame = read_main(train_path)
        adapter = args.output_dir/"oof_adapter"
        nested = adapter/"artifacts/top1100/v2_r1_oof/fresh"
        corrected = adapter/"corrected_cb"
        nested.mkdir(parents=True); corrected.mkdir()
        for year in (2021,2022,2023,2024):
            path = args.oof_project/"artifacts/top1100/v2_r1_oof/fresh"/f"v2_r1_o{year}.npz"
            rows = frame.loc[frame["season"].eq(year)]
            with np.load(path,allow_pickle=False) as saved:
                target = saved["target"]
                probability = saved["p_v2_nested"]
                if not np.array_equal(target,rows["control_success"].to_numpy()):
                    raise ValueError("OOF target/official season order mismatch")
                if probability.shape != target.shape or not np.all(np.isfinite(probability)):
                    raise ValueError("Invalid OOF predictions")
            # Use official row IDs, not the object-serialized ID field in old NPZ schemas.
            ids = rows["row_id"].astype(str).to_numpy(dtype="U")
            types = rows["game_type"].astype(str).to_numpy(dtype="U")
            np.savez_compressed(nested/f"v2_r1_o{year}.npz",
                                row_id=ids,target=target,p_v2_nested=probability)
            # The historical advanced trainer reads only V2, not CatBoost predictions.
            np.savez_compressed(corrected/f"corrected_cb_o{year}.npz",
                                row_id=ids,target=target,v2=probability,game_type=types)
        del frame
        if args.component == "advanced":
            from src.finalize_advanced_domain_residual import run
            run(args.data_project,corrected,model_dir)
        else:
            from src.finalize_corrected_residual import run
            run(args.data_project,adapter,model_dir)
    report = {"component":args.component,"full_fit_complete":True,"versions":versions,
              "official_train_sha256":DATA_HASHES["train.csv"],
              "fresh_v2_oof_sha256":source_oof_hashes,
              "saved_model_reused":False,"private_score_recomputed":False,
              "model_manifest":json.loads((model_dir/"manifest.json").read_text(encoding="utf-8"))}
    (args.output_dir/"training.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
