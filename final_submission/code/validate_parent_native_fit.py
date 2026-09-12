"""Compare native parent-model fits and train-derived tables; final ensemble doses remain separate."""
import argparse
import io
import json
import math
from pathlib import Path
import tempfile
import zipfile
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator
from train import sha256, FINAL_SHA256
from validate_strict_fresh_fit import compare_native

PREFIX = "model/v124/model/parent/parent/champion/"
COMPONENTS = {
    "futures_initial": ("f_catboost_seed42.cbm", "f_catboost_feature_spec.json"),
    "legacy": ("legacy_cb_axis.cbm", "legacy_cb_axis_spec.json",
               "legacy_cb_pitcher_prior.csv", "legacy_cb_batter_prior.csv"),
    "advanced": ("advanced_domain_residual_lgb.txt", "advanced_domain_residual_spec.json"),
    "corrected": ("corrected_state_residual_lgb.txt", "corrected_state_residual_spec.json",
                  "corrected_state_pitcher_prior.csv", "corrected_state_batter_prior.csv"),
    "base": ("lgb_model.txt", "feature_spec.json", "rf_model.joblib"),
    "trackman": ("trackman_lgb_model.txt", "trackman_feature_spec.json",
                 "trackman_pitcher_profiles.csv"),
    "recency_rf": ("rf_recency_h1.joblib",),
    "recent_exact": ("recent_exact_lgb.txt", "recent_exact_ridge.joblib",
                     "recent_stable_ridge.joblib", "recent_exact_preprocess.joblib",
                     "recent_exact_spec.json"),
    "v14_refinement": ("v14_anchor_ridge.joblib", "v14_f_trend_lgb.txt",
                       "v14_refinement_preprocess.joblib",
                       "v14_refinement_spec.json"),
    "v25_anchor": ("v25_postbreak_anchor.joblib",
                   "v25_postbreak_anchor_spec.json"),
    "joint_state_mode": (
        "joint_selected_h05_l15_b075.txt",
        "joint_global_h4_l15_b075.txt",
        "joint_global_h05_l31_b075.txt",
        "joint_global_h05_l15_b100.txt",
        "joint_global_h05_l15_b050.txt",
        "joint_domain_h05_l15_b075_R_CORE.txt",
        "joint_domain_h05_l15_b075_R_ANCHOR.txt",
        "joint_domain_h05_l15_b075_F.txt",
        "joint_mode_classifier_h0.5.txt",
        "joint_mode_classifier_h2.txt",
        "joint_mode_classifier_h4.txt",
        "joint_mode_outcome.txt",
        "joint_state_mode_preprocess.joblib",
        "joint_state_mode_spec.json",
    ),
    "v20_pfd": ("v20_pfd_control.txt", "v20_pfd_soft_l050.txt",
                "v20_target1160_spec.json"),
    "v21_eb": ("v21_context_state_eb_spec.json",),
    "v22_low_variance": ("v22_low_variance_spec.json",),
}


def _rf_native(model):
    pre = model.named_steps["pre"]
    forest = model.named_steps["clf"]
    cat = pre.named_transformers_["cat"]
    numeric = pre.named_transformers_["num"]
    return {
        "params": {
            key: value for key, value in forest.get_params().items()
            if key != "n_jobs"
        },
        "cat_categories": [np.asarray(values).tolist() for values in cat.categories_],
        "numeric_statistics": np.asarray(numeric.statistics_).tolist(),
        "classes": np.asarray(forest.classes_).tolist(),
        "trees": [
            {
                "children_left": tree.tree_.children_left.tolist(),
                "children_right": tree.tree_.children_right.tolist(),
                "feature": tree.tree_.feature.tolist(),
                "threshold": tree.tree_.threshold.tolist(),
                "impurity": tree.tree_.impurity.tolist(),
                "n_node_samples": tree.tree_.n_node_samples.tolist(),
                "weighted_n_node_samples": tree.tree_.weighted_n_node_samples.tolist(),
                "value": tree.tree_.value.tolist(),
            }
            for tree in forest.estimators_
        ],
    }


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, pd.Series):
        return {
            "name": _plain(value.name),
            "index": _plain(value.index.tolist()),
            "values": _plain(value.tolist()),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "columns": _plain(value.columns.tolist()),
            "index": _plain(value.index.tolist()),
            "data": _plain(value.to_numpy().tolist()),
        }
    if isinstance(value, BSpline):
        return {
            "class": "BSpline",
            "t": _plain(value.t),
            "c": _plain(value.c),
            "k": int(value.k),
            "extrapolate": _plain(value.extrapolate),
            "axis": int(value.axis),
        }
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, BaseEstimator):
        return {"class": type(value).__name__, "state": _plain(value.__dict__)}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component",choices=COMPONENTS,required=True)
    for key in ("reference-zip","candidate-dir","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    if sha256(args.reference_zip) != FINAL_SHA256:
        raise ValueError("Unrecognized reference ZIP")
    reports, dose_changes, assembly_annotations, serialization_defaults = {}, {}, {}, {}
    with zipfile.ZipFile(args.reference_zip) as archive, tempfile.TemporaryDirectory(prefix="parent-native-") as temporary:
        temp = Path(temporary)
        for name in COMPONENTS[args.component]:
            original = archive.read(PREFIX+name)
            candidate = args.candidate_dir/name
            if name.endswith(".cbm"):
                (temp/"old.cbm").write_bytes(original)
                old, new = CatBoostClassifier(), CatBoostClassifier()
                old.load_model(str(temp/"old.cbm"))
                new.load_model(str(candidate))
                old.save_model(str(temp/"old.json"),format="json")
                new.save_model(str(temp/"new.json"),format="json")
                a,b = (json.loads((temp/n).read_text(encoding="utf-8")) for n in ("old.json","new.json"))
                # Ignore training timestamps/GUIDs, not any predictive model section.
                a.pop("model_info",None); b.pop("model_info",None)
            elif name.endswith(".txt"):
                original_text = original.decode("utf-8").replace("\r\r\n","\n").replace("\r\n","\n")
                a = lgb.Booster(model_str=original_text).dump_model()
                b = lgb.Booster(model_str=candidate.read_text(encoding="utf-8")).dump_model()
            elif name.endswith(".csv"):
                old_frame = pd.read_csv(io.BytesIO(original)).astype(object)
                new_frame = pd.read_csv(candidate).astype(object)
                a = old_frame.where(pd.notna(old_frame),None).to_dict("split")
                b = new_frame.where(pd.notna(new_frame),None).to_dict("split")
            elif name.endswith(".joblib"):
                (temp / "old.joblib").write_bytes(original)
                old_object = joblib.load(temp / "old.joblib")
                new_object = joblib.load(candidate)
                if (
                    hasattr(old_object, "named_steps")
                    and hasattr(new_object, "named_steps")
                    and "pre" in old_object.named_steps
                    and "clf" in old_object.named_steps
                ):
                    a = _rf_native(old_object)
                    b = _rf_native(new_object)
                else:
                    a = _plain(old_object)
                    b = _plain(new_object)
            else:
                a = json.loads(original)
                b = json.loads(candidate.read_text(encoding="utf-8"))
                if name == "feature_spec.json" and "drop_columns" not in a and b.get("drop_columns") == []:
                    serialization_defaults["feature_spec.drop_columns"] = {
                        "submitted": "absent",
                        "fresh_fit": [],
                        "predictive_effect": "none; empty optional default",
                    }
                    b.pop("drop_columns")
                if args.component == "v14_refinement" and name == "v14_refinement_spec.json":
                    for field in ("anchor_weight", "f_trend_alpha"):
                        dose_changes[field] = {
                            "submitted": a.pop(field, None),
                            "initial_fit": b.pop(field, None),
                        }
                    for field in ("candidate", "selection_rule", "v124_public_quadratic_stack"):
                        assembly_annotations[field] = {
                            "submitted": a.pop(field, None),
                            "initial_fit": b.pop(field, None),
                        }
                if args.component == "v25_anchor" and name == "v25_postbreak_anchor_spec.json":
                    dose_changes["blend_eta"] = {
                        "submitted": a.pop("blend_eta", None),
                        "initial_fit": b.pop("blend_eta", None),
                    }
                    for field in (
                        "protocol",
                        "selection_note",
                        "leaderboard_probe_parent",
                        "leaderboard_probe_parent_sha256",
                        "local_evidence_reference",
                        "v124_public_quadratic_stack",
                    ):
                        assembly_annotations[field] = {
                            "submitted": a.pop(field, None),
                            "initial_fit": b.pop(field, None),
                        }
                if args.component == "v20_pfd" and name == "v20_target1160_spec.json":
                    for index, (submitted, initial) in enumerate(
                        zip(a["eb_recipes"], b["eb_recipes"], strict=True)
                    ):
                        dose_changes[f"eb_recipes[{index}].weight"] = {
                            "submitted": submitted.pop("weight"),
                            "initial_fit": initial.pop("weight"),
                        }
                    for section, field in (
                        ("extra_mode", "weight"),
                        ("pfd", "overlay_weight"),
                    ):
                        dose_changes[f"{section}.{field}"] = {
                            "submitted": a[section].pop(field),
                            "initial_fit": b[section].pop(field),
                        }
                    assembly_annotations["v124_public_quadratic_stack"] = {
                        "submitted": a.pop("v124_public_quadratic_stack", None),
                        "initial_fit": b.pop("v124_public_quadratic_stack", None),
                    }
                if args.component == "v21_eb" and name == "v21_context_state_eb_spec.json":
                    for index, (submitted, initial) in enumerate(
                        zip(a["recipes"], b["recipes"], strict=True)
                    ):
                        dose_changes[f"recipes[{index}].weight"] = {
                            "submitted": submitted.pop("weight"),
                            "initial_fit": initial.pop("weight"),
                        }
                    assembly_annotations["v124_public_quadratic_stack"] = {
                        "submitted": a.pop("v124_public_quadratic_stack", None),
                        "initial_fit": b.pop("v124_public_quadratic_stack", None),
                    }
                if args.component == "v22_low_variance" and name == "v22_low_variance_spec.json":
                    for domain in ("R_CORE", "R_ANCHOR", "F"):
                        dose_changes[f"domain_calibration.{domain}.weight"] = {
                            "submitted": a["domain_calibration"][domain].pop("weight"),
                            "initial_fit": b["domain_calibration"][domain].pop("weight"),
                        }
                    dose_changes["asof_prior.weight"] = {
                        "submitted": a["asof_prior"].pop("weight"),
                        "initial_fit": b["asof_prior"].pop("weight"),
                    }
                    assembly_annotations["v124_public_quadratic_stack"] = {
                        "submitted": a.pop("v124_public_quadratic_stack", None),
                        "initial_fit": b.pop("v124_public_quadratic_stack", None),
                    }
                for field in ("effect_weight","eta"):
                    if field in a or field in b:
                        dose_changes[field] = {"submitted":a.pop(field,None),"initial_fit":b.pop(field,None)}
                for field in ("parent_effect_weight","variant_parent"):
                    if field in a or field in b:
                        assembly_annotations[field] = {"submitted":a.pop(field,None),"initial_fit":b.pop(field,None)}
            reports[name] = compare_native(a,b)
    result = {"component":args.component,"reference_inference_sha256":FINAL_SHA256,
              "candidate_files_sha256":{n:sha256(args.candidate_dir/n) for n in COMPONENTS[args.component]},
              "native_assets":reports,
              "native_model_and_state_parity":all(v["within_tolerance"] for v in reports.values()),
              "final_ensemble_doses_excluded_from_model_fit_comparison":dose_changes,
              "assembly_annotations_excluded":assembly_annotations,
              "serialization_defaults_excluded":serialization_defaults,
              "whole_parent_ensemble_fresh_fit":False,"private_score_recomputed":False}
    args.output_dir.mkdir(parents=True)
    (args.output_dir/(args.component+"_fresh_fit.json")).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    if not result["native_model_and_state_parity"]:
        raise ValueError("Native parent fit differs; preserve original inference")


if __name__ == "__main__":
    main()
