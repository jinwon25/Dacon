"""Row-independence smoke test for the fallback XGB asset."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT.parents[1] / "reference-hyunku-lga-data" / "lga_data-main"
ASSET = ROOT / "artifacts" / "private_oof_runtime" / "jy_fallback_xgb_asset"
sys.path.insert(0, str(HY))
import codex_reproduction.repro_v10a_core as rr


def predict(rows, history, model, columns):
    old = rr.ART
    rr.ART = ASSET
    try:
        f = rr.add_public_context_features(rows.copy(), history, rr.make_base_features(rows.copy(), history))
    finally:
        rr.ART = old
    f = f.reindex(columns=columns, fill_value=0.)
    return model.predict_proba(f)[:, 1]


def main():
    train = pd.read_csv(HY / "data" / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(HY / "data" / "test.csv", encoding="utf-8-sig")
    columns = json.loads((ASSET / "feature_columns.json").read_text(encoding="utf-8"))
    model = xgb.XGBClassifier(); model.load_model(ASSET / "fallback_xgb.json")
    whole = predict(test, train, model, columns)
    single = np.array([predict(test.iloc[[i]], train, model, columns)[0] for i in range(len(test))])
    shuffled = predict(test.sample(frac=1, random_state=20260828), train, model, columns)
    order = test.sample(frac=1, random_state=20260828).index.to_numpy()
    restored = pd.Series(shuffled, index=order).reindex(test.index).to_numpy()
    print({"rows":len(test), "min":float(whole.min()), "max":float(whole.max()), "single_max_abs_diff":float(np.max(np.abs(whole-single))), "shuffle_max_abs_diff":float(np.max(np.abs(whole-restored)))})
    if not (np.allclose(whole,single,atol=1e-12,rtol=0) and np.allclose(whole,restored,atol=1e-12,rtol=0)):
        raise SystemExit("row independence failed")


if __name__ == "__main__":
    main()
