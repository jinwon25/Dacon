"""Freeze all train-only lookups needed by the fallback XGB test runtime."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HY = ROOT.parents[1] / "reference-hyunku-lga-data" / "lga_data-main"
ASSET = ROOT / "artifacts" / "private_oof_runtime" / "jy_fallback_xgb_asset"
sys.path.insert(0, str(HY))
import codex_reproduction.repro_v10a_core as rr


def pairs(frame, key, values):
    return {int(k): tuple(float(v) for v in row) for k, row in frame.set_index(key)[values].iterrows()}


def main():
    train = pd.read_csv(HY / "data" / "train.csv", encoding="utf-8-sig")
    y = train.control_success.to_numpy(float)
    payload = {"target_mean": float(y.mean()), "priors": {}, "cat": {}, "anchors": {}, "situations": {}, "tm": {}, "features_version": 1}
    for col in rr.CAT:
        payload["cat"][col] = {str(v): int(i) for i, v in enumerate(pd.Index(train[col].fillna("__NA__").astype(str).unique()))}
    for idcol, ncol, ratecol, pref in rr.SPECS:
        anchor = rr._anchor_table(train, idcol, ncol, ratecol).sort_values("season").groupby(idcol, sort=False).tail(1)
        payload["anchors"][pref] = pairs(anchor, idcol, [ncol, "succ"])
        payload["priors"][pref] = float(train[ratecol].mean())
    situations = {
        "3ball": train.balls_before.eq(3), "2strk": train.strikes_before.eq(2),
        "ahead": train.strikes_before.gt(train.balls_before), "behind": train.balls_before.gt(train.strikes_before),
        "risp": train.runner_on_2b.eq(1) | train.runner_on_3b.eq(1), "on1b": train.runner_on_1b.eq(1),
        "vsL": train.batter_hand.eq(1), "vsR": train.batter_hand.eq(2), "late": train.inning.ge(7),
        "hiLI": train.li.gt(1.5), "loLI": train.li.lt(.5), "blowout": train.score_diff_pitcher_team.abs().ge(5),
    }
    total = train.groupby("pitcher_id", sort=False).control_success.agg(["sum", "count"])
    overall = (total["sum"] + 300*y.mean()) / (total["count"] + 300)
    payload["overall"] = {int(k): float(v) for k, v in overall.items()}
    for name, mask in situations.items():
        tab = train.loc[mask].groupby("pitcher_id", sort=False).control_success.agg(["sum", "count"]).reindex(total.index).fillna(0.)
        rate = (tab["sum"] + 300*(total["sum"] / total["count"])) / (tab["count"] + 300)
        payload["situations"][name] = {int(k): float(v) for k, v in rate.items()}
    pb = train.assign(_key=train.pitcher_id.astype("int64")*100000 + train.batter_id.astype("int64")).groupby("_key",sort=False).control_success.agg(["sum","count"])
    payload["pb"] = pairs(pb.reset_index(), "_key", ["sum", "count"])
    gid = (train.inning.diff().fillna(0) < 0).cumsum()
    ppa = (train.assign(_gid=gid).groupby(["_gid","pitcher_id","season"],sort=False).size().groupby(["pitcher_id","season"],sort=False).median().rename("ppa").reset_index())
    ppa_map = ppa.groupby("pitcher_id",sort=False).ppa.median()
    payload["ppa"] = {int(k): float(v) for k,v in ppa_map.items()}
    payload["ppa_default"] = float(ppa.ppa.median())
    profile = pd.read_pickle(ASSET / "trackman_prior_profile_v2.pkl")
    for col in ["rel_speed","rel_speed_sd","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side"]:
        rate = profile.groupby("pitcher_id",sort=False)[col].mean()
        payload["tm"][col] = {int(k): float(v) for k,v in rate.dropna().items()}
    joblib.dump(payload, ASSET / "fallback_lookups.joblib", compress=3)
    print("saved", ASSET / "fallback_lookups.joblib", "bytes", (ASSET / "fallback_lookups.joblib").stat().st_size)


if __name__ == "__main__":
    main()
