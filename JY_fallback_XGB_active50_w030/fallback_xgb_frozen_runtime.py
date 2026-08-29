"""Frozen train-only feature transform for the fallback TrackMan XGB."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

TARGET = "control_success"
CAT = ["top_bottom","game_type","base_state","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","pitcher_id","batter_id"]
SPECS = [("pitcher_id","asof_pitcher_n","asof_pitcher_success_rate","p_succ"),("pitcher_id","asof_pitcher_n","asof_pitcher_reverse_rate","p_rev"),("pitcher_id","asof_pitcher_n","asof_pitcher_middle_rate","p_mid"),("pitcher_id","asof_pitcher_n","asof_pitcher_ball_rate","p_ball"),("pitcher_id","asof_pitcher_n","asof_pitcher_strike_rate","p_stk"),("batter_id","asof_batter_n","asof_batter_success_rate","b_succ"),("batter_id","asof_batter_n","asof_batter_middle_rate","b_mid")]
SITS = ["3ball","2strk","ahead","behind","risp","on1b","vsL","vsR","late","hiLI","loLI","blowout"]
TM = ["rel_speed","rel_speed_sd","spin_rate","induced_vert_break","horz_break","extension","rel_height","rel_side"]


def build(frame: pd.DataFrame, asset: Path) -> pd.DataFrame:
    a = joblib.load(asset / "fallback_lookups.joblib")
    x = frame.drop(columns=["row_id", TARGET, "asof_pitcher_pitchmix_n"], errors="ignore").copy()
    for c in CAT:
        x[c] = frame[c].fillna("__NA__").astype(str).map(a["cat"][c]).fillna(-1).astype(np.float32)
    x["hand_mix"] = x["pitcher_hand"] * 2 + x["batter_hand"]
    for idcol,ncol,ratecol,pref in SPECS:
        ids = frame[idcol].astype(int).to_numpy(); pairs = a["anchors"][pref]
        an = np.asarray([pairs.get(int(v),(0.,0.))[0] for v in ids]); ass = np.asarray([pairs.get(int(v),(0.,0.))[1] for v in ids])
        n = frame[ncol].to_numpy(float); r = frame[ratecol].to_numpy(float); prior = a["priors"][pref]
        dn=np.maximum(n-an,0); ds=np.maximum(np.nan_to_num(n*r)-ass,0)
        x[pref+"_ssn"]=(ds+150*prior)/(dn+150); x[pref+"_ssn_vs_car"]=x[pref+"_ssn"]-np.nan_to_num(r,nan=prior)
        if pref in ("p_succ","b_succ"): x[pref+"_ssn_n"]=dn
    x["p_prev5_vs_car"] = frame["asof_pitcher_prev5_game_success_rate"] - frame["asof_pitcher_success_rate"]
    x["p_prev1_vs_prev5"] = frame["asof_pitcher_prev1_game_success_rate"] - frame["asof_pitcher_prev5_game_success_rate"]
    # Match original future-season branch: multi-k has a zero anchor.
    for idcol,ncol,ratecol,pref in [("pitcher_id","asof_pitcher_n","asof_pitcher_success_rate","p_succ"),("batter_id","asof_batter_n","asof_batter_success_rate","b_succ")]:
        n=frame[ncol].to_numpy(float); r=frame[ratecol].to_numpy(float); prior=a["priors"][pref]
        for k in (25,75,400,1000): x[f"{pref}_k{k}"]=(np.nan_to_num(n*r)+k*prior)/(n+k)
    pid=frame.pitcher_id.astype(int).to_numpy()
    overall=np.asarray([a["overall"].get(int(v),np.nan) for v in pid])
    for name in SITS:
        rate=np.asarray([a["situations"][name].get(int(v),np.nan) for v in pid]); x["p_sit_"+name]=rate
    x["p_sit_overall"]=overall
    for name in SITS: x["p_sit_"+name+"_d"]=x["p_sit_"+name]-overall
    masks=[frame.balls_before.eq(3),frame.strikes_before.eq(2),frame.strikes_before.gt(frame.balls_before),frame.balls_before.gt(frame.strikes_before),frame.runner_on_2b.eq(1)|frame.runner_on_3b.eq(1),frame.runner_on_1b.eq(1),frame.batter_hand.eq(1),frame.batter_hand.eq(2),frame.inning.ge(7),frame.li.gt(1.5),frame.li.lt(.5),frame.score_diff_pitcher_team.abs().ge(5)]
    matched=np.full(len(frame),np.nan)
    for name,mask in zip(SITS,masks):
        delta=x["p_sit_"+name+"_d"].to_numpy(); take=mask.to_numpy() & np.isnan(matched); matched[take]=delta[take]
    x["p_sit_matched"]=matched
    key=(frame.pitcher_id.astype("int64")*100000+frame.batter_id.astype("int64")).to_numpy(); pb=a["pb"]
    vals=[pb.get(int(k),(0.,0.)) for k in key]; su=np.asarray([v[0] for v in vals]); cnt=np.asarray([v[1] for v in vals])
    x["pb_n"]=cnt; x["pb_rate"]=(su+30*a["target_mean"])/(cnt+30); x["pb_logn"]=np.log1p(cnt)
    ppa=np.asarray([a["ppa"].get(int(v),a["ppa_default"]) for v in pid]); x["p_ppa"]=ppa; x["p_est_apps"]=x["p_succ_ssn_n"]/np.clip(ppa,5,None); x["p_inning_x_role"]=frame.inning.to_numpy(float)*np.log1p(ppa); x["p_ssn_per_month"]=x["p_succ_ssn_n"]/np.clip(frame.game_month.to_numpy(float),3,None)
    for c in TM: x["tm_"+c]=np.asarray([a["tm"][c].get(int(v),np.nan) for v in pid])
    columns=json.loads((asset / "feature_columns.json").read_text(encoding="utf-8"))
    return x.apply(pd.to_numeric,errors="coerce").astype(np.float32).reindex(columns=columns)


def predict(frame: pd.DataFrame, asset: Path) -> np.ndarray:
    model=xgb.XGBClassifier(); model.load_model(asset / "fallback_xgb.json")
    return model.predict_proba(build(frame,asset))[:,1]
