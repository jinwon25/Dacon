"""Score fallback blend weights on the pressure route against the parent OOF.

The parent package already predicts every row.  This asks a narrower question:
on R_CORE rows with runners on or high leverage, where the parent is already
confident (p >= 0.50), how much does mixing in the independent fallback XGB
help?  Weights are scored per audit season, then by month and by a pitcher-level
bootstrap for 2024, so a gain has to survive resampling before it is adopted.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OOF = ROOT.parent / "artifacts" / "private_oof_runtime"
DEFAULT_TRAIN = ROOT.parent / "data" / "train.csv"


def bss(y, p):
    rate = y.mean()
    return 1e5 * (1 - np.mean((y - p) ** 2) / (rate * (1 - rate)))


def main(train_csv: Path = DEFAULT_TRAIN, oof_dir: Path = DEFAULT_OOF):
    OOF = oof_dir
    cols = ["season", "game_type", "pitcher_team_id", "batter_team_id", "num_runners_on", "li", "game_month", "pitcher_id"]
    train = pd.read_csv(train_csv, usecols=cols, encoding="utf-8-sig")
    records = []
    for year, name in [(2022, "v84_full_2022.npz"), (2023, "v84_late_2023.npz"), (2024, "v148_full_2024.npz")]:
        z = np.load(OOF / name, allow_pickle=True)
        raw = z["raw_index"]
        frame = train.iloc[raw].reset_index(drop=True)
        y, p = z["target"].astype(float), z["parent"].astype(float)
        regular = frame.game_type.astype(str).eq("R").to_numpy()
        core = regular & ~(frame.pitcher_team_id.eq(13) | frame.batter_team_id.eq(13)).to_numpy()
        active = core & ((frame.num_runners_on.to_numpy() > 0) | (frame.li.to_numpy() >= 1.5))
        full_r = np.flatnonzero(train.season.eq(year) & train.game_type.astype(str).eq("R"))
        pos = pd.Series(np.arange(len(full_r)), index=full_r).reindex(raw).to_numpy()
        fallback = np.load(OOF / f"fallback_xgb_oof_{year}.npy")
        x = p.copy()
        x[regular] = fallback[pos[regular].astype(int)]
        gate = active & (p >= .50)
        for weight in [.10, .20, .30, .40]:
            cand = p.copy()
            cand[gate] = (1 - weight) * p[gate] + weight * x[gate]
            records.append({"year": year, "weight": weight, "support": int(gate.sum()), "base_bss": bss(y,p), "bss": bss(y,cand), "delta_bss": bss(y,cand)-bss(y,p)})
    out = pd.DataFrame(records)
    out.to_csv(OOF / "fallback_route_weight_scan.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nmean by weight")
    print(out.groupby("weight").delta_bss.mean().to_string(float_format=lambda x: f"{x:+.6f}"))
    z = np.load(OOF / "v148_full_2024.npz", allow_pickle=True)
    raw = z["raw_index"]
    frame = train.iloc[raw].reset_index(drop=True)
    y, p = z["target"].astype(float), z["parent"].astype(float)
    regular = frame.game_type.astype(str).eq("R").to_numpy()
    core = regular & ~(frame.pitcher_team_id.eq(13) | frame.batter_team_id.eq(13)).to_numpy()
    gate = core & ((frame.num_runners_on.to_numpy() > 0) | (frame.li.to_numpy() >= 1.5)) & (p >= .50)
    full_r = np.flatnonzero(train.season.eq(2024) & train.game_type.astype(str).eq("R"))
    pos = pd.Series(np.arange(len(full_r)), index=full_r).reindex(raw).to_numpy()
    x = p.copy(); x[regular] = np.load(OOF / "fallback_xgb_oof_2024.npy")[pos[regular].astype(int)]
    cand = p.copy(); cand[gate] = .7 * p[gate] + .3 * x[gate]
    monthly = []
    for month in sorted(frame.game_month.dropna().unique()):
        m = frame.game_month.to_numpy() == month
        monthly.append({"month": int(month), "rows": int(m.sum()), "active": int(gate[m].sum()), "delta_bss": bss(y[m], cand[m]) - bss(y[m], p[m])})
    monthly = pd.DataFrame(monthly)
    monthly.to_csv(OOF / "fallback_route_2024_monthly.csv", index=False, encoding="utf-8-sig")
    print("\n2024 monthly, weight .30")
    print(monthly.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    rng = np.random.default_rng(20260828)
    pid = frame.pitcher_id.to_numpy(); ids = np.unique(pid); draws=[]
    groups = {u: np.flatnonzero(pid == u) for u in ids}
    for _ in range(500):
        sampled = rng.choice(ids, len(ids), replace=True)
        ix = np.concatenate([groups[u] for u in sampled])
        draws.append(bss(y[ix], cand[ix]) - bss(y[ix], p[ix]))
    print("2024 pitcher bootstrap p05/p50/p95/P+", np.quantile(draws, [.05,.5,.95]), float(np.mean(np.asarray(draws)>0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args()
    main(args.train_csv, args.oof_dir)
