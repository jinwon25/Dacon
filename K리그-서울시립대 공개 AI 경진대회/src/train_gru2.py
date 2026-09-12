"""강화 GRU — 시퀀스(GRU) + v4 풍부 스칼라 피처 주입 + 시드 배깅.

목표: GRU 단독을 14.85 → ~14.0 수준으로 끌어올려, GBDT와의 앙상블이 진짜 효과를 내게.
- 시퀀스: 이벤트 흐름을 GRU로 인코딩 (decorrelation 원천).
- 스칼라: features_v4의 숫자 피처(표준화)를 MLP로 → GRU 출력과 합쳐 예측.
- 시드 2개 배깅으로 NN 변동성 완화.
OOF/테스트 예측 → data/cache/gru_oof.npz (앙상블용).
"""
import os
import sys

import duckdb
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, "src")
from features_v4 import FEATURE_COLS_V4, build_features_v4  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 24            # 시퀀스 윈도우(인자로 변경 가능)
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/cache/gru_oof.npz"
NUM = ["sx", "sy", "ex", "ey", "dx", "dy", "tt", "is_home", "same_team"]
SCAL_COLS = [c for c in FEATURE_COLS_V4 if c != "prev_type"]


def load_events(src):
    return duckdb.connect().execute(f"""
        SELECT game_episode, game_id, action_id, time_seconds, type_name, team_id,
               start_x, start_y, end_x, end_y, CAST(is_home AS INT) AS is_home
        FROM {src} ORDER BY game_episode, action_id""").df()


def prep(ev, typemap):
    ev = ev.copy()
    amax = ev.groupby("game_episode")["action_id"].transform("max")
    ev["is_final"] = (ev["action_id"] == amax).astype(int)
    fin = ev[ev["is_final"] == 1].set_index("game_episode")
    ev["same_team"] = (ev["team_id"] == ev["game_episode"].map(fin["team_id"])).astype(int)
    m = ev["is_final"] == 1
    ev.loc[m, "end_x"] = ev.loc[m, "start_x"]
    ev.loc[m, "end_y"] = ev.loc[m, "start_y"]
    ev["sx"] = ev["start_x"] / 105.0; ev["sy"] = ev["start_y"] / 68.0
    ev["ex"] = ev["end_x"] / 105.0; ev["ey"] = ev["end_y"] / 68.0
    ev["dx"] = (ev["end_x"] - ev["start_x"]) / 50.0
    ev["dy"] = (ev["end_y"] - ev["start_y"]) / 50.0
    ev["tt"] = ev["time_seconds"] / 300.0
    ev["tix"] = ev["type_name"].map(typemap).fillna(0).astype(int)
    return ev, fin


def build_seq(ev, fin):
    eps = fin.index.to_numpy()
    idx = {e: i for i, e in enumerate(eps)}
    N = len(eps)
    Xn = np.zeros((N, K, len(NUM)), np.float32)
    Xt = np.zeros((N, K), np.int64)
    length = np.ones(N, np.int64)
    for ep, g in ev.groupby("game_episode", sort=False):
        i = idx[ep]; gg = g.iloc[-K:]; L = len(gg)
        length[i] = L
        Xn[i, :L] = gg[NUM].to_numpy(np.float32)
        Xt[i, :L] = gg["tix"].to_numpy()
    return dict(eps=eps, Xn=Xn, Xt=Xt, length=length,
                start_x=fin["start_x"].to_numpy(np.float32),
                start_y=fin["start_y"].to_numpy(np.float32),
                game_id=fin["game_id"].to_numpy())


class Net(nn.Module):
    def __init__(self, n_types, n_scal, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n_types + 1, 8)
        self.gru = nn.GRU(len(NUM) + 8, hid, batch_first=True)
        self.smlp = nn.Sequential(nn.Linear(n_scal, 128), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(hid + 128, 128), nn.ReLU(), nn.Dropout(0.15),
                                  nn.Linear(128, 2))

    def forward(self, xn, xt, ln, sc):
        x = torch.cat([xn, self.emb(xt)], dim=2)
        packed = pack_padded_sequence(x, ln.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(torch.cat([h[-1], self.smlp(sc)], dim=1))


def batch(d, S, i):
    return (torch.tensor(d["Xn"][i]), torch.tensor(d["Xt"][i]),
            torch.tensor(d["length"][i]), torch.tensor(S[i]))


def predict(model, d, S, i, bs=1024):
    model.eval(); outs = []
    with torch.no_grad():
        for s in range(0, len(i), bs):
            xn, xt, ln, sc = batch(d, S, i[s:s + bs])
            outs.append(model(xn, xt, ln, sc).numpy())
    return np.concatenate(outs)


# ---- 데이터 ----
tr_ev = load_events("'data/train.csv'")
te_ev = load_events("read_csv('data/test/*/*.csv', union_by_name = true)")
types = sorted(tr_ev["type_name"].dropna().unique())
typemap = {t: i + 1 for i, t in enumerate(types)}
tr_ev, tr_fin = prep(tr_ev, typemap)
te_ev, te_fin = prep(te_ev, typemap)
D = build_seq(tr_ev, tr_fin)
T = build_seq(te_ev, te_fin)

# 스칼라(v4 숫자 피처) 정렬+표준화
trf = build_features_v4("'data/train.csv'").set_index("game_episode")
tef = build_features_v4("read_csv('data/test/*/*.csv', union_by_name = true)").set_index("game_episode")
tgt = trf.reindex(D["eps"])[["tgt_x", "tgt_y"]].to_numpy(np.float32)
Str = trf.reindex(D["eps"])[SCAL_COLS].to_numpy(np.float32)
Ste = tef.reindex(T["eps"])[SCAL_COLS].to_numpy(np.float32)
mu = np.nanmean(Str, 0); sd = np.nanstd(Str, 0) + 1e-6
Str = np.nan_to_num((Str - mu) / sd); Ste = np.nan_to_num((Ste - mu) / sd)
print(f"[데이터] train {len(D['eps']):,} / test {len(T['eps']):,} | 종류 {len(types)} | 스칼라 {len(SCAL_COLS)}")

fold = (D["game_id"].astype("int64") % 5)
Y = np.stack([(tgt[:, 0] - D["start_x"]) / 50.0, (tgt[:, 1] - D["start_y"]) / 50.0], 1).astype(np.float32)
SEEDS = [42, 7]
oof_dx = np.zeros(len(D["eps"]), np.float32); oof_dy = np.zeros(len(D["eps"]), np.float32)
test_dx = np.zeros(len(T["eps"]), np.float32); test_dy = np.zeros(len(T["eps"]), np.float32)

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for f in range(5):
        tr_i = np.where(fold != f)[0]; va_i = np.where(fold == f)[0]
        model = Net(len(types), len(SCAL_COLS))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.MSELoss()
        best, best_state, patience = 1e9, None, 0
        for epoch in range(50):
            model.train(); np.random.shuffle(tr_i)
            for s in range(0, len(tr_i), 256):
                j = tr_i[s:s + 256]
                xn, xt, ln, sc = batch(D, Str, j)
                opt.zero_grad()
                loss = lossf(model(xn, xt, ln, sc), torch.tensor(Y[j]))
                loss.backward(); opt.step()
            pv = predict(model, D, Str, va_i)
            px = np.clip(D["start_x"][va_i] + pv[:, 0] * 50, 0, 105)
            py = np.clip(D["start_y"][va_i] + pv[:, 1] * 50, 0, 68)
            ed = np.sqrt((px - tgt[va_i, 0]) ** 2 + (py - tgt[va_i, 1]) ** 2).mean()
            if ed < best - 1e-4:
                best, best_state, patience = ed, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 7:
                    break
        model.load_state_dict(best_state)
        pv = predict(model, D, Str, va_i)
        oof_dx[va_i] += pv[:, 0] / len(SEEDS); oof_dy[va_i] += pv[:, 1] / len(SEEDS)
        pt = predict(model, T, Ste, np.arange(len(T["eps"])))
        test_dx += pt[:, 0] / (5 * len(SEEDS)); test_dy += pt[:, 1] / (5 * len(SEEDS))
        print(f"  seed {seed} fold {f}: val {best:.4f}")

oof_x = np.clip(D["start_x"] + oof_dx * 50, 0, 105)
oof_y = np.clip(D["start_y"] + oof_dy * 50, 0, 68)
cv = np.sqrt((oof_x - tgt[:, 0]) ** 2 + (oof_y - tgt[:, 1]) ** 2).mean()
pred_x = np.clip(T["start_x"] + test_dx * 50, 0, 105)
pred_y = np.clip(T["start_y"] + test_dy * 50, 0, 68)
print(f"\n  ★ 강화 GRU(K={K}) 로컬 CV {cv:.4f}")
np.savez(OUT, ep=D["eps"].astype(str), oof_x=oof_x, oof_y=oof_y,
         test_ep=T["eps"].astype(str), pred_x=pred_x, pred_y=pred_y)
print(f"[완료] {OUT}")
