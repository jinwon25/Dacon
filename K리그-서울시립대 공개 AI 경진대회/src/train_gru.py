"""GRU 시퀀스 모델 — GBDT와 '근본적으로 다른 시야'로 decorrelation 만들기.

왜? GBDT 여럿은 같은 피처를 보고 corr~0.99 → 앙상블 무력. 시퀀스 NN은 이벤트의
'순서/흐름' 자체를 읽으므로 GBDT와 틀리는 지점이 달라진다 → 앙상블 이득(4th place 레시피).

설계
- 입력 = 에피소드의 최근 K개 이벤트 시퀀스. 각 이벤트 = [정규화 좌표/변위, 시간, 홈, 같은팀여부] + 종류 임베딩.
  마지막(예측 대상) 패스는 도착 가림(end=start, dx=dy=0).
- GRU로 시퀀스 인코딩 → 마지막 패스의 알려진 맥락(scalars)과 합쳐 → 변위(dx,dy) 예측.
- 타깃 = (도착-출발)/50.  복원: 출발 + 예측변위.  5-fold(game_id) OOF.
- OOF/테스트 예측 저장 → 앙상블 단계에서 GBDT와 결합.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, "src")

torch.manual_seed(42)
np.random.seed(42)
K = 24                 # 최근 이벤트 수 cap
NUM = ["sx", "sy", "ex", "ey", "dx", "dy", "tt", "is_home", "same_team"]


def load_events(source):
    return duckdb.connect().execute(f"""
        SELECT game_episode, game_id, action_id, time_seconds, type_name, team_id,
               start_x, start_y, end_x, end_y, CAST(is_home AS INT) AS is_home
        FROM {source} ORDER BY game_episode, action_id
    """).df()


def prep(ev, typemap):
    ev = ev.copy()
    amax = ev.groupby("game_episode")["action_id"].transform("max")
    ev["is_final"] = (ev["action_id"] == amax).astype(int)
    # 타깃(최종 패스 실제 도착) — 마스킹 전에 추출
    fin = ev[ev["is_final"] == 1].set_index("game_episode")
    fteam = fin["team_id"]
    ev["same_team"] = (ev["team_id"] == ev["game_episode"].map(fteam)).astype(int)
    # 최종 패스 도착 가림
    m = ev["is_final"] == 1
    ev.loc[m, "end_x"] = ev.loc[m, "start_x"]
    ev.loc[m, "end_y"] = ev.loc[m, "start_y"]
    # 정규화 채널
    ev["sx"] = ev["start_x"] / 105.0
    ev["sy"] = ev["start_y"] / 68.0
    ev["ex"] = ev["end_x"] / 105.0
    ev["ey"] = ev["end_y"] / 68.0
    ev["dx"] = (ev["end_x"] - ev["start_x"]) / 50.0
    ev["dy"] = (ev["end_y"] - ev["start_y"]) / 50.0
    ev["tt"] = ev["time_seconds"] / 300.0
    ev["tix"] = ev["type_name"].map(typemap).fillna(0).astype(int)
    return ev, fin


def build_tensors(ev, fin, has_target):
    eps = fin.index.to_numpy()
    idx = {e: i for i, e in enumerate(eps)}
    N = len(eps)
    Xn = np.zeros((N, K, len(NUM)), dtype=np.float32)
    Xt = np.zeros((N, K), dtype=np.int64)
    length = np.ones(N, dtype=np.int64)
    scal = np.zeros((N, 5), dtype=np.float32)
    for ep, g in ev.groupby("game_episode", sort=False):
        i = idx[ep]
        gg = g.iloc[-K:]
        L = len(gg)
        length[i] = L
        Xn[i, :L] = gg[NUM].to_numpy(dtype=np.float32)
        Xt[i, :L] = gg["tix"].to_numpy()
    # 최종 패스 맥락 스칼라
    sx = fin["start_x"].to_numpy(); sy = fin["start_y"].to_numpy()
    scal[:, 0] = sx / 105.0
    scal[:, 1] = sy / 68.0
    scal[:, 2] = fin["is_home"].to_numpy()
    scal[:, 3] = fin["time_seconds"].to_numpy() / 300.0
    scal[:, 4] = np.hypot(105 - sx, 34 - sy) / 120.0
    out = dict(eps=eps, Xn=Xn, Xt=Xt, length=length, scal=scal,
               start_x=sx.astype(np.float32), start_y=sy.astype(np.float32),
               game_id=fin["game_id"].to_numpy())
    if has_target:
        out["tgt_x"] = fin["end_x"].to_numpy().astype(np.float32)
        out["tgt_y"] = fin["end_y"].to_numpy().astype(np.float32)
    return out


class GRUNet(nn.Module):
    def __init__(self, n_types, hid=96):
        super().__init__()
        self.emb = nn.Embedding(n_types + 1, 8)
        self.gru = nn.GRU(len(NUM) + 8, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + 5, 128), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(128, 2))

    def forward(self, xn, xt, length, scal):
        x = torch.cat([xn, self.emb(xt)], dim=2)
        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(torch.cat([h[-1], scal], dim=1))


def to_t(d, i):
    return (torch.tensor(d["Xn"][i]), torch.tensor(d["Xt"][i]),
            torch.tensor(d["length"][i]), torch.tensor(d["scal"][i]))


def predict(model, d, i, bs=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(i), bs):
            j = i[s:s + bs]
            xn, xt, ln, sc = to_t(d, j)
            outs.append(model(xn, xt, ln, sc).numpy())
    return np.concatenate(outs)


# ---- 데이터 ----
tr_ev = load_events("'data/train.csv'")
te_ev = load_events("read_csv('data/test/*/*.csv', union_by_name = true)")
types = sorted(tr_ev["type_name"].dropna().unique())
typemap = {t: i + 1 for i, t in enumerate(types)}
tr_ev, tr_fin = prep(tr_ev, typemap)
te_ev, te_fin = prep(te_ev, typemap)
D = build_tensors(tr_ev, tr_fin, True)
T = build_tensors(te_ev, te_fin, False)
print(f"[데이터] train {len(D['eps']):,} / test {len(T['eps']):,} | 종류 {len(types)} | K={K}")

fold = (D["game_id"].astype("int64") % 5)
oof_dx = np.zeros(len(D["eps"]), dtype=np.float32)
oof_dy = np.zeros(len(D["eps"]), dtype=np.float32)
test_dx = np.zeros(len(T["eps"]), dtype=np.float32)
test_dy = np.zeros(len(T["eps"]), dtype=np.float32)
tgt_dx = (D["tgt_x"] - D["start_x"]) / 50.0
tgt_dy = (D["tgt_y"] - D["start_y"]) / 50.0
Y = np.stack([tgt_dx, tgt_dy], axis=1).astype(np.float32)

for f in range(5):
    tr_i = np.where(fold != f)[0]
    va_i = np.where(fold == f)[0]
    model = GRUNet(len(types))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    best, best_state, patience = 1e9, None, 0
    for epoch in range(45):
        model.train()
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), 256):
            j = tr_i[s:s + 256]
            xn, xt, ln, sc = to_t(D, j)
            opt.zero_grad()
            out = model(xn, xt, ln, sc)
            loss = lossf(out, torch.tensor(Y[j]))
            loss.backward()
            opt.step()
        # val euclidean (복원)
        pv = predict(model, D, va_i)
        px = np.clip(D["start_x"][va_i] + pv[:, 0] * 50, 0, 105)
        py = np.clip(D["start_y"][va_i] + pv[:, 1] * 50, 0, 68)
        ed = np.sqrt((px - D["tgt_x"][va_i]) ** 2 + (py - D["tgt_y"][va_i]) ** 2).mean()
        if ed < best - 1e-4:
            best, best_state, patience = ed, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    pv = predict(model, D, va_i)
    oof_dx[va_i], oof_dy[va_i] = pv[:, 0], pv[:, 1]
    pt = predict(model, T, np.arange(len(T["eps"])))
    test_dx += pt[:, 0] / 5
    test_dy += pt[:, 1] / 5
    print(f"  fold {f}: val euclidean {best:.4f}")

oof_x = np.clip(D["start_x"] + oof_dx * 50, 0, 105)
oof_y = np.clip(D["start_y"] + oof_dy * 50, 0, 68)
cv = np.sqrt((oof_x - D["tgt_x"]) ** 2 + (oof_y - D["tgt_y"]) ** 2).mean()
pred_x = np.clip(T["start_x"] + test_dx * 50, 0, 105)
pred_y = np.clip(T["start_y"] + test_dy * 50, 0, 68)

print(f"\n  ★ GRU 로컬 CV {cv:.4f}   (GBDT v4 = 14.02)")
os.makedirs("data/cache", exist_ok=True)
np.savez("data/cache/gru_oof.npz",
         ep=D["eps"].astype(str), oof_x=oof_x, oof_y=oof_y,
         test_ep=T["eps"].astype(str), pred_x=pred_x, pred_y=pred_y)
print("[완료] data/cache/gru_oof.npz")
