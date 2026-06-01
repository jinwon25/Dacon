"""가중 SMAPE 평가 + 로컬 검증 도구 — DACON 식음업장 메뉴 수요 예측 (236559).

공식 규칙 (확정):
  - 영업장별 가중치가 있는 SMAPE. '담하'·'미라시아'가 다른 업장보다 **높은 가중치**.
    단, **정확한 가중치 값은 비공개** → 로컬에서 LB 절대값을 똑같이 재현할 수 없다.
  - 실제 매출수량이 **0인 항목은 평가에서 제외**.
  - Public = test 샘플 50%, Private = 100%.
  - 베이스라인(메뉴별 LSTM) 점수: Public 0.711046 / Private 0.693935 (낮을수록 좋음).

스케일:
  베이스라인이 ~0.69 라는 점에서 점수는 다음 형태(0~2, ×100 없음)로 본다.
      smape_i = 2 * |pred - true| / (|pred| + |true|)
      score   = (0 제외 후) 영업장 가중 평균
  ⇒ 로컬 점수가 LB(0.69~0.71) 와 같은 자릿수로 나오면 스케일이 맞은 것.

전략적 함의:
  가중치 값이 비공개이므로 **영업장별 SMAPE 분해(`score_per_store`)**를 1차 지표로 삼고,
  특히 담하·미라시아 SMAPE 를 낮추는 데 집중한다. 아래 HIGH_WEIGHT(=2) 는 진단용 proxy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# --- 평가 규칙 (가중치 값은 비공개 → ×2 는 진단용 proxy) ----------------------
HIGH_WEIGHT_STORES = ("담하", "미라시아")
HIGH_WEIGHT = 2.0   # proxy (공식 값 비공개)
LOW_WEIGHT = 1.0
EXCLUDE_ZERO_ACTUAL = True  # 실제값==0 항목 제외 (공식 확정)
# ---------------------------------------------------------------------------


def store_of(item: str) -> str:
    """'담하_물냉면' -> '담하'."""
    return str(item).split("_", 1)[0]


def store_weight(item: str) -> float:
    return HIGH_WEIGHT if store_of(item) in HIGH_WEIGHT_STORES else LOW_WEIGHT


def smape_vec(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2|F-A|/(|F|+|A|), 0~2. 분모 0 이면 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.zeros_like(denom)
    nz = denom > 0
    out[nz] = 2.0 * np.abs(y_pred[nz] - y_true[nz]) / denom[nz]
    return out


def weighted_smape(y_true, y_pred, weights=None) -> float:
    """1D 배열 가중 SMAPE (0~2 스케일, LB 와 동일 자릿수). 실제값 0 항목 제외."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones_like(y_true) if weights is None else np.asarray(weights, dtype=float)

    mask = (y_true != 0) if EXCLUDE_ZERO_ACTUAL else np.ones_like(y_true, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    s = smape_vec(y_true[mask], y_pred[mask])
    return float(np.average(s, weights=w[mask]))


def score_long(df: pd.DataFrame, true_col="true", pred_col="pred", item_col="item") -> float:
    """long (item, true, pred) → 영업장 가중 SMAPE (proxy 가중치)."""
    w = df[item_col].map(store_weight).to_numpy()
    return weighted_smape(df[true_col].to_numpy(), df[pred_col].to_numpy(), weights=w)


def score_per_store(df: pd.DataFrame, true_col="true", pred_col="pred", item_col="item") -> pd.DataFrame:
    """영업장별 SMAPE 분해 (가중치 비공개 대응 — 1차 진단 지표)."""
    rows = []
    g_all = df.assign(_store=df[item_col].map(store_of))
    for store, g in g_all.groupby("_store"):
        n_eval = int((g[true_col] != 0).sum()) if EXCLUDE_ZERO_ACTUAL else len(g)
        rows.append({
            "store": store,
            "high_w": store in HIGH_WEIGHT_STORES,
            "n_eval": n_eval,
            "smape": weighted_smape(g[true_col].to_numpy(), g[pred_col].to_numpy()),
        })
    out = pd.DataFrame(rows).sort_values("smape", ascending=False).reset_index(drop=True)
    return out


# --- 로컬 검증 분할 ----------------------------------------------------------
def make_holdout(train_long: pd.DataFrame, lookback=28, horizon=7, n_folds=1, gap=0):
    """train(long)에서 test 와 동일 구조(28일 입력 → 7일 정답)의 폴드 생성.

    각 메뉴 시계열 끝에서 horizon 일을 정답으로, 그 직전 lookback 일을 입력으로 둔다.
    fold k 는 끝에서 k*(horizon+gap) 만큼 앞당김. k=0 폴드가 test 시점(2024-06-16~)과 가장 가깝다.
    반환: list[{input_df, answer_df}] (long, 컬럼: date,item,qty).
    """
    df = train_long.rename(columns={"영업일자": "date", "영업장명_메뉴명": "item", "매출수량": "qty"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"])
    folds = []
    for k in range(n_folds):
        end_offset = k * (horizon + gap)
        inp, ans = [], []
        for item, g in df.groupby("item"):
            g = g.reset_index(drop=True)
            n = len(g)
            ans_end = n - end_offset
            ans_start = ans_end - horizon
            inp_start = ans_start - lookback
            if inp_start < 0:
                continue
            inp.append(g.iloc[inp_start:ans_start])
            ans.append(g.iloc[ans_start:ans_end])
        folds.append({
            "input_df": pd.concat(inp, ignore_index=True),
            "answer_df": pd.concat(ans, ignore_index=True),
        })
    return folds


if __name__ == "__main__":
    t = np.array([10, 0, 5, 100])
    p = np.array([12, 3, 5, 80])
    print("smape_vec (0~2):", smape_vec(t, p))
    print("weighted_smape (0 제외, LB 스케일):", round(weighted_smape(t, p), 4))
