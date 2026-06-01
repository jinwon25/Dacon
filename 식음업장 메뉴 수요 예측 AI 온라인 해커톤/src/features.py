"""글로벌 모델용 피처 빌더 — direct multi-horizon (28일 입력 → h=1..7 직접 예측).

누수 규칙 준수: 모든 피처는 anchor t 기준 직전 28일 윈도우 + 예측일(t+h)의 달력정보만 사용.
같은 함수(`make_feature_frame`)를 학습(전체 이력)과 추론(28일 윈도우)에 동일하게 써서 일관성 보장.

핵심 피처:
  - anchor 기준 lag/롤링 통계 (수준·추세·변동·0비율)
  - 같은 요일 과거값 swl1..4 = y[t+h-7k] (주간 계절성), 그 평균 swl_mean (= 요일평균 예측기)
  - 예측일 달력: 요일/주말/일/월/공휴일
  - item·store 범주형
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# 한국 공휴일 (도메인 지식 — 규정상 허용). train/test 기간(2023~2024) 커버.
KR_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-06-06", "2023-08-15",
    "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-03", "2023-10-09",
    "2023-12-25",
    "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15",
    "2024-06-06", "2024-08-15", "2024-09-16", "2024-09-17", "2024-09-18",
    "2024-10-03", "2024-10-09", "2024-12-25",
])
_HOLISET = set(KR_HOLIDAYS.normalize())

LAGS = [0, 1, 2, 3, 6, 7, 13, 14, 21, 27]
FEATURE_COLS = (
    [f"lag{k}" for k in LAGS]
    + ["r7_mean", "r7_std", "r7_max", "r14_mean", "r28_mean", "r28_std",
       "r28_max", "r28_min", "zr28", "nz28_mean", "trend7", "anchor_wd"]
    + ["h", "swl1", "swl2", "swl3", "swl4", "swl_mean",
       "tgt_wd", "tgt_weekend", "tgt_dom", "tgt_month", "tgt_holiday"]
)
CAT_COLS = ["item", "store"]


def store_of(item: str) -> str:
    return str(item).split("_", 1)[0]


def make_feature_frame(item: str, dates: pd.Series, qty: np.ndarray,
                       horizon=7, with_target=True, min_anchor=27) -> pd.DataFrame:
    """한 메뉴의 연속 일별 시계열 → (t,h) 행 피처 프레임.

    dates: 연속 일별 Timestamp (정렬됨), qty: 같은 길이 매출수량.
    with_target=False 이면 target 없이 추론용(28행 윈도우면 t=27 한 anchor → 7행).
    """
    s = pd.Series(np.asarray(qty, dtype=float))
    n = len(s)
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))

    base = {}
    for k in LAGS:
        base[f"lag{k}"] = s.shift(k)
    base["r7_mean"] = s.rolling(7).mean()
    base["r7_std"] = s.rolling(7).std()
    base["r7_max"] = s.rolling(7).max()
    base["r14_mean"] = s.rolling(14).mean()
    base["r28_mean"] = s.rolling(28).mean()
    base["r28_std"] = s.rolling(28).std()
    base["r28_max"] = s.rolling(28).max()
    base["r28_min"] = s.rolling(28).min()
    base["zr28"] = (s == 0).rolling(28).mean()
    nz_sum = s.rolling(28).sum()
    nz_cnt = (s > 0).rolling(28).sum()
    base["nz28_mean"] = (nz_sum / nz_cnt.replace(0, np.nan)).fillna(0.0)
    base["trend7"] = s.rolling(7).mean() - s.shift(7).rolling(7).mean()
    base["anchor_wd"] = dates.dt.weekday.astype(float)
    base = pd.DataFrame(base)

    store = store_of(item)
    blocks = []
    for h in range(1, horizon + 1):
        df = base.copy()
        df["h"] = h
        for j, k in enumerate([1, 2, 3, 4], start=1):
            df[f"swl{j}"] = s.shift(7 * k - h)  # y[t+h-7k], 같은 요일 과거값
        df["swl_mean"] = df[["swl1", "swl2", "swl3", "swl4"]].mean(axis=1)
        tgt_date = dates + pd.Timedelta(days=h)
        df["tgt_wd"] = tgt_date.dt.weekday.astype(float)
        df["tgt_weekend"] = (tgt_date.dt.weekday >= 5).astype(float)
        df["tgt_dom"] = tgt_date.dt.day.astype(float)
        df["tgt_month"] = tgt_date.dt.month.astype(float)
        df["tgt_holiday"] = tgt_date.dt.normalize().isin(_HOLISET).astype(float)
        df["target_date"] = tgt_date
        df["anchor_idx"] = np.arange(n)
        if with_target:
            df["target"] = s.shift(-h)
        blocks.append(df)

    out = pd.concat(blocks, ignore_index=True)
    out["item"] = item
    out["store"] = store
    # 유효행: 28일 윈도우 완성(anchor>=min_anchor) + 핵심 피처 결측 없음
    valid = out["anchor_idx"] >= min_anchor
    if with_target:
        valid &= out["target"].notna()
    out = out[valid].reset_index(drop=True)
    return out


def build_training_table(train_long: pd.DataFrame, cutoff_date=None, horizon=7, stride=1):
    """전체 train(long) → 학습용 (X, y, meta). cutoff_date 이상 target은 제외(홀드아웃 보호)."""
    df = train_long.rename(columns={"영업일자": "date", "영업장명_메뉴명": "item", "매출수량": "qty"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"])
    frames = []
    for item, g in df.groupby("item"):
        g = g.sort_values("date")
        fr = make_feature_frame(item, g["date"].values, g["qty"].values, horizon, with_target=True)
        frames.append(fr)
    full = pd.concat(frames, ignore_index=True)
    if cutoff_date is not None:
        full = full[full["target_date"] < pd.Timestamp(cutoff_date)].reset_index(drop=True)
    if stride > 1:
        full = full[full["anchor_idx"] % stride == 0].reset_index(drop=True)
    return full


def build_inference_table(input_long_28: pd.DataFrame, horizon=7):
    """28일 입력(long: date,item,qty) → 추론용 피처 (item별 7행)."""
    df = input_long_28.rename(columns={"영업일자": "date", "영업장명_메뉴명": "item", "매출수량": "qty"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"])
    frames = []
    for item, g in df.groupby("item"):
        g = g.sort_values("date")
        fr = make_feature_frame(item, g["date"].values, g["qty"].values, horizon, with_target=False)
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)
