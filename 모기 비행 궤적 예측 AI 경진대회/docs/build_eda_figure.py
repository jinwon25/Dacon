"""build_eda_figure — 발표자료 §3(데이터 이해) 슬라이드용 EDA 그림 생성.

train 궤적에서 속도·가속도·turn-rate(연속 속도벡터 사잇각) 분포 + 급기동 샘플 궤적을
2x2 패널로 그려 PNG로 저장한다. (축 라벨은 폰트 이슈 없게 영어)

사용: python docs/build_eda_figure.py
출력: docs/eda_slide3.png  (+ 다운로드 폴더에도 복사 시도)
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 등록 (맑은 고딕) — suptitle 한글 깨짐 방지
for _fp in (r"C:\Windows\Fonts\malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
CACHE = DATA / "cache" / "xtrain_xtest.npz"
DT = 0.040
OUT = HERE / "eda_slide3.png"

# ---- load trajectories (cache 우선, 없으면 샘플 로드) ----
if CACHE.exists():
    X = np.load(CACHE)["X_train"].astype(np.float64)
    print(f"[data] cache 사용: {X.shape}")
else:
    files = sorted(glob.glob(str(DATA / "train" / "*.csv")))
    rng = np.random.RandomState(0)
    files = [files[i] for i in rng.choice(len(files), size=min(3000, len(files)), replace=False)]
    X = np.stack([pd.read_csv(f)[["x", "y", "z"]].to_numpy() for f in files], axis=0).astype(np.float64)
    print(f"[data] 샘플 로드: {X.shape}")

# ---- 물리량 ----
v = np.diff(X, axis=1) / DT                       # (N,10,3) velocity
a = np.diff(v, axis=1) / DT                        # (N,9,3)  accel
speed = np.linalg.norm(v, axis=-1)                 # (N,10)
acc = np.linalg.norm(a, axis=-1)                   # (N,9)
# turn-rate: 연속 속도벡터 사잇각(deg)
v1, v2 = v[:, :-1, :], v[:, 1:, :]
cos = (v1 * v2).sum(-1) / np.clip(np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1), 1e-9, None)
turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))  # (N,9)

# ---- 급기동(누적 turn 큰) 샘플 + 완만 샘플 ----
total_turn = turn.sum(axis=1)
hard = np.argsort(-total_turn)[:6]
easy = np.argsort(total_turn)[:2]

# ---- plot ----
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.edgecolor": "#888", "figure.dpi": 150})
BLUE = "#2563eb"
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))

ax[0].hist(speed.ravel(), bins=60, color=BLUE, alpha=0.85)
ax[0].set_title("Speed  |v|  (per 40ms step)", fontweight="bold")
ax[0].set_xlabel("speed (units/s)"); ax[0].set_ylabel("count")

ax[1].hist(acc.ravel(), bins=60, color="#7c3aed", alpha=0.85)
ax[1].set_title("Acceleration  |a|", fontweight="bold")
ax[1].set_xlabel("accel (units/s^2)")

ax[2].hist(turn.ravel(), bins=60, color="#dc2626", alpha=0.85)
ax[2].set_title("Turn-rate  (angle between consecutive v)", fontweight="bold")
ax[2].set_xlabel("turn per step (deg)")
med = float(np.median(turn)); p90 = float(np.percentile(turn, 90))
ax[2].axvline(med, color="k", ls="--", lw=1); ax[2].axvline(p90, color="k", ls=":", lw=1)
ax[2].text(0.97, 0.92, f"median {med:.1f}°\n p90 {p90:.1f}°", transform=ax[2].transAxes,
           ha="right", va="top", fontsize=10, bbox=dict(fc="white", ec="#ccc"))

# 샘플 궤적 (xy 평면, last obs 원점 정렬)
for k, idx in enumerate(hard):
    P = X[idx] - X[idx, -1]
    ax[3].plot(P[:, 0], P[:, 1], "-", color="#dc2626", alpha=0.55, lw=1.4)
    ax[3].plot(P[-1, 0], P[-1, 1], "o", color="#dc2626", ms=4)
for idx in easy:
    P = X[idx] - X[idx, -1]
    ax[3].plot(P[:, 0], P[:, 1], "-", color=BLUE, alpha=0.7, lw=1.4)
    ax[3].plot(P[-1, 0], P[-1, 1], "o", color=BLUE, ms=4)
ax[3].set_title("Sample trajectories (xy, last obs = origin)", fontweight="bold")
ax[3].set_xlabel("x"); ax[3].set_ylabel("y"); ax[3].axis("equal")
ax[3].plot([], [], "-", color="#dc2626", label="high-turn (급기동)")
ax[3].plot([], [], "-", color=BLUE, label="near-linear")
ax[3].legend(loc="best", fontsize=9)

fig.suptitle("모기 궤적 EDA — 급기동(높은 turn-rate)이 잦아 선형 외삽이 어긋난다",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"[saved] {OUT}  ({OUT.stat().st_size} bytes)")

# 다운로드 폴더에도 복사(있으면)
dl = Path.home() / "Downloads"
if dl.exists():
    import shutil
    shutil.copy(OUT, dl / "mosquito_eda.png")
    print(f"[copied] {dl / 'mosquito_eda.png'}")

print(f"\n[stats] speed med={np.median(speed):.3f}  acc med={np.median(acc):.3f}  "
      f"turn med={med:.1f}deg p90={p90:.1f}deg")
