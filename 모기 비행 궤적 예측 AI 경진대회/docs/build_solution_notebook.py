"""build_solution_notebook — 모기 대회 2등 솔루션 '통합 코드' 노트북(.ipynb) 생성기.

실제 src/*.py 에서 핵심 정의를 라인 단위로 그대로 추출(전사 오류 방지)하여
하나의 통합 노트북으로 조립한다. 구성:

  §0 개요/방법론 (markdown)
  §1 환경·데이터·공통 코어 (Kalman / canonical frame / scalar features)   — 실제 코드
  §2 Pool A : Kalman 잔차 GRU                                             — 실제 코드
  §3 Pool B : Neural ODE (RK4)                                            — 실제 코드
  §4 Pool C : Frenet 3D-프레임 + control-head                             — 실제 코드
  §5 CREE 회전물리 멤버 (HyperPhysics, 공개 baseline 포팅)                — 실제 코드
  §6 DE 블렌드(base) + 최종 α 주입 (v157)                                  — 레시피
  §7 빠른 재현 & 검증 (frozen inputs → 최종 제출 재생성, <1초)            — 실행 가능

§1~§6 은 방법론을 코드로 보여주는 '아키텍처 레퍼런스'이며 전체 from-scratch 학습은
src/ 스크립트로 수행(멤버 ~40개, CPU/GPU 15~20h). §7 만 단독 실행으로 최종 결과를 검증한다.

사용: python docs/build_solution_notebook.py
출력: notebooks/mosquito_2nd_place_solution.ipynb
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "mosquito_2nd_place_solution.ipynb"


def grab(rel: str, start: int, end: int) -> str:
    """src/<rel> 의 start..end 줄(1-indexed, inclusive)을 그대로 반환."""
    lines = (SRC / rel).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end])


def md(*parts: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(parts)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []

# ============================================================ §0
cells.append(md(
    "# 모기 비행 궤적 예측 AI 경진대회 — Private 2위(0.703151) 통합 솔루션 코드",
    "",
    "> 40ms 간격 11개 시점의 3D 좌표로 마지막 관측 **+80ms** 좌표를 예측. 지표 **R-Hit@1cm**(1cm 이내 hit 비율, 높을수록 좋음).",
    "",
    "이 노트북은 솔루션 전체를 **하나의 코드로 통합**한 것입니다. 모델 정의는 실제 제출 코드"
    "(`src/*.py`)에서 그대로 가져왔습니다.",
    "",
    "| 구분 | 점수 | 비고 |",
    "|---|---:|---|",
    "| 시작 베이스라인 (candidate-selection) | 0.6306 | — |",
    "| Kalman 잔차 NN 풀 | LB 0.6888 | 멤버 corr ~0.99 plateau |",
    "| + Neural ODE | LB 0.6912 | 1차 돌파 |",
    "| + Frenet/control-head | LB 0.697 | 2차 돌파 |",
    "| + CREE 회전물리 (수동 α 주입) | LB 0.7016→**0.7022** | 3차 돌파 |",
    "| **최종 v157 (Public 0.7022)** | **Private 0.703151** | **2위** |",
    "",
    "---",
    "## 핵심 아이디어 — '직교 메커니즘 다양성'의 앙상블",
    "",
    "이 대회의 본질적 난점은 **데이터 천장**이었습니다. 40여 개 모델을 쌓아도 LB 0.6888에서 막혔는데, "
    "원인은 전부 같은 **Kalman 잔차 base** 위에서 학습돼 예측이 **상관계수 ~0.99**로 묶였기 때문입니다. "
    "앙상블 이득은 멤버 간 **직교성**에서 나오므로, 같은 메커니즘의 변종은 아무리 많아도 새 정보가 없습니다.",
    "",
    "점수를 움직인 모든 돌파는 **근본적으로 다른 예측 메커니즘(paradigm)을 새로 도입**한 순간이었습니다:",
    "",
    "```",
    "                  ┌─ Pool A: Kalman 잔차 프레임 (BiGRU/TCN/Transformer/MDN)",
    "   DE 블렌드 ─────┼─ Pool B: Neural ODE (위치+속도 6D 상태, RK4 적분)",
    "   (base, OOF      ├─ Pool C: Frenet 3D-프레임 ODE + control-head 닫힌형 적분",
    "    0.6831)        └─ (각 paradigm의 boundary refinement 변종)",
    "        │",
    "        │   v157 = (1−α)·base + α·CREE_ens3      ← 최종 수동 주입 (α=0.40, 0.45)",
    "        ▼",
    "   CREE 회전물리 3-앙상블 (Rodrigues 회전 turn-rate 물리, base와 2.82mm 직교)",
    "```",
    "",
    "### 결정적 인사이트 — \"OOF는 직교 멤버의 LB 프록시가 아니다\"",
    "- OOF 최고 블렌드(`v148blend`, OOF 0.6831, CREE weight 0.082) → LB 0.6996",
    "- OOF가 **더 낮은** 블렌드(raw CREE 25% **수동 주입**, OOF 0.6808) → LB **0.7016**",
    "- 즉 **더 낮은 OOF + 더 직교한 변종이 LB를 +0.0020 이긴다.** 1cm hit 지표는 OOF blend CV가 "
    "못 잡는 다양성을 보상한다. α를 0.25→0.30→0.40 으로 키우며 Public 0.7016→0.7020→0.7022 단조 상승.",
    "",
    "> **규칙 준수 — 회전물리(CREE) 멤버 출처**: 내부 코드네임 `CREE`는 **동명의 본 대회 참가자와 무관**합니다. "
    "대회 기간 중 Dacon **코드 공유 게시판에 공개되었던** 회전 기반 turn-rate 물리 baseline(HyperPhysics 계열)의 "
    "모델 구조를 참고했습니다(규칙 8조 B항 공개 코드 공유 허용). 해당 게시물은 **현재 삭제되어 링크 제시 불가**하나 "
    "메커니즘은 교과서적 물리(Rodrigues 회전 + EMA 필터)입니다. **공개 모델 구조 코드를 포팅(구조 보존)** 해 우리 "
    "5-fold CV·데이터에 연결했고, **가중치 차용 없이 train만으로 from-scratch 학습**했습니다. test 학습·외부데이터·"
    "원격 API 모두 미사용, 시드 고정 재현 가능합니다.",
))

# ============================================================ §1
cells.append(md(
    "---",
    "## §1. 환경 · 데이터 · 공통 코어",
    "",
    "모든 paradigm이 공유하는 토대: **Kalman CV base 예측**, **canonical local frame**(마지막 속도 벡터로 yaw 정렬), "
    "**scalar features**(속도/가속도/jerk/noise 등), **시퀀스 텐서**. (출처: `src/v23_train.py`)",
    "",
    "> 데이터는 `data/train/`, `data/test/`, `data/train_labels.csv`, `data/sample_submission.csv` 에 배치. "
    "전체 학습은 GPU 권장(멤버당 15~30분).",
))
cells.append(code(
    "from __future__ import annotations\n"
    "import gc, glob, os, random, time\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n"
    "from typing import Tuple\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "from scipy.interpolate import CubicSpline\n"
    "from scipy.signal import savgol_filter\n"
    "from sklearn.model_selection import KFold\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from tqdm.auto import tqdm\n\n"
    "# --- 경로: 노트북(notebooks/) 기준 프로젝트 루트 탐색 ---\n"
    "def find_root(start: Path) -> Path:\n"
    "    p = start.resolve()\n"
    "    for q in [p, *p.parents]:\n"
    "        if (q / 'submissions' / 'inputs').exists() or (q / 'data').exists():\n"
    "            return q\n"
    "    return p\n\n"
    "ROOT = find_root(Path.cwd())\n"
    "DATA_DIR = ROOT / 'data'\n"
    "CACHE_DIR = ROOT / 'data' / 'cache'; CACHE_DIR.mkdir(parents=True, exist_ok=True)\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "os.environ['PYTHONHASHSEED'] = '0'; random.seed(0); np.random.seed(0); torch.manual_seed(0)\n"
    "print('ROOT =', ROOT, '| device =', device)"
))
cells.append(md("**상수 + 데이터 로딩**"))
cells.append(code(grab("v23_train.py", 68, 70) + "\n\n\n" +
                  grab("v23_train.py", 76, 104).replace("CACHE_DIR / ", "CACHE_DIR / ")))
cells.append(md("**Kalman CV base 예측** — σ_obs=0.3mm, σ_proc=1.0 (상수속도 모델)"))
cells.append(code(grab("v23_train.py", 110, 144)))
cells.append(md("**Scalar features** — 속도·가속도·jerk·직진성·turn·noise(poly2/savgol/LOO-spline) 등 + log1p"))
cells.append(code(grab("v23_train.py", 150, 245)))
cells.append(md("**Canonical frame(yaw 정렬) + 시퀀스 텐서**"))
cells.append(code(grab("v23_train.py", 251, 266) + "\n\n\n" + grab("v23_train.py", 272, 293)))

# ============================================================ §2
cells.append(md(
    "---",
    "## §2. Pool A — Kalman 잔차 GRU (baseline paradigm)",
    "",
    "canonical local frame에서 **Kalman 잔차**를 타깃으로 GRU 학습. 보조 헤드(F: last-obs 잔차, W: 다른 σ Kalman 잔차)와 "
    "**combo loss = euclid + 0.3·softhit**. yaw + y-mirror 증강으로 회전 불변. "
    "백본을 BiGRU/TCN/Transformer/MDN으로 바꿔 멤버를 다양화했지만 **이 풀 내부 corr ~0.99 → LB 0.6888 천장**. "
    "(출처: `src/v23_train.py`, `src/v77_bigru.py` 등)",
))
cells.append(md("**모델 + 손실**"))
cells.append(code(grab("v23_train.py", 299, 331)))
cells.append(md("**5-fold OOF 학습 러너** (Pool A) — 멤버마다 OOF/test 예측을 생성해 블렌드 풀에 적재"))
cells.append(code(grab("v23_train.py", 337, 448)))

# ============================================================ §3
cells.append(md(
    "---",
    "## §3. Pool B — Neural ODE (1차 돌파, LB 0.6912)",
    "",
    "타깃을 Kalman과 무관한 `y − last_obs`로 바꿔 **완전히 다른 base**. **6D 상태(위치·속도)** 에 학습된 감쇠 + "
    "neural acceleration field를 두고 80ms 구간을 **RK4 4-eval**로 적분. 단독 OOF는 낮지만(0.66) base 풀과 "
    "**L2 ~2.2mm 직교** → DE 블렌더가 ~40% 가중치 부여 → 0.6888 → **0.6912** 돌파. (출처: `src/v120_neural_ode.py`)",
))
cells.append(md("**yaw 회전(시퀀스) + mirror 증강**"))
cells.append(code(grab("v120_neural_ode.py", 65, 83)))
cells.append(md("**Neural ODE 모델** — Encoder → latent, dp/dt=v, dv/dt=−damping⊙v+a(p,v,z,speed), RK4 적분"))
cells.append(code(grab("v120_neural_ode.py", 88, 180)))
cells.append(md("**손실** — huber + softhit + accel 정규화"))
cells.append(code(grab("v120_neural_ode.py", 186, 200)))

# ============================================================ §4
cells.append(md(
    "---",
    "## §4. Pool C — Frenet 3D-프레임 + control-head (2차 돌파, LB 0.697)",
    "",
    "**Frenet 프레임**: tangent(속도)·normal(가속도)·binormal 으로 만든 완전 3D 직교 프레임. yaw(xy)만 회전하는 "
    "v120과 달리 **z 처리가 근본적으로 달라** decorrelation 최대. **control-head**: RK4 대신 NN이 가속도(control)를 "
    "출력하고 `p = v₀·T + ½·a·T²` **닫힌형 적분** → 또 다른 오차 구조. conservative 블렌드 OOF 0.6807, 신규 멤버 "
    "가중 58% → LB **0.697** (변환률 +0.0165). (출처: `src/v131_paradigm_variants.py`, `src/v135_control_head.py`)",
))
cells.append(md("**좌표 프레임(yaw/Frenet) + 프레임 정합 mirror**"))
cells.append(code(grab("v131_paradigm_variants.py", 54, 110)))
cells.append(md("**GRU-encoder ODE** (프레임 무관; Frenet/yaw 어디서나 동작)"))
cells.append(code(grab("v131_paradigm_variants.py", 115, 151)))
cells.append(md("**control-head** — 닫힌형 적분 멤버 (`p = v_scale·v₀·T + ½·a·T²` [+ ⅙·j·T³])"))
cells.append(code(grab("v135_control_head.py", 42, 62)))

# ============================================================ §5
cells.append(md(
    "---",
    "## §5. CREE 회전물리 멤버 (최종 돌파, LB 0.7016 → 0.7022)",
    "",
    "**공개 Dacon 코드공유 baseline(HyperPhysics, 회전 기반 turn-rate 물리)** 을 우리 5-fold OOF 파이프라인에 포팅. "
    "Rodrigues 회전으로 속도 벡터를 회전 + 학습된 angular velocity(omega) + EMA 속도/가속도 필터 + world-up 프레임. "
    "우리 RK4 적분 계열과 메커니즘이 완전히 달라 **base와 2.82mm 직교**(우리 frenet 멤버끼리는 0.6mm). "
    "`dirnet(seed42) + dirnet(seed1) + 3step-heading` **3-앙상블**로 강화 — 내부 분산만 줄이고 교차-paradigm 직교성은 보존. "
    "(출처: `src/v148_cree_xy2.py`)",
    "",
    "> 출처·규칙 준수는 §0 상단 주석 참조. 아래는 포팅한 모델 구조(가중치 차용 없이 train만으로 학습).",
))
cells.append(md("**Sliding-window 데이터셋 + EMA + soft-hit 손실 + 회전-프레임 피처 추출**"))
cells.append(code(grab("v148_cree_xy2.py", 26, 104)))
cells.append(md("**HyperPhysics 모델** — ResBlock / PriorBiasedLinear / Rodrigues 회전 / turn-rate 물리"))
cells.append(code(grab("v148_cree_xy2.py", 106, 181)))
cells.append(md("**5-fold OOF 학습/추론** — `train_fold` + `predict_full`"))
cells.append(code(grab("v148_cree_xy2.py", 184, 216)))

# ============================================================ §6
cells.append(md(
    "---",
    "## §6. DE 블렌드(base) + 최종 α 주입 (v157)",
    "",
    "**base** = 위 paradigm 멤버들(~40~99개)의 OOF에 `scipy.differential_evolution`으로 softmax 가중을 학습한 "
    "conservative 블렌드(OOF 0.6831, 결정론적). **최종**은 여기에 CREE 3-앙상블을 **수동 α로 over-convert** 주입:",
    "",
    "```python",
    "cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)        # CREE 회전물리 3-앙상블",
    "v157_a040 = 0.60 * base + 0.40 * cree_ens3                # Public 0.7022",
    "v157_a045 = 0.55 * base + 0.45 * cree_ens3                # Public 0.7022  →  Private 0.703151",
    "```",
    "",
    "DE 블렌드의 핵심 로직(가중 학습 + per-axis 보정 + CREE-forward 주입)은 `src/v148_reblend.py`, "
    "최종 제출 생성은 `src/v157_final_submission.py` 입니다. base 재현은 전체 멤버 캐시(`data/cache/*_state.npz`)가 "
    "필요하므로, 아래 §7에서는 **frozen된 base + 3-CREE 예측**으로 최종 제출을 즉시 재현·검증합니다.",
))

# ============================================================ §7
cells.append(md(
    "---",
    "## §7. 빠른 재현 & 검증 (실행 가능, <1초, GPU 불필요)",
    "",
    "`submissions/inputs/` 의 frozen 예측(base + 3-CREE)만으로 최종 제출 2개를 재생성하고 "
    "원본(`submissions/submission_v157_*.csv`)과 일치하는지 검증합니다. (이 셀은 단독 실행 가능)",
))
cells.append(code(
    "def _xyz(p):\n"
    "    return pd.read_csv(p)[['x', 'y', 'z']].to_numpy()\n\n"
    "INP = ROOT / 'submissions' / 'inputs'\n"
    "SUB = ROOT / 'submissions'\n\n"
    "base = _xyz(INP / 'base_v148blend.csv')\n"
    "cree = [_xyz(INP / f'cree_{t}.csv') for t in ('xy2', 'xy2s1', 'xy2h3')]\n"
    "cree_ens3 = np.mean(cree, axis=0)\n"
    "ids = pd.read_csv(INP / 'base_v148blend.csv')['id']\n\n"
    "specs = [(0.40, 'submission_v157_ens3a0.40_FINAL.csv'),\n"
    "         (0.45, 'submission_v157_ens3a0.45_FINAL.csv')]\n"
    "for a, fn in specs:\n"
    "    pred = (1.0 - a) * base + a * cree_ens3\n"
    "    out = pd.DataFrame({'id': ids, 'x': pred[:, 0], 'y': pred[:, 1], 'z': pred[:, 2]})\n"
    "    ref_path = SUB / fn\n"
    "    if ref_path.exists():\n"
    "        d = np.linalg.norm(pred - _xyz(ref_path), axis=-1).max() * 1000\n"
    "        print(f'[alpha={a}] {fn}: max diff = {d:.5f} mm  -> {\"MATCH\" if d < 0.01 else \"MISMATCH\"}')\n"
    "    else:\n"
    "        print(f'[alpha={a}] {fn}: 원본 없음(스킵). 예측 생성만 수행.')\n"
    "print('\\n[done] 최종 제출 = submission_v157_ens3a0.40 / a0.45  (Private 0.703151, 2위)')"
))

# ============================================================ footer
cells.append(md(
    "---",
    "### 검증된 dead-end (재시도 가치 없음)",
    "",
    "| 시도 | 결과 |",
    "|---|---|",
    "| Disagreement selector (per-sample 모델 선택) | DEAD — route-acc 0.17 ≈ 무작위 |",
    "| Mode-seeking / geometric-median 집계 | Δ ≤ 0 (active 멤버 동질 군집) |",
    "| IMM / analytic Constant-Turn 필터 | 0.24~0.55 < naive linear 0.58 |",
    "| Neural CDE (torchcde) | DEAD — OOF 0.2768, 학습 실패 |",
    "| Flow/SONODE 추가 주입 (4-mechanism) | Public 0.6994 < 순수 CREE 0.7022, 폐기 |",
    "| 같은 frenet 프레임 encoder 변종(Transformer/LRU/TCN) | DE weight 0 (포화) |",
    "| pseudo-label | OOF 과적합, LB 변환률 붕괴 |",
    "",
    "**교훈**: 점수를 움직인 건 항상 *직교한 새 메커니즘*이었고, 신규 멤버는 **OOF-vs-TEST 예측 L2 일관성**을 "
    "필수 검증해야 한다. 자세한 내용은 첨부 PDF(솔루션 문서) 참조.",
))

for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[saved] {OUT}  ({OUT.stat().st_size} bytes, {len(cells)} cells)")
