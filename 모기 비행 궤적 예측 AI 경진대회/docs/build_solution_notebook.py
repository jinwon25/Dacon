"""build_solution_notebook — 모기 대회 2등 솔루션 '통합 코드' 노트북(.ipynb) 생성기.

실제 src/*.py 에서 핵심 정의를 라인 단위로 그대로 추출(전사 오류 방지)하여
하나의 통합 코드 노트북으로 조립한다. 방법론 설명/서술은 첨부 PDF가 담당하므로
노트북에는 탐색용 한 줄 섹션 헤더만 두고 코드 위주로 구성한다.

  §1 환경·데이터·공통 코어 (Kalman / canonical frame / scalar features)
  §2 Pool A : Kalman 잔차 GRU
  §3 Pool B : Neural ODE (RK4)
  §4 Pool C : Frenet 3D-프레임 + control-head
  §5 CREE 회전물리 멤버 (HyperPhysics, 공개 baseline 포팅)
  §6 DE 블렌드(base) + 최종 α 주입 (v157)
  §7 빠른 재현 & 검증 (frozen inputs → 최종 제출 재생성, <1초, 실행 가능)

§1~§6 은 방법론을 코드로 보여주는 부분이며 전체 from-scratch 학습은 src/ 스크립트로
수행(멤버 ~40개, 15~20h). §7 만 단독 실행으로 최종 결과를 검증한다.

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

# ---- 타이틀 (최소) ----
cells.append(md(
    "# 모기 비행 궤적 예측 — Private 2위(0.703151) 통합 솔루션 코드",
    "",
    "지표 R-Hit@1cm. 모델 정의는 실제 제출 코드(`src/*.py`)에서 그대로 가져왔습니다. "
    "**방법론 상세는 첨부 PDF 참조.** 전체 학습은 `src/` 스크립트로(멤버 ~40개), "
    "§7 만 단독 실행으로 최종 결과를 재현·검증합니다.",
))

# ---- §1 ----
cells.append(md("## §1. 환경 · 데이터 · 공통 코어 — Kalman / canonical frame / scalar features  `(src/v23_train.py)`"))
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
    "# 경로: 노트북(notebooks/) 기준 프로젝트 루트 자동 탐색\n"
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
cells.append(code(grab("v23_train.py", 68, 70) + "\n\n\n" + grab("v23_train.py", 76, 104)))
cells.append(code(grab("v23_train.py", 110, 144)))
cells.append(code(grab("v23_train.py", 150, 245)))
cells.append(code(grab("v23_train.py", 251, 266) + "\n\n\n" + grab("v23_train.py", 272, 293)))

# ---- §2 ----
cells.append(md("## §2. Pool A — Kalman 잔차 GRU (baseline paradigm)  `(src/v23_train.py)`"))
cells.append(code(grab("v23_train.py", 299, 331)))
cells.append(code(grab("v23_train.py", 337, 448)))

# ---- §3 ----
cells.append(md("## §3. Pool B — Neural ODE (RK4, 1차 돌파 LB 0.6912)  `(src/v120_neural_ode.py)`"))
cells.append(code(grab("v120_neural_ode.py", 65, 83)))
cells.append(code(grab("v120_neural_ode.py", 88, 180)))
cells.append(code(grab("v120_neural_ode.py", 186, 200)))

# ---- §4 ----
cells.append(md("## §4. Pool C — Frenet 3D-프레임 + control-head (2차 돌파 LB 0.697)  `(src/v131_paradigm_variants.py, src/v135_control_head.py)`"))
cells.append(code(grab("v131_paradigm_variants.py", 54, 110)))
cells.append(code(grab("v131_paradigm_variants.py", 115, 151)))
cells.append(code(grab("v135_control_head.py", 42, 62)))

# ---- §5 ----
cells.append(md(
    "## §5. CREE 회전물리 멤버 (HyperPhysics, 최종 돌파 LB 0.7016→0.7022)  `(src/v148_cree_xy2.py)`",
    "",
    "<sub>공개 Dacon 코드공유 baseline의 모델 구조를 포팅(구조 보존), 가중치 차용 없이 train만으로 from-scratch 학습. "
    "코드네임 CREE는 동명 참가자와 무관. 출처·규칙 준수는 첨부 PDF 참조.</sub>",
))
cells.append(code(grab("v148_cree_xy2.py", 26, 104)))
cells.append(code(grab("v148_cree_xy2.py", 106, 181)))
cells.append(code(grab("v148_cree_xy2.py", 184, 216)))

# ---- §6 ----
cells.append(md(
    "## §6. DE 블렌드(base) + 최종 α 주입 — v157  `(src/v148_reblend.py, src/v157_final_submission.py)`",
    "",
    "```python",
    "cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)        # CREE 회전물리 3-앙상블",
    "base      = DE(differential_evolution) conservative 블렌드 (OOF 0.6831, 결정론적)",
    "v157_a040 = 0.60 * base + 0.40 * cree_ens3                # Public 0.7022",
    "v157_a045 = 0.55 * base + 0.45 * cree_ens3                # Public 0.7022  →  Private 0.703151",
    "```",
))

# ---- §7 ----
cells.append(md("## §7. 빠른 재현 & 검증 (실행 가능, <1초, GPU 불필요)"))
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
