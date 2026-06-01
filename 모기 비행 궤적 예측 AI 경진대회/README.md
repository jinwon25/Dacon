# 모기 비행 궤적 예측 AI 경진대회

데이콘에서 주최한 [**모기 비행 궤적 예측 AI 경진대회**](https://dacon.io/) 솔루션.

## 최종 결과

> **Private 리더보드 2위 (0.703151).** 단순 candidate-selection 베이스라인(LB 0.6306) 대비 **+0.073** 절대 상승.

| | OOF | LB | 비고 |
|---|---:|---:|---|
| 시작 베이스라인 (candidate-selection) | - | 0.6306 | 첫 제출 |
| Kalman 잔차 NN 풀 (BiGRU 등) | 0.6770 | 0.6888 | plateau (멤버 corr ~0.99) |
| Neural ODE 도입 | 0.6769 | 0.6912 | 1차 돌파 |
| Frenet/control-head paradigm | 0.6805 | 0.697 | 2차 돌파 |
| CREE 회전물리 멤버 주입 | 0.6808 | 0.7016 | 3차 돌파 |
| **최종 3-CREE 앙상블 주입 (v157)** | — | **Public 0.7022 / Private 0.703151** | **2위** |

평가 지표: **R-Hit@1cm** — 예측과 정답의 유클리드 거리가 1cm 이내인 샘플의 비율 (높을수록 좋음).

최종 선택 2슬롯: `submissions/submission_v157_ens3a0.40_FINAL.csv`, `submissions/submission_v157_ens3a0.45_FINAL.csv` (둘 다 Public 0.7022 → Private 0.703151).

---

## 문제 정의

- **입력**: 40ms 간격 11개 시점의 3D 좌표 `(x, y, z)` (구간 −400ms ~ 0ms).
- **출력**: 마지막 관측 시점 **+80ms** 의 3D 좌표 `(x, y, z)`.
- **데이터**: `data/train/` 10,000 궤적 + `data/train_labels.csv`, `data/test/` 10,000 궤적.
- **지표**: R-Hit@1cm — 1cm 이내 hit 비율.
- **핵심 성질**: 1cm 임계값의 **이진 hit 지표**라, 평균 오차(mm) 최소화보다 **1cm 경계에 걸친 샘플을 넘기는 것**이 점수를 좌우한다. 이것이 "단일 모델 정확도 < 앙상블 다양성" 전략을 결정했다.

---

## 솔루션 아키텍처

### 직교 메커니즘 다양성의 앙상블

이 대회의 본질적 난점은 **데이터 천장**이었다. 40여 개 모델이 전부 같은 **Kalman 잔차 base** 위에서 학습돼 서로의 예측이 **상관 ~0.99** 로 묶였고, LB 0.6888에서 막혔다. 점수를 움직인 모든 돌파는 **근본적으로 다른 예측 메커니즘(paradigm)을 새로 도입**한 순간이었다.

```
                  ┌─ Pool A: Kalman 잔차 프레임 (BiGRU/TCN/Transformer/MDN)
   DE 블렌드 ─────┼─ Pool B: Neural ODE (위치+속도 6D 상태, RK4 적분)
   (base, OOF      ├─ Pool C: Frenet 3D-프레임 ODE + control-head 적분
    0.6831)        └─ (각 paradigm의 boundary refinement 변종)
        │
        │   v157 = (1−α)·base + α·CREE_ens3      ← 최종 수동 주입 (α=0.40, 0.45)
        ▼
   CREE 회전물리 3-앙상블 (Rodrigues 회전 turn-rate 물리, base와 2.82mm 직교)
```

### 핵심 설계 결정

1. **paradigm diversity가 LB를 움직인다.** 같은 base의 변종은 corr ~0.99로 새 정보 없음. Kalman→ODE→Frenet→회전물리 순으로 **직교한 base를 추가**할 때마다 plateau를 넘었다.
2. **Neural ODE (1차 돌파).** 타깃을 Kalman 무관 `y − last_obs`로 바꾸고 6D 상태(위치·속도)를 RK4로 적분 → base 풀과 L2 ~2.2mm 직교 → 0.6888 → 0.6912.
3. **Frenet 3D-프레임 (2차 돌파).** tangent(속도)·normal(가속도)·binormal 직교 프레임에서 예측 → z 처리가 근본적으로 달라 decorrelation 최대 → 0.697.
4. **CREE 회전물리 + 수동 α 주입 (3차 돌파).** 공개 Dacon baseline(HyperPhysics 회전물리)을 decorrelated 멤버로 포팅. DE가 OOF-greedy라 직교 멤버를 과소평가하므로, **수동 α로 over-convert** → 0.7016 → 0.7022.

### 결정적 인사이트 — "OOF는 직교 멤버의 LB 프록시가 아니다"

- OOF 최고 블렌드(`v148blend`, OOF **0.6831**, CREE weight 0.082) → LB 0.6996.
- OOF가 **더 낮은** 블렌드(raw CREE 25% **수동 주입**, OOF 0.6808) → LB **0.7016**.
- 즉 **더 낮은 OOF + 더 직교한 변종이 LB를 +0.0020 이긴다.** 1cm hit 지표는 OOF blend CV가 못 잡는 다양성을 보상한다. α를 0.25→0.30→0.40으로 키우며 Public이 0.7016→0.7020→0.7022 단조 상승.

### 시도했지만 효과 없음

| 기법 | 결과 |
|---|---|
| Disagreement selector (per-sample 모델 선택) | DEAD — route-acc 0.17 ≈ 무작위 |
| Mode-seeking / geometric-median 집계 | Δ ≤ 0 (active 멤버 동질 군집) |
| IMM / analytic Constant-Turn 필터 | 0.24~0.55 < naive linear 0.58 |
| Neural CDE (torchcde) | DEAD — OOF 0.2768, 학습 실패 |
| Flow/SONODE 추가 주입 (4-mechanism) | Public 0.6994 < 순수 CREE 0.7022, 폐기 |
| 같은 frenet 프레임 encoder 변종 (Transformer/LRU/TCN) | DE weight 0 (포화) |
| pseudo-label | OOF 과적합, LB 변환률 붕괴 |

---

## 폴더 구조

```text
.
├── README.md                     # 프로젝트 요약 (이 파일)
├── requirements.txt              # Python 의존성 (버전 고정)
├── .gitignore
│
├── src/                          # 학습/블렌딩/진단 코드
│   ├── v23_train.py              # 공유 모듈 (load_data, kalman, scalar feats)
│   ├── v120_neural_ode.py        # Neural ODE backbone (RK4)
│   ├── v131_paradigm_variants.py # Frenet/GRU-encoder ODE
│   ├── v135_control_head.py      # control-head analytic-integrator
│   ├── v148_cree_xy2.py          # CREE 회전물리 멤버 (공개 baseline 포팅)
│   ├── v148_reblend.py           # DE 블렌드 (base 생성)
│   ├── v157_final_submission.py  # 최종 제출 생성 (base + 3-CREE → v157)
│   └── legacy/                   # 옛 실험/탐색 스크립트
│
├── submissions/                  # 최종 제출 + 재현 패키지
│   ├── submission_v157_ens3a0.40_FINAL.csv   # ★ Private 0.703151 후보
│   ├── submission_v157_ens3a0.45_FINAL.csv   # ★ Private 0.703151 후보
│   ├── rebuild.py                # 자급식 재현 (inputs/만으로 최종 재생성+검증)
│   ├── inputs/                   # 재현 입력 (base + 3-CREE 예측)
│   ├── CHECKSUMS.sha256          # 무결성
│   └── historical/              # 과거 LB-실측 후보 (0.6770~0.697)
│
├── docs/                         # 솔루션 문서 + 작업 로그
│   ├── SOLUTION.md               # 솔루션 상세 (PDF 변환용)
│   ├── WORK_LOG.md               # 시간순 작업 로그
│   └── reports/                  # 분석 보고서 (EDA, ceiling 진단 등)
│
├── notebooks/                    # 파이프라인/참조 노트북
│
└── data/                         # 대회 원본 데이터 + 캐시 (gitignore)
    ├── train/  test/  train_labels.csv  sample_submission.csv
    └── cache/                    # 학습 산출물 (*_state.npz 등)
```

커밋 기준:
- 포함: 코드(`src/`), README, `docs/`, requirements, `submissions/` 전체(최종 제출 + 재현 입력)
- 제외: `data/`(대회 원본 데이터·캐시), `outputs/`(로그) — 재생성 가능하거나 라이선스 이슈

---

## 재현 방법

### 0. 환경 설치 (Python 3.11, CPU만으로 가능)

```bash
pip install -r requirements.txt
```

### 1. 빠른 재현 — 최종 제출 검증 (권장, <1초, GPU 불필요)

`submissions/inputs/` 의 frozen 예측(base + 3-CREE)만으로 최종 제출 2개를 재생성하고 원본과 일치(오차 < 0.001mm)를 검증한다.

```bash
cd submissions && python rebuild.py
# -> rebuilt_*.csv 생성 + 원본과 MATCH 확인
```

최종 레시피:

```
cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)            # CREE 회전물리 3-앙상블
v157_a040 = 0.60 * base + 0.40 * cree_ens3                    # Public 0.7022
v157_a045 = 0.55 * base + 0.45 * cree_ens3                    # Public 0.7022
```

### 2. 전체 재현 — 학습부터 (CPU/GPU, 멤버당 15~30분)

대회 데이터를 `data/` 에 배치 (`data/train/`, `data/test/`, `data/train_labels.csv`, `data/sample_submission.csv`).

```bash
# (a) base paradigm 풀 — Kalman 잔차 / Neural ODE / Frenet / control-head 멤버 학습
#     (각 스크립트가 data/cache/*_state.npz 산출; 전체 목록은 docs/SOLUTION.md 참고)
python src/v77_bigru.py --mode full
python src/v107_deep_transformer.py --mode full
python src/v109_mdn_wta.py --mode K8
python src/v120_neural_ode.py --mode full --tag full
python src/v131_paradigm_variants.py --frame frenet --encoder gru --mode full --tag frenet_gru
python src/v135_control_head.py --frame frenet --order accel --mode full --tag ch_frenet
#  ... + 각 paradigm의 boundary refinement 변종

# (b) CREE 회전물리 3-앙상블 멤버 (공개 baseline 포팅)
python src/v148_cree_xy2.py --mode full --tag xy2                    # dirnet, seed42
python src/v148_cree_xy2.py --mode full --tag xy2s1 --seed 1         # dirnet, seed1
python src/v148_cree_xy2.py --mode full --tag xy2h3 --heading 3step  # 3step-heading

# (c) base DE 블렌드 (seed 고정 → 결정론적) → data/submission_v148blend_oof0.6831.csv
python src/v148_reblend.py

# (d) 최종 제출 생성 (base + 3-CREE → v157)
python src/v157_final_submission.py
```

DE 블렌드(`scipy.differential_evolution`)와 멤버 학습은 모두 seed 고정으로 **결정론적**이다. 단계 (a)의 전체 멤버 목록과 역할은 `docs/SOLUTION.md` 참고.

### 재현 자원 / 소요 시간

| 경로 | 자원 | 시간 |
|---|---|---|
| 빠른 재현 (`submissions/rebuild.py`) | CPU, RAM 2GB | < 1초 |
| base 블렌드만 (`v148_reblend.py`, 캐시 사용) | CPU 16-thread | ~5–15분 |
| 전체 재학습 (멤버 ~40개) | CPU 16-thread 또는 Colab T4/L4 | 약 15–20시간 (멤버당 15–30분, 순차) |

> **공식 재현 코드** = 위 "재현 방법"에 문서화된 명령(`submissions/rebuild.py` 및 `src/v157_final_submission.py`·`v148_reblend.py`·멤버 학습 스크립트)이며, 모두 오류 없이 실행됨을 확인했다. `src/legacy/` 는 대회 중 탐색했던 보조·폐기 스크립트(연구 과정 보존용)로 공식 재현 경로에 포함되지 않는다.

---

## 대회 규칙 준수

- **사전학습 모델**: 별도 사전학습 가중치 미사용. 모든 모델을 본 대회 train으로 from-scratch 학습. ✓
- **CREE 멤버 출처**: 대회 기간 중 Dacon **코드 공유 게시판에 공개되었던** baseline(HyperPhysics 회전물리)의 모델 구조를 참고(규칙 8조 B항 공개 코드 공유 허용). 해당 게시물은 **현재 삭제되어 링크 제시 불가**하나, 메커니즘은 교과서적 물리(Rodrigues 회전 + EMA 필터). **공개 모델 구조 코드를 포팅(구조 보존)** 해 우리 CV·데이터에 연결, **가중치 차용 없이 train만으로 from-scratch 학습**. 앙상블·주입·전체 파이프라인은 독자 기여. ✓
- **원격 API 모델** (OpenAI, Gemini 등): 미사용. 모두 로컬 실행. ✓
- **test 데이터 학습 금지**: 학습 fit에 test 미포함, pseudo-label 미사용 (실험 후 폐기). ✓
- **외부 데이터**: 미사용. ✓
- **시드 고정 재현 가능**: 멤버 학습 + DE 블렌드 + 최종 주입 전부 결정론적. ✓

---

## 사용 라이브러리

| 라이브러리 | 버전 | 라이선스 |
|---|---|---|
| PyTorch | 2.7.1+cpu | BSD |
| NumPy | 2.2.6 | BSD-3 |
| pandas | 2.2.3 | BSD-3 |
| SciPy | 1.15.3 | BSD-3 |
| scikit-learn | 1.6.1 | BSD-3 |
| LightGBM | 4.6.0 | MIT |

모두 오픈소스 + 상업적 이용 허용 라이선스.

## 환경

- Python: 3.11.2
- OS: Windows 11 (10.0.26200) — Linux/Mac 호환. 학습은 Colab T4/L4 GPU에서도 수행.
- 최종 블렌드/제출 재현은 **CPU만으로 충분** (GPU 불필요).
