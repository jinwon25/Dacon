# 모기 비행 궤적 예측 AI 경진대회

데이콘 **모기 비행 궤적 예측 AI 경진대회** 솔루션 정리.
40ms 간격 11점의 3D 좌표를 입력으로, 마지막 관측 +80ms 좌표를 예측한다. 평가 지표 **R-Hit@1cm** (1cm 이내 = hit).

> 진행 중 — 2026-05-26 시점 LB 0.6912, plateau 0.6888 돌파 (Neural ODE 도입). 마감 2026-06-01.

## 결과 (LB 진행)

| 시점 | 후보 | OOF | LB | 비고 |
|---|---|---:|---:|---|
| 2026-05 초 | v10 단순 candidate-selection | - | 0.6306 | baseline 시작 |
| 2026-05-20 | v77 BiGRU | 0.6633 | 0.6750~ | NN 패러다임 안착 |
| 2026-05-23 | v98 5-way blend | 0.6760 | 0.6882 | 변환률 +0.0122 |
| 2026-05-24 | v106 DE15w | 0.6770 | 0.6888 | plateau 진입 |
| 2026-05-25 | v112_v107_diverse | 0.6768 | 0.6888 | 변환률 +0.0120 (당시 최고) |
| **2026-05-26** | **v122c (Neural ODE blend)** | **0.6769** | **0.6912** ★ | **변환률 +0.0143, plateau 돌파** |

핵심 — **kalman residual baseline에 묶인 framework는 모델간 corr~0.99 floor에 갇혀 LB 0.6888이 천장**이었다. v120(Neural ODE: position+velocity 6D state, RK4 단계 적분) family를 pool에 추가하면서 paradigm diversity (L2 distance ~2.2mm vs 기존 0.15mm) 확보 → 변환률 +0.0143로 plateau 정확히 돌파.

## 문제 정의

- 입력: `open/train/*.csv`, `open/test/*.csv` (각 10,000개, 11×(timestep_ms, x, y, z))
- 정답: `open/train_labels.csv` — 마지막 관측 +80ms의 `(x, y, z)`
- 제출: `sample_submission.csv` 형식
- 지표: R-Hit@1cm — `‖pred - y‖ < 1cm` 비율

원본 데이터는 `open/` 아래 두고 Git에는 포함하지 않는다.

## 핵심 접근 — 두 paradigm pool + DE blend

### Pool A — Kalman 잔차 framework (v77~v118)
- canonical local frame (마지막 속도 벡터 yaw 정렬) + kalman residual을 target으로 학습
- backbones: BiGRU(v77/v90/v94), TCN(v42), Transformer(v107), MDN-WTA(v109)
- boundary refinement (v91/v94/v97/v101/v103/v104/v111): cap 적용 residual 보정
- yaw aug + y-mirror로 회전 invariance
- pool 안 멤버끼리 corr_3d ~0.99 floor (residual baseline 공유 영향)

### Pool B — Neural ODE family (v120~v126, 2026-05-26 도입)
- target = `y - last_obs` (kalman 미사용 → 완전히 다른 base)
- 6D state (pos, vel) + learned damping + neural acceleration field
- RK4 4-eval 단일 80ms step (v120) / 2-step (v120_n2) / 4-step
- 변종: hidden=128/latent=128 (v120_big), rfft magnitude+phase scalar feature (v126)
- boundary refinement (v121): v111 패턴 동일 — cap 1.0/1.5
- 단독 OOF 0.66 수준이지만 pool에 합치면 **DE blender가 ~40% weight를 부여** → paradigm 진정성 검증

### Blend layer
- `v110_de_ensemble.py`: scipy.optimize.differential_evolution으로 softmax weights 학습 (전체 pool)
- `v112_conservative_blend.py`: top-7 OOF + force-include 핵심 paradigm 멤버
- **v122c**: v106 + v112 + v120 family conservative blend → OOF 0.6769, LB 0.6912

## EDA 핵심 발견 (`reports/eda_post_v122c.md`)

- **Oracle min(v112, v120) = 0.6989** — per-sample 완벽 selector 천장
- v122c가 oracle hit의 96.72% 캡처 (단순 selector 추가 lift 거의 0)
- 메타특징만으로 only_v112 vs only_v120 식별 어려움 (Δz < 0.14) → **v125 disagreement selector AUC 0.5562 (dead)**
- Neither subset (n=3011, 둘 다 miss) mean d=26mm — 새 paradigm 필요한 영역
- Hard subset (speed top20% × turn top20%, n=246) hit=0.354 — NN 약점

## 폴더 구조

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── NEXT_SESSION_BRIEF.md     # 다음 세션 인수인계, LB/OOF 현황
│
├── reports/                       # 분석 보고서
│   ├── eda_post_v122c.md          # 이번 sprint EDA
│   ├── sprint_d7_step_AB_summary.md  # ceiling/offset 진단
│   ├── ceiling_diagnosis.md/.json   # 데이터 천장 분석
│   ├── error_diagnosis.md          # 에러 분포/subset 진단
│   ├── hit_offset.md/.json         # global/body/speed offset 진단
│   ├── v118_*.md                   # STEP3 residual corr floor 진단
│   ├── v120_v121_v122c_sprint.md   # Neural ODE 도입 sprint
│   └── base_repr.md
│
├── scripts/                       # 학습/추론 단일 스크립트
│   ├── v23_train.py               # 핵심 공유 모듈 (load_data, kalman, scalar feats)
│   ├── v77_bigru.py               # BiGRU 베이스 backbone
│   ├── v90_yaw_mirror_aug.py      # yaw + y-mirror aug 패턴
│   ├── v107_deep_transformer.py   # Transformer backbone
│   ├── v109_mdn_wta.py            # MDN K-way WTA
│   ├── v110_de_ensemble.py        # DE blend 메인
│   ├── v111_boundary_on_v109.py   # MDN boundary refine
│   ├── v112_conservative_blend.py # 보수 blend (force-include)
│   ├── v118_aug_hit.py            # STEP3 residual corr 진단
│   ├── v120_neural_ode.py         # Neural ODE backbone (RK4)
│   ├── v121_boundary_on_v120.py   # v120 boundary refine
│   ├── v125_disagreement_selector.py  # selector (dead, 학습 결과 보존)
│   ├── v126_fft_neural_ode.py     # v120 + FFT scalar feature
│   ├── v127_neural_cde.py         # Neural CDE (kidger 2020, 미실행)
│   ├── v122d_blend_after_training.py  # 11-member full DE
│   ├── v122d_blend_quick.py       # 11-member 경량 DE (n_starts=2)
│   ├── eda_post_v122c.py          # 이번 sprint EDA
│   ├── eda_selector_probe.py      # selector EDA
│   ├── ceiling_diagnosis.py       # ceiling 진단
│   ├── hit_offset.py / hit_offset_cv.py
│   └── legacy/                    # 옛 보조 스크립트 (v5~v8 시절)
│
├── src/                           # 옛 selector/boundary 파이프라인 모듈
│
├── notebooks/                     # pipeline/reference 노트북
│
├── open/                          # 원본 데이터 (gitignore)
│   ├── train/, test/              # 각 10,000 CSV
│   ├── train_labels.csv
│   └── sample_submission.csv
│
├── cache/                         # 학습 중간 산출물 (gitignore) — *_state.npz, kalman.npz 등
├── outputs/                       # 옛 실험 산출물 (gitignore)
└── final_candidates/              # 락된 최고 제출 후보
    ├── submission_v106_DE15w_oof0.6770.csv         # LB 0.6888
    ├── submission_v112_v107_diverse_oof0.6768.csv  # LB 0.6888
    ├── submission_v117_*                            # selector 후보 2종
    └── submission_v122c_v121diverse_oof0.6769.csv  # LB 0.6912 ★
```

## 재현 — Neural ODE family 학습부터 v122c blend까지

Python 3.11 (Colab T4 권장, CPU도 가능, 1 모델당 15~30분).

```bash
pip install -r requirements.txt
```

```bash
# 1. 베이스 paradigm pool (kalman residual framework) — 시간 매우 김
python scripts/v77_bigru.py --mode full
python scripts/v90_yaw_mirror_aug.py --mode full
python scripts/v94_boundary_on_v90.py        # boundary 변종
python scripts/v107_deep_transformer.py --mode full
python scripts/v109_mdn_wta.py --mode K8
python scripts/v111_boundary_on_v109.py --tag wm_cap10/15

# 2. Neural ODE paradigm
python scripts/v120_neural_ode.py --mode full --tag full
python scripts/v120_neural_ode.py --mode full --tag big_full --latent_dim 128 --hidden 128
python scripts/v126_fft_neural_ode.py --mode full
python scripts/v121_boundary_on_v120.py --cap 10
python scripts/v121_boundary_on_v120.py --cap 15

# 3. 11-member DE blend
python scripts/v122d_blend_quick.py
# → submission_v122d_quick_oof0.67xx.csv, submission_v122e_quick_oof0.67xx.csv
```

`final_candidates/` 안의 csv가 LB 실측된 후보. CHECKSUMS.txt로 무결성 확인.

## 다음 카드 (`docs/NEXT_SESSION_BRIEF.md` 상세)

| 우선 | 카드 | 잠재 LB lift |
|---|---|---|
| 1 | v122d/e quick blend (현재 진행 중) | +0.001~+0.003 |
| 2 | **v127 Neural CDE** (torchcde, kidger 2020) | +0.002~+0.005 (큰 paradigm) |
| 3 | Frenet-frame coordinate transformation | +0.001~+0.003 |
| 4 | per-axis (x/y/z) 분리 v120 변종 | +0.001~+0.003 |
| ❌ | 메타 only selector | dead, AUC 0.5562 |

## 환경

- Python 3.11 (CPU 16 thread / Colab T4 GPU)
- PyTorch 2.7+ (CPU 빌드 OK), NumPy 2.x, scipy, scikit-learn, lightgbm
- Neural CDE 카드용: `pip install torchcde`

## 커밋 정책

- 포함: README, 코드, reports, NEXT_SESSION_BRIEF, final_candidates의 csv
- 제외 (gitignore): `open/` 원본 데이터, `cache/`, `outputs/`, `archive/`, `.npz`, `logs_*.txt`
- 노트북은 커밋 전 출력 셀 제거 권장
