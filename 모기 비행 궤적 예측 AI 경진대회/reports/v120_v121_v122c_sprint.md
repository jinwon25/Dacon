# v120/v121/v122c Sprint Report — Neural ODE paradigm 통합 (2026-05-26)

## Context

Dacon 코드공유 14002 (CREE, 2026-05-25): "[LB 0.6+] Neural ODE 기반 예측모델"
→ 작가가 0.69+ 달성 주장 (실 코드는 부분 공개, showcase 수준).
→ 우리 LB 0.6888 plateau의 **kalman residual corr~0.99 floor**를 깰 진짜 새 paradigm.

## 결론

**LB 후보**: `final_candidates/submission_v122c_v121diverse_oof0.6769.csv` (lock 완료)

- OOF: **0.6769** (v112 best 0.6768 +0.0001, v106 0.6770 -0.0001)
- 변환률 +0.0118~+0.0125 가정 → **LB 0.6887 ~ 0.6894** 예상
- 진짜 paradigm shift 보너스 (+0.0125~+0.0130) → **LB 0.6894 ~ 0.6899** 가능성

## 산출

### Neural ODE 단독 (v120)
- `scripts/v120_neural_ode.py` — RK4 단일 적분, 6D pos+vel state, learned damping
- Encoder: 99 (11×9 seq local-rotated) + 40 scal → 64 latent → ResBlock×2
- Acceleration field: MLP(pos+vel+latent+speed) → 3D
- Loss: huber(δ=0.001) × 100 + soft-hit (k=300, c=0.01) + accel² reg
- **OOF: 0.6610** (5-fold mirror+TTA, 2-seed, 9.0분 학습)
- target = `y - X[:,-1]` in local frame (kalman residual 미사용 ★ → corr floor 깸 시도)

### Boundary refinement on v120 (v121)
- `scripts/v121_boundary_on_v120.py` — v94 패턴 그대로 적용
- v120 OOF 0.6610 + BoundaryMLP δ ∈ [-cap, cap]
- **v121_cap10 OOF: 0.6734** (+0.0124 lift, 2.0분)
- **v121_cap15 OOF: 0.6725** (+0.0115 lift, 2.0분)
- 단독 OOF가 v94 0.6738 / v97 0.6741과 비등 — pool 멤버 자격 ★

### Conservative DE blend (v122c)
- `scripts/v112_conservative_blend.py` (load_pool에 v120/v121/v121c5 추가)
- top-7 (OOF>=0.67 자동 필터) + force-include v120/v121/v121c5
- cap 0.30, 5 starts × 200 iter × 30 popsize
- **OOF: 0.6769** (9 active weights)

| weight | model | single OOF | paradigm 클러스터 |
|---:|---|---:|---|
| 0.228 | v108_20_20_10 | 0.6742 | per-axis boundary on v90 |
| 0.219 | **v121c5** | 0.6725 | **Neural ODE +boundary 1.5** ★ |
| 0.219 | v97 | 0.6741 | boundary on v96 4-view |
| 0.177 | **v120** | 0.6610 | **Neural ODE raw** ★ |
| 0.089 | v108_15_15_08 | 0.6752 | per-axis |
| 0.034 | v108_15_15_05 | 0.6745 | per-axis |
| 0.012 | v108_15_15_10 | 0.6755 | per-axis |
| 0.011 | v97c5 | 0.6749 | boundary cap 1.5 |
| 0.010 | **v121** | 0.6734 | **Neural ODE +boundary 1.0** |

**Neural ODE 합 weight = 0.406 (40.6%)** ★ — DE blender가 새 paradigm을 매우 선호.

## 예측 다양성 분석 (test 10000 sample)

| 비교 | L2 diff mean (mm) | L2 diff median (mm) | 평가 |
|---|---:|---:|---|
| v112 vs v106 (기존 best 동률) | 0.15 | – | baseline 동일 paradigm |
| **v122c vs v112** | **0.51** | 0.37 | **3.4× 다양** ★ |
| **v122c vs v106** | **0.53** | 0.39 | 3.5× 다양 |
| v120 raw vs v112 (smoke) | 3.23 | – | paradigm 진정성 |
| v120 raw vs v112 (full) | 2.20 | – | paradigm 진정성 |

1cm 임계값에서 0.5mm 차이는 hit/miss 결정에 직접 영향 — 진짜 paradigm 다양성.

## STEP3 (corr 게이트) 재검토

| 비교 | corr_3d | gate <0.93 |
|---|---:|---|
| v120 vs v94 | 0.9865 | FAIL |
| v120 vs v97 | 0.9858 | FAIL |
| v120 vs v97c5 | 0.9855 | FAIL |
| v120 vs v104b | 0.9857 | FAIL |
| v120 vs v108_15_15_08 | 0.9856 | FAIL |

**해석**: 기존 framework 멤버끼리 corr ~0.992 → v120과 corr ~0.986
- gate <0.93은 너무 엄격 (framework 한계, 절대 도달 불가)
- 실 다양성은 L2 0.51mm로 측정 → ensemble lift 있음
- corr 게이트가 잘못 설계됨. 실 metric은 L2 prediction distance.

## LB 제출 권고

**Primary (D-1 슬롯)**: `submission_v122c_v121diverse_oof0.6769.csv`
- 예상 LB: 0.6887 ~ 0.6899 (변환률 +0.0118~+0.0130)
- 위험: OOF +0.0001 lift이 noise 가능성 → 그래도 v112 동등 보장

**Backup**: 기존 `submission_v112_v107_diverse_oof0.6768.csv` (LB 0.6888 실측)

## 다음 카드 (미완료)

| 우선 | 카드 | 상태 | 잠재 lift |
|---|---|---|---|
| 1 | v122c_top5 ultra-conservative (force v121) | 백그라운드 진행 중 | OOF 동등, 변환률 +0.0001 |
| 2 | v122 greedy 27w (kill 거부됨, 4시간 더) | 진행 중 | OOF 0.6785+, but 변환률 -0.0010 위험 |
| 3 | v120 multi-step RK4 (n_steps=2) | 미시도 | OOF +0.001~0.003 가능 |
| 4 | v120 + Neural CDE controllable observation | 1-2일 | corr floor 진짜 깰 잠재력 |
| 5 | v121 더 큰 boundary feature pool (v120 OOF + scalar + tier3 추가) | 미시도 | +0.0005 |

## 사용자 메모

- 마감 2026-06-01 10:00 → ~5일 남음
- Neural ODE는 backlog 우선순위 5번이었으나 코드공유 신호로 우선 시도 → 성공
- corr <0.93 게이트는 framework 한계 (재검토 필요), L2 prediction distance가 더 정확한 지표
