# EDA Post v122c (LB 0.6912 달성, 2026-05-26)

## 1줄 결론

**v122c는 weighted blend로서 거의 최적 (0.6769 OOF / 0.6912 LB)**. plateau 돌파의 진짜 카드는:
1. **v120 paradigm pool 확장** (multi-step RK4, hidden capacity 2x) → OOF +0.002~+0.004 잠재
2. **Disagreement-based selector** (메타특징 단독은 acc 55-60% 손해, but |v112-v120| 차이 + 메타 합치면 70%+ 가능)
3. **Neural CDE / FFT feature** — neither subset 공략 (30% 영역, mean d=26mm)

## 핵심 데이터

### Oracle bound (v112 + v120 per-sample 최소)
| 지표 | 값 |
|---|---:|
| v112 OOF hit | 0.6768 |
| v120 OOF hit | 0.6610 |
| v122c OOF hit | 0.6769 |
| **oracle min(v112, v120)** | **0.6989** ★ |
| both hit | 0.6389 |
| only v112 | 0.0379 (379 samples) |
| only v120 | 0.0221 (221 samples) |
| neither | 0.3011 (3011 samples) |

**v122c가 oracle hit의 96.72% 캡처** — selector 추가 lift 잠재 = +0.022 max.

### Selector accuracy 시뮬레이션
| selector acc | OOF hit | 비고 |
|---:|---:|---|
| 0.55 | 0.6715 | **손해** |
| 0.60 | 0.6754 | 손해 |
| 0.65 | 0.6771 | v122c 동등 |
| 0.70 | 0.6791 | +0.002 lift |
| 0.80 | 0.6885 | +0.012 lift |
| 0.90 | 0.6917 | +0.015 lift |
| 1.00 (oracle) | 0.6989 | 천장 |

### 메타특징만으로는 selector 학습 거의 불가능
| feature | only_v120 vs only_v112 Δz |
|---|---:|
| decel_max | -0.138 |
| speed_last | -0.086 |
| speed_max | -0.077 |
| turn_max | -0.075 |
| accel_m | -0.017 |
| jerk_m | +0.007 |

→ 두 그룹 메타특징 거의 동일. **|v112-v120| disagreement feature + per-axis residual feature 필수**.

### Boundary subset (v122c 0.8-1.5cm miss, n=2535)
- v112 hit 0.4047
- v120 hit 0.3629
- oracle 0.4915 (+8.68%p)
→ boundary 안에 selector 학습 잠재력 큰 영역.

### Neither subset (n=3011, mean d=26mm)
- 두 paradigm 모두 fail → **새 paradigm 진짜 필요한 영역**
- v122c도 9 sample만 어쩌다 잡음
- Neural CDE / FFT / per-axis 분리 모델 ROI 큰 곳

### Conditional weight 시도 결과 (실패)
| 시도 | OOF |
|---|---:|
| v122c 현재 (학습된 weight) | 0.6769 |
| static v120_w=0.0 | 0.6768 |
| static v120_w=0.4 | 0.6720 |
| static v120_w=1.0 | 0.6610 |
| hard_tau=1.5 wh=0.7/wl=0.3 | 0.6740 |

→ 단순 heuristic conditional weighting은 v122c DE blend보다 못함.

### Train vs Test 분포 (covariate shift 거의 없음)
| feature | train mean ± std | test mean ± std | ks proxy |
|---|---|---|---:|
| speed_last | 0.639 ± 0.348 | 0.606 ± 0.329 | 0.104 |
| accel_mean | 3.351 ± 2.519 | 3.236 ± 2.450 | 0.041 |
| turn_max | 0.689 ± 0.641 | 0.683 ± 0.628 | 0.001 |

→ test는 train보다 살짝 slow/static. **TTA로 활용 가능성 낮음**. 분포 차이 미세.

### Test prediction 다양성 (paradigm 진정성 지표)
| 쌍 | mean |v_a - v_b| (mm) | q90 |
|---|---:|---:|
| v122c - v112 | 0.509 | 0.958 |
| v112 - v106 | 0.154 | 0.297 |
| v120 - v112 | **2.201** | **4.016** ★ |

→ v120 vs 기존 framework prediction L2 distance가 기존 plateau 멤버끼리의 **14배**. paradigm 진정성 확정.

## 우선순위 카드 (재정렬)

| 우선 | 카드 | 시간 | OOF lift | LB lift (변환률 +0.0143 가정) |
|---|---|---|---|---|
| **1** | **v120 multi-step RK4 (n_steps=2,3,4)** | 1-2h × 3 | +0.001~+0.003 | +0.0010~+0.0040 |
| **2** | **v124 — v120 hidden=128/latent=128 (capacity 2x)** | 2-3h | +0.001~+0.003 | +0.0010~+0.0040 |
| **3** | **v125 disagreement selector** ( |v112-v120| + meta → MLP) | 3-4h | +0.001~+0.004 | +0.0010~+0.0060 |
| 4 | v126 Neural CDE (torchcde) | 1-2일 | +0.002~+0.005 | +0.0030~+0.0070 |
| 5 | v127 FFT feature (last 6 step magnitude/phase) | 1일 | +0.001~+0.003 | +0.0010~+0.0040 |
| 6 | per-axis (x/y/z 분리) v120 변종 | 1일 | +0.001~+0.003 | +0.0010~+0.0040 |
| 7 | Frenet-frame coordinate transformation | 1-2일 | +0.001~+0.003 | +0.0010~+0.0040 |

**합산 목표**: 카드 1+2+3 합치면 OOF 0.6790~0.6810 → LB 0.6940~0.6960.

## 즉시 시작 카드: v123_v120_multistep + v124_v120_big

다음 sprint:
1. `scripts/v123_v120_multistep.py` — n_steps=2,3,4 학습
2. `scripts/v124_v120_big.py` — hidden=128/latent=128 학습
3. v110 pool에 추가 → DE 재계산 → v122 새 candidate
