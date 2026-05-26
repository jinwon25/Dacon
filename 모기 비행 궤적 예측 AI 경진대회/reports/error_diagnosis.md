# TASK 1: 에러 진단 (2026-05-25)

## 데이터 구조
- 모든 train/test CSV: 정확히 11 timestep × (timestep_ms, x, y, z), dt=40ms, 총 관측 시간 400ms
- 타깃: 마지막 관측 +80ms 좌표
- 분석 대상: `submission_v112_v107_diverse_oof0.6768.csv` (LB 0.6888 확정 best)

## 1.1 거리 분포 (OOF R-Hit@1cm = 0.6768)

| 거리 d | 개수 | 비율 |
|---|---:|---:|
| d < 1cm   (hit) | 6,768 | 67.68% |
| d 1-3cm (가까운 miss, boundary 영역) | 2,445 | 24.45% |
| d 3-5cm | 410 | 4.10% |
| d ≥ 5cm  (큰 miss) | 377 | 3.77% |

분위수 (cm): Q50 0.680 / Q75 1.201 / Q90 2.412 / Q95 4.209 / Q99 8.743

**해석**: 24.45%가 1-3cm 영역 (boundary refinement가 영향 줄 수 있는 영역). 3cm 초과는 7.87% — 큰 miss는 NN 패러다임 자체 한계.

## 1.2-1.3 메타특징별 R-Hit (분위수 5-bin)

| feature | Q1 | Q2 | Q3 | Q4 | Q5 | spread |
|---|---:|---:|---:|---:|---:|---:|
| **accel_mean** | 0.923 | 0.796 | 0.664 | 0.556 | **0.444** | **+0.479** ★ |
| jerk_mean | 0.904 | 0.789 | 0.678 | 0.554 | 0.458 | +0.446 |
| accel_max | 0.912 | 0.754 | 0.667 | 0.582 | 0.469 | +0.443 |
| speed_max | 0.896 | 0.765 | 0.684 | 0.560 | 0.478 | +0.418 |
| speed_last3 | 0.875 | 0.771 | 0.671 | 0.580 | 0.488 | +0.387 |
| speed_mean | 0.863 | 0.756 | 0.692 | 0.579 | 0.494 | +0.369 |
| turn_mean | 0.806 | 0.748 | 0.692 | 0.634 | 0.504 | +0.302 |
| turn_max | 0.801 | 0.719 | 0.686 | 0.651 | 0.526 | +0.275 |
| dir_std | 0.797 | 0.721 | 0.683 | 0.664 | 0.518 | +0.279 |
| kappa_mean | 0.641 | 0.690 | 0.687 | 0.673 | 0.693 | +0.051 |
| kappa_max | 0.672 | 0.690 | 0.676 | 0.680 | 0.665 | +0.024 |

### Subset miss 농축

| subset | n | R-Hit |
|---|---:|---:|
| speed_max top 10% | 1,000 | 0.502 (-0.175) |
| turn_max top 10% | 1,000 | 0.452 (-0.225) |
| kappa_max top 10% | 1,000 | 0.653 (-0.024) |
| **turn_max top 20% & speed_max top 20%** | **388** | **0.353** (-0.324) ★★ |

## 결론 / 우선순위 재조정

1. **NN 약점은 "고가속·고속도·고turn-rate" subset에 집중** — accel_mean spread +0.479
2. **kappa(기하학적 곡률)는 R-Hit 차이 거의 없음** — 모기 trajectory의 곡률 자체는 NN이 잘 잡는데, 동역학(속도/가속도 크기)이 핵심 miss 원인
3. **turn_max top 20% & speed_max top 20% (388 sample, 전체 3.9%)에서 R-Hit 0.353** — 이게 ensemble pool 천장의 주범
4. 이 subset은 정확히 **constant-acceleration / coordinated-turn 물리 외삽기가 잡을 영역**:
   - 고속 직진: CA 외삽 정확
   - 갑작스러운 turn: IMM coordinated-turn mode
5. 1-3cm subset (24.45%)은 boundary 가족이 이미 차지. 추가 lift 어려움.
6. ≥3cm subset (7.87%)이 **물리 외삽기의 ROI**.

## TASK 2/3/4 우선순위 (수정)

| 우선 | 카드 | 잠재 R-Hit lift |
|---|---|---:|
| **1** | **TASK 2: SavGol + CA 외삽 + CA Kalman + IMM** | 고가속/고속도 subset 잡으면 +0.005~+0.020 |
| 2 | TASK 4: LightGBM 메타러너 (메타특징 라우팅) | +0.002~+0.005 |
| 3 | TASK 4: geometric median | +0.001~+0.003 |
| 4 | TASK 3: MDN mode 선택 (고turn subset만) | +0.001~+0.003 |

## 산출물
- `cache/meta_features.npz`: 11개 메타특징 + d + ids
- `reports/error_diagnosis.md` (본 문서)
