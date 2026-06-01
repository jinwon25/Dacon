# STEP 3 결과 종합 — v118 aug+hit+band (D-7 sprint)

날짜: 2026-05-25
실행 환경: Win11 CPU (per-fold ~3-5분, batch=256, ep≤150 patience=25)

## 진행 요약

| STEP | 결과 |
|---|---|
| 0 안전 백업 | ✅ `final_candidates/` 4개 락 (v106 / v112_v107_diverse / v117 K8sel τ0.45 / v117 selector τ0.60) |
| 1 base 입력표현 audit | ✅ v23 framework: canonical(yaw=last v →+x) + 변위(kalman residual) **이미 채택** → STEP 2 skip |
| 2 rotation TTA | ⏭️ skip (STEP 1 게이트 만족) |
| 3 새 base 단일 fold | ❌ **gate FAIL** — residual corr 구조적 floor |
| 4 5-fold + 블렌드 | ⏭️ STEP 3 FAIL로 skip |

## STEP 3 ablation 표 (fold0)

기존 baseline 참고: v77 BiGRU setup A 5-fold avg OOF = 0.6605, v90 mirror 5-fold avg OOF = 0.6636.

| 실험 | aug | band | setup | OOF (fold0 va) | corr_3d (v112) | fast+turn rh | wallclock |
|---|---|---|---|---|---|---|---|
| #1 v118 aug+band | ✅ | ✅(×2.5) | A | 0.6495 | 0.9885 | 0.250 | 5.2m |
| #2 control noaug-noband | ❌ | ❌ | A | 0.6630 | 0.9920 | 0.250 | 2.8m |
| #3 control noaug-noband | ❌ | ❌ | B | 0.6655 | 0.9919 | 0.350 | ~3m |
| **#4 aug-only** | ✅ | ❌ | B | **0.6715** | 0.9922 | 0.250 | ~5m |

게이트 조건:
- OOF ≥ 0.665 ✅ (실험 #3, #4)
- residual corr_3d_mag (v112_v107_diverse) < 0.93 ❌ (전부 0.988~0.992)
- AND 필요 → 4개 실험 모두 FAIL.

## 핵심 통찰

### 1) band weight ×2.5는 OOF를 깎는다
- 실험 #1(band on, ep 150) vs #2(band off, 같은 setup) = 0.6495 vs 0.6630 → **band -0.0135**.
- 이유 추정: 1-3cm CV/CA-error 샘플들에 over-focus, 쉬운 <1cm 샘플들의 학습 신호 약화.
- 결론: 이 framework에서 단순 sample-weight 증폭은 역효과.

### 2) random yaw + 50% flip 증강은 setup B에서 +0.0060 lift
- 실험 #3(noaug) vs #4(aug) on setup B: 0.6655 → 0.6715.
- canonical frame이 이미 yaw alignment 했지만, **추가 random yaw는 rotation equivariance 강화**해 OOF 약상승.
- 하지만 residual corr는 0.9919 → 0.9922 (변화 미미). 같은 sample들에서 같은 에러 패턴.

### 3) residual correlation ~0.99는 framework 구조적 floor
- Pool 멤버 전부 **kalman residual + canonical + GRU/Transformer/MDN** 공통 구조.
- base prediction은 동일한 kalman; NN은 잔차만 학습.
- → 어떤 NN 변종이 학습하든 (kalman residual 자체 에러)에서 자유롭지 못함.
- 게이트 0.93 도달하려면 **base를 바꿔야** (kalman-free / different baseline / different feature):
  - Neural CDE / ODE (사용자 금지 카드)
  - 다른 baseline (예: spline extrapolation + NN residual)
  - 다른 feature representation (예: position-velocity-acceleration → frequency domain)
  - Per-sample uncertainty 기반 base 선택 (selector — 시도된 v116/v117 → 변환률 변수 큼)

### 4) fast+turn subset은 어떤 변종도 못 잡는다
- v118 모든 실험: 0.25 (4/20 hit) 또는 0.35 (7/20)
- 메모리 기준 "fast+sharp-turn 388개 R-Hit 0.353" 와 일치 (subset 정의 다소 다름)
- → 약점 subset은 신경망 변종이 아니라 *다른 paradigm*이 필요.

## 변환률 관점

- 현 best LB 실측: **v106 = LB 0.6888 (OOF 0.6770, 변환률 +0.0118)**
- v110_v3 = LB 0.6884 (OOF 0.6775, 변환률 +0.0109) — over-fit 시그널
- v118 aug-only (실험 #4) 가정: 5-fold full → standalone OOF ~0.67 추정, 블렌드 lift 거의 0 (corr=0.99).
- 게이트 통과 못 했으므로 **블렌드에 넣어도 LB 변화 미미**.

## 최종 권고 (D-7)

1. **다음 슬롯 제출**: `final_candidates/submission_v112_v107_diverse_oof0.6768.csv`
   - 보수 blend (cap 0.30), paradigm 강제 다양화 (v107 deep Trans boundary 포함)
   - 변환률 +0.0118~+0.0125 안정 기대 → LB 예상 0.6886~0.6893 (0.69 가능)
2. **보수 백업**: `final_candidates/submission_v106_DE15w_oof0.6770.csv` (LB 0.6888 실측)
3. **STEP 3 cache 보존**: cache/v118_aug_hit_*.npz 4개 → 향후 base 변경 paradigm 시도 시 비교용
4. **시도하지 말 것**:
   - v118 → 5-fold 완주 (residual corr 0.99 → 블렌드 lift 0)
   - 추가 후처리 (이미 plateau)
   - 같은 framework의 NN 변종 재학습
5. **다음 세션 시도 가치 있는 방향** (D-day 이후 / 사후 분석용):
   - kalman residual을 **다른 baseline residual**로 교체 (예: spline extrapolation)
   - 입력 feature를 **frequency domain** 추가 (FFT magnitudes/phases for 마지막 N step)
   - Per-axis 분리 학습 (x,y,z 각각 다른 모델)

## 파일 산출

- `final_candidates/` — 4 락된 제출 + CHECKSUMS.txt
- `scripts/v118_aug_hit.py` — 새 base 학습 스크립트 (재사용 가능, Colab 호환)
- `cache/v118_aug_hit_fold0_setupA.npz` — 실험 #1 state
- `cache/v118_aug_hit_fold0_setupA_noaug_noband.npz` — 실험 #2
- `cache/v118_aug_hit_fold0_setupB_noaug_noband.npz` — 실험 #3
- `cache/v118_aug_hit_fold0_setupB_noband.npz` — 실험 #4 (최고 OOF 0.6715)
- `reports/base_repr.md`, `reports/v118_aug_hit_*.md` — STEP별 보고
