# D-7 Sprint 2차 종합 — STEP A (ceiling) + STEP B (offset)

날짜: 2026-05-25 (v118 STEP3 게이트 FAIL 직후 ceiling + cheap offset 진단)

## 한 줄 결론

**STEP A 천장 시그널 확인 + STEP B 게이트 FAIL → v112_v107_diverse 그대로 D-day 제출 권고**.
포스트-처리/재학습으로 LB 0.69 도달 신호 없음. 0.6888 plateau 재확인.

---

## STEP A — Ceiling Diagnosis 핵심 (reports/ceiling_diagnosis.md)

### A.1 매끈 subset best-possible d

| subset | predictor | median (mm) | hit1cm |
|---|---|---:|---:|
| smooth20 (n=2000) | **oracle (per-sample min of 6 geom)** | **2.51** | **0.950** |
| smooth20 | v112 OOF | 3.20 | 0.923 |
| smooth20 | CV (단일 best geom) | 3.64 | 0.900 |
| smooth20 | Kalman | 3.58 | 0.906 |
| all (10000) | oracle | 5.74 | 0.731 |
| all | v112 OOF | 6.80 | 0.677 |
| rough20 | oracle | 10.86 | 0.442 |
| rough20 | v112 OOF | 11.44 | 0.444 |
| fast+turn (n=96) | oracle | 15.94 | **0.177** |
| fast+turn | v112 OOF | 15.63 | **0.271** |

**해석**:
- 매끈 subset에서 단순 외삽 oracle도 **0.95 hit가 천장**. v112가 oracle 대비 2.7%p (-3 hits/100) 모자람.
- rough subset에서 v112 ≈ oracle. NN이 단순 외삽 이상 거의 못 함.
- **fast+turn subset에서만 v112가 oracle 대비 +9.4%p lift** — NN이 진짜 학습한 부분.

### A.2 관측 노이즈

| 방법 | σ_x | σ_y | σ_z | (mm) |
|---|---:|---:|---:|---|
| 2차차분 var (smooth20) | 0.464 | 0.482 | 0.396 | 가장 깨끗한 estimator |
| Kalman 코드 가정 | 0.300 | 0.300 | 0.300 | 50% 작게 추정 |

**관측노이즈만으로 d 중앙값 floor ≈ 2.7 mm** (smooth20 oracle 2.51mm는 이 floor에 거의 도달).
Kalman 가정이 1.5x under-trust observations지만 v112 안에 여러 Kalman σ 섞여있어 critical 아님.

### A.3 양자화

x/y/z 모두 **1e-6 mm storage precision**에서 완벽 snap, 그보다 큰 grid 없음.
→ **STEP C (snapping) 무효 — 폐기 확정**.

### STEP A 결론
- 매끈 subset이 이미 95% hit → 5% miss는 noise floor.
- rough subset은 v112 ≈ oracle → 단순 외삽 한계 = NN 한계.
- 진짜 learnable signal은 fast+turn 좁은 부분집합 (n~96~388)에 국한.
- → **데이터 ceiling 매우 가까움**.

---

## STEP B — Hit-Aware Offset (reports/hit_offset.md)

base: v112 OOF hit = **0.6768**.  Gate: +0.0008.

### B.1 Global 3D δ (±3mm coarse → ±0.5mm fine)
- best δ = (0.000, 0.000, 0.000) mm
- hit = 0.6768 (**Δ = 0**) → **FAIL**
- residual mean = (-0.02, +0.17, -0.10) mm — v112 이미 unbiased.

### B.2 Body-frame δ (along / cross_h / vert)
- best δ = (0.000, -0.100, -0.100) mm
- hit = 0.6770 (**Δ = +0.0002**) → **FAIL**

### B.3 Speed-conditional 5-bin δ
- In-sample aggregate: 0.6794 (**Δ = +0.0026** 보이지만…)
- **5-fold leave-out CV-honest: 0.6742 (Δ = -0.0026)** → 완벽 overfit.

→ STEP B 전 변종 명확히 FAIL.

---

## STEP D — aWTA (수행 안 함, 정성 평가)

| 평가 | 결과 |
|---|---|
| 시간 비용 | 5-fold full 학습 ~6-10h |
| 게이트 통과 가능성 | 매우 낮음 |
| 이유 1 | v118 같은 framework 변종 모두 corr 0.99 floor (이미 검증) |
| 이유 2 | hard subset n=96 (좁음). +0.02 R-Hit 위해서는 ~2 hits flip — 노이즈 수준 |
| 이유 3 | fast+turn v112=0.271 vs oracle=0.177 → NN이 이미 +9%p, 추가 +2%p 헤드룸 좁음 |

→ **STEP D skip** 결정. 메모리의 사후 분석용 방향(Neural CDE, frequency domain, per-axis 분리)에 부합.

---

## D-day 제출 권고 (확정)

**Primary**: `final_candidates/submission_v112_v107_diverse_oof0.6768.csv`  → LB 0.6888 실측, 변환률 +0.0120 (최강).

**Backup**: `final_candidates/submission_v106_DE15w_oof0.6770.csv`  → LB 0.6888 실측.

## 산출

- `scripts/ceiling_diagnosis.py` — STEP A 진단 (재실행 가능)
- `scripts/hit_offset.py` — STEP B global/body/speed offset 탐색
- `scripts/hit_offset_cv.py` — speed-conditional 5-fold leave-out 검증
- `reports/ceiling_diagnosis.md` + `.json`
- `reports/hit_offset.md` + `.json`
- `reports/sprint_d7_step_AB_summary.md` (본 문서)

## 사후 분석용 (D-day 후 시도 가치)

천장에 가깝다는 STEP A 진단을 고려해도, 좁은 hard subset이 진짜 learnable signal — 단,
같은 kalman+canonical+NN framework로는 corr 0.99 floor 못 깸. 새 paradigm만이 의미 있음:

1. **kalman 대체 baseline** (spline/poly extrapolation residual learning)
2. **frequency domain feature** (FFT magnitude/phase last N step)
3. **per-axis 분리 모델** (x/y/z 다른 NN)
4. **Neural CDE / ODE backbone**
5. **hard-subset only K-mode WTA** (full 안 하고 96-388 sample 직접 학습 → cheap이지만 LB 변환 불확실)
