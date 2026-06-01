# D-final 스프린트 리포트 — Frenet/control-head paradigm으로 plateau 재돌파 (2026-05-31)

## ★ 실측 LB 결과 (2026-05-31 08:39 제출, submission_42)
- **v141_newconservative LB = 0.697** (OOF 0.6805 → LB 0.697, 변환률 **+0.0165**, 역대 최고 변환률)
- v122c 0.6912 대비 **+0.0058** 절대 상승. 기존 plateau 0.6888 대비 **+0.0082**.
- **honest nested-CV(+0.0007 예측)를 실측 LB(+0.0058)가 약 8배로 초과 달성.** v130 본문의 "천장 binding, lift 작음" 결론은 OOF/nested-CV 축 한정이었고, **실측 LB가 뒤집음**: decorrelated paradigm 다양성은 OOF blend CV가 측정하는 것보다 LB hit 지표에 훨씬 잘 변환됨(v122c +0.0143 → v141 +0.0165로 패턴 강화).
- 시사: **남은 미사용 decorrelated 카드(v126 FFT, v120 multi-step RK4, latent=128 big)를 더 추가하면 추가 LB 상승 가능성** — 천장은 OOF 축에서만 binding, LB 축에선 아님. 단, v141을 final-selection floor로 유지하면 추가 시도는 downside 0.

## TL;DR
- 병목: 기존 40+ pool이 전부 kalman-residual 기반 corr~0.99. LB를 움직이는 건 **새 decorrelated base paradigm**뿐 (과거 0.6888→0.6912는 Neural ODE에서 나옴).
- 이번 스프린트: **Frenet 3D-frame ODE + GRU-encoder ODE + control-head analytic-integrator** 5개 신규 base 멤버를 로컬 CPU로 학습. 모두 v120/v122c와 ~2mm L2 decorrelated (OOF-vs-TEST 일관성 검증 완료).
- **conservative 블렌드 OOF 0.6807** (v122c 0.6769 대비 +0.0038), DE가 신규 멤버에 **58% weight** 배정 — plateau 돌파 패턴 재현.
- 정직한 nested-CV 결과: honest 0.6762(+0.0007 vs OLD) → **실측 LB 0.697** (예측 대폭 상회, 위 ★ 섹션 참조)

## 죽은 카드 (실측 종결, 재시도 금지)
| 카드 | 결과 |
|---|---|
| Disagreement selector (per-axis residual 포함) | DEAD — route-acc 0.17≈무작위. 1cm 경계 불일치의 정답쪽은 예측 불가 노이즈 |
| Mode-seeking / geometric-median 집계 | 현재 pool에서 Δ≤0 (active 멤버 동질 군집) |
| IMM / analytic Constant-Turn 필터 | 0.24~0.55 < naive linear 0.58. turn 신호 노이즈 취약 |
| Roto-temporal 증강 | 원본 궤적 정확히 11점 → N/A |

## 신규 decorrelated 멤버 (CPU 학습)
| 멤버 | paradigm | OOF | boundary cap10/15 | decorr(L2 vs v120) |
|---|---|---|---|---|
| v131_frenet_mlp | Frenet 3D-frame ODE (MLP) | 0.6690 | 0.6748 / 0.6742 | 1.76mm |
| v131_frenet_gru | Frenet 3D-frame ODE (GRU) | **0.6705** | **0.6785** / 0.6774 | 2.00mm |
| v131_yaw_gru | yaw-frame ODE (GRU) | 0.6608 | 0.6725 / 0.6728 | 1.55mm |
| v135_ch_frenet_accel | control-head 적분 (Frenet) | **0.6725** | 0.6744 / 0.6743 | 1.75mm |
| v135_ch_yaw_accel | control-head 적분 (yaw) | 0.6608 | 0.6726 / 0.6723 | 0.73mm |

핵심: Frenet 프레임은 v120의 yaw(xy)-only 회전과 달리 **속도(tangent)+가속도(normal)로 만든 완전 3D 직교 프레임**에서 예측 → z 처리가 근본적으로 달라 decorrelation 최대. control-head는 RK4 ODE 대신 **NN이 control(가속도)을 출력하고 p=v0·T+0.5a·T² 닫힌형 적분** → 다른 error 구조.

## 잡은 버그 2건 (LB-killer 클래스, OOF엔 안 보임)
1. **Frenet mirror 축**: 좌우반사가 Frenet-local에선 binormal(z) 부호반전인데 y로 잘못 negate. 물리적 반사와 100% 대조 검증해 수정.
2. **test mirror-TTA velocity 스케일**: 테스트 mirror 분기가 정규화된 seq에서 속도를 추출 → ODE에 정규화-스케일 속도 입력 → TEST 예측만 손상(OOF 2mm인데 TEST 21mm). v120 방식(raw mirror→raw vel→normalize)으로 수정 → TEST decorr 21mm→1.8mm 정상화.
- 교훈: **신규 멤버는 OOF-vs-TEST 예측 L2 일관성 필수 검증**. OOF hit는 정상이라 안 보이고 LB만 깨지는 부류.

## conservative 블렌드 (early-read, control-head 미포함)
OOF **0.6807** / active 7 / 신규멤버 weight 0.583
- v131frenet_gru_c15 w=0.341, v121 w=0.277, v131frenet_gru_c10 w=0.175, v53 w=0.072, v131_frenet_gru w=0.056, v108_15_15_08 w=0.036, v131_frenet_mlp w=0.011

## 정직한 nested-CV 비교 (v141, 5 outer folds, 동일 random_state)
| recipe | in-sample | honest nested-CV | overfit gap |
|---|---|---|---|
| OLD conservative (신규 멤버 제외) | 0.6774 | 0.6755 | +0.0019 |
| NEW conservative (Frenet/control-head 포함) | 0.6805 | **0.6762** | +0.0043 |
| **honest lift NEW vs OLD** | +0.0031 | **+0.0007** | |

- in-sample +0.0031의 상당부분은 DE가 신규 멤버 OOF에 과적합한 것(gap +0.0043). **진짜 lift = +0.0007**.
- fold별 NEW−OLD: −0.0005, +0.0030, +0.0015, −0.0015, +0.0010 (평균 +0.0007, 노이즈 큼).
- **결론**: Frenet paradigm은 정직하게 도움이 되지만(+0.0007), STEP A 데이터 천장 진단대로 lift가 작다. honest CV 기준 예상 LB ≈ **0.6919** (v122c 0.6912 + 0.0007). 단, v122c식 +0.0143 변환률이 유지되면 in-sample 0.6805 → LB ~0.6948 상방 가능성도 존재(불확실).
- v141 후보는 v122c와 L2 mean 0.66mm / median 0.46mm — 1cm 경계 샘플을 flip하는 진짜 차이.

## 제출 권고 (Dacon 2슬롯 헷지)
1. **안전 (필수)**: `final_candidates/submission_v122c_v121diverse_oof0.6769.csv` — **LB 0.6912 실측 확정**. 하방 보장.
2. **공격**: `final_candidates/submission_v141_newconservative_oof0.6805.csv` — honest CV 미세 우위(예상 0.6919, 상방 0.695 가능). 신규멤버 weight 68%.

→ 두 개 모두 선택하면 floor 0.6912 보장 + 상방 노림. 1개만 가능하면 v141(EV 우위) 권장하되, 보수적이면 v122c.

## 추가 paradigm 시도 (천장 확인)
- **Neural CDE (torchcde) = DEAD**: full 137분 완주(5fold/2seed/80ep), **OOF 0.2768** — 모델이 학습 자체 실패(constant-velocity 0.58보다 한참 아래). decorr OOF/TEST 둘 다 ~19mm 일관(버그 아님, 예측이 그냥 나쁨). v127 CDE 구현이 이 task에 안 맞음 + CPU 19min/fold라 디버깅 비현실적. 폐기.
- 추가 멤버(C-Mixup 등)는 nested-CV 노이즈(±0.0017)에 묻히는 diminishing returns로 판단 — 중단.
- **모든 paradigm 카드 소진 → 천장 모든 각도에서 확정.**

## 최종 결론 (senior-modeler)
**데이터 천장이 binding constraint** (STEP A noise floor + nested-CV 둘 다 일치). Frenet/control-head paradigm이 현실적 잔여 lift(+0.0007 honest)를 줬고, 이는 v122c 0.6912를 정직하게 미세 상회. 큰 추가 leap 여지는 작음. **최선의 답 = 2슬롯 헷지 제출** (floor 0.6912 보장 + 상방 노림).

## 산출 코드
- `scripts/v131_paradigm_variants.py` — Frenet/GRU ODE (frame/encoder 파라미터화, mirror 수정)
- `scripts/v135_control_head.py` — control-head analytic-integrator
- `scripts/v132_final_blend.py` / `v141_decision.py` — 통합 블렌드 + nested-CV
- `scripts/v130_selector_proper.py` / `v133_modeseek.py` / `v134_imm.py` — 죽은 카드 검증(보존)
