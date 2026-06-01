# 작업 로그 (시간순)

모기 비행 궤적 예측 — 베이스라인부터 Private 2위(0.703151)까지의 진행 기록. 상세 분석은 `docs/reports/`, 솔루션 요약은 `docs/SOLUTION.md` 참고.

## LB 진행 요약

| 단계 | 후보 | OOF | LB | 비고 |
|---|---|---:|---:|---|
| 1 | candidate-selection 베이스라인 | - | 0.6306 | 첫 제출 |
| 2 | v77 BiGRU (Kalman 잔차 NN) | 0.6633 | ~0.6750 | NN paradigm 안착 |
| 3 | v98 5-way blend | 0.6760 | 0.6882 | 변환률 +0.0122 |
| 4 | v106 DE15w / v112 diverse | 0.6770 | 0.6888 | **plateau** (멤버 corr ~0.99) |
| 5 | **v122c Neural ODE blend** | 0.6769 | **0.6912** | 1차 돌파 (paradigm diversity) |
| 6 | **v141 Frenet/control-head** | 0.6805 | **0.697** | 2차 돌파 (변환률 +0.0165) |
| 7 | v148blend (DE 최적, CREE w=0.082) | 0.6831 | 0.6996 | OOF 최고지만 LB는 아래 |
| 8 | **v148creefwd α=0.25 (CREE 수동주입)** | 0.6808 | **0.7016** | 3차 돌파 (over-conversion) |
| 9 | v157_ens3a0.3 (3-CREE 앙상블 α0.30) | — | 0.7020 | α 단조 상승 확인 |
| 10 | **v157_ens3a0.40 / a0.45** | — | **Public 0.7022 / Private 0.703151** | **최종 2위** |

## 단계별 메모

### 1~4단계 — Kalman 잔차 NN 풀 (plateau 0.6888)
- canonical local frame(마지막 속도 yaw 정렬) + Kalman 잔차 타깃 학습.
- 백본: BiGRU(v77/v90), TCN(v42), Transformer(v107), MDN-WTA(v109).
- boundary refinement(cap 적용 잔차 보정), yaw aug + y-mirror.
- **병목 진단**: 모든 멤버가 같은 잔차 base 공유 → corr ~0.99 floor. 어떤 selector/aug/weight로도 못 깸. **base 자체를 바꿔야 함.**

### 5단계 — Neural ODE (1차 돌파, 0.6912)
- 타깃 `y − last_obs`(Kalman 미사용), 6D 상태(위치·속도) + neural acceleration field, RK4 80ms 적분.
- 단독 OOF는 낮으나 base 풀과 L2 ~2.2mm 직교 → DE가 ~40% weight 부여.

### 6단계 — Frenet/control-head (2차 돌파, 0.697)
- Frenet 3D 프레임(tangent·normal·binormal): z 처리가 근본적으로 달라 decorrelation 최대.
- control-head: NN이 가속도 출력 + `p = v₀·T + ½a·T²` 닫힌형 적분.
- **버그 2건 수정**(OOF엔 안 보이고 LB만 깨지는 클래스): Frenet mirror 축(binormal z 부호반전), test mirror-TTA velocity 스케일. → 신규 멤버는 OOF-vs-TEST L2 일관성 필수 검증.

### 7~10단계 — CREE 회전물리 + 수동 주입 (3차 돌파, 0.703151)
- 공개 Dacon baseline(HyperPhysics 회전물리)을 5-fold OOF로 포팅 → base와 2.82mm 직교.
- DE는 OOF-greedy라 CREE를 0.082로 과소평가 → **수동 α 주입(over-conversion)** 이 LB를 이김.
- 3-CREE 앙상블(dirnet seed42 + dirnet seed1 + 3step-heading)로 주입원 강화.
- α를 0.25→0.30→0.40으로 키우며 Public 0.7016→0.7020→0.7022 단조 상승 → 최종 2슬롯 α=0.40/0.45 선택 → **Private 0.703151 (2위)**.

## 검증된 Dead-end (재시도 금지)
disagreement selector, mode-seeking/geometric-median, IMM/analytic-turn, Neural CDE, Flow/SONODE 4-mechanism 주입, 같은 프레임 encoder 변종, pseudo-label. 상세는 `docs/SOLUTION.md` §7.
