# 풍력발전 프로젝트 문서 안내

이 프로젝트는 다음 날의 기상 예보로 시간별 발전량을 예측합니다. [프로젝트 README](../README.md)는 최종 Public `0.6474704399`와 시도·한계를 요약합니다. 목표값이 들어간 문서 이름은 연구 목표이며 달성 성과가 아닙니다.

## 읽는 순서

1. [모델링 전략](modeling_strategy.md): 기본 입력·그룹별 모델·보정의 초기 설계.
2. [데이터 누출·가용 시점 감사](P0_DATA_AUDIT.md): 예보 발행 시각, 정답 범위, SCADA 단위와 시차.
3. [실험 결과와 채택 상태](SPRINT_066_STATUS.md) → [실험 보고서](EXPERIMENT_REPORT_066.md): 로컬 검증과 실제 공개 결과.
4. [상세 연구 이력](RESEARCH_HISTORY.md): 최종 선택 계보·지표 감사·기각 경로·과거 명령.
5. [개선 실험 계획](SPRINT_066_PLAN.md), [승격 검사 서비스](agent_service.md), [당시 병목 기록](score_bottleneck_20260718.md): 사전 기준과 의사결정 과정.

## 먼저 이해할 용어

| 용어 | 이 프로젝트에서의 뜻 |
|---|---|
| NWP / LDAPS / GFS | 수치 기상 예보 / 제공된 두 기상 예보 자료의 이름입니다. |
| day-ahead | 다음 날을 미리 예측하는 작업입니다. 예측 기준 시각 이후의 관측을 입력으로 쓰지 않습니다. |
| SCADA / proxy | 터빈 운전 관측 자료 / 이를 대신 추정하는 모델입니다. 평가 기간의 실제 SCADA를 쓴다는 뜻이 아닙니다. |
| NMAE | 설비용량으로 정규화한 평균 절대 오차입니다. 최종 점수에는 1−NMAE를 사용합니다. |
| FiCR | 실제 발전량으로 가중한 정산 지표입니다. 단순 적중률 평균과 다릅니다. |
| OOF / exact lineage | 학습 외 예측 / 실제 기준 모델의 구성과 계보를 맞춘 검증입니다. |
| Q1·Q2 / H1·H2 | 연도의 분기 / 반기를 나누는 표기입니다. 선택 구간과 후속 확인 구간을 구분합니다. |
| gate / promotion | 후보가 통과해야 할 검증 기준 / 후보를 선택된 기준으로 바꾸는 결정입니다. |
| incumbent | 연구 당시 선택된 기준 예측입니다. 과거 보고서의 기준이 최종 제출과 같지는 않습니다. |
| q05 / bootstrap | 재추출한 점수 분포의 하위 5% 지점 / 표본을 다시 뽑아 불확실성을 점검하는 방법입니다. |
| projected score | 로컬 개선이 이전된다고 가정한 예상 점수입니다. 실제 제출 점수와 다릅니다. |

## 날짜별 보고서

영문 보고서는 한국어 목적·핵심 판단을 먼저 읽고 당시 원문을 확인합니다. 외부 논문 제목·코드 식별자·수치·과거 판정은 근거로 보존했습니다. 높은 로컬 점수만으로 최종 선택을 판단하지 않습니다.

| 보고서 | 파일의 날짜 표기 |
|---|---|
| [점수 병목과 후보 판정](reports/agent_bottleneck_sprint_2026-07-17.md) | `2026-07-17` |
| [산출물 정리와 보존 기준](reports/artifact_cleanup_2026-07-17.md) | `2026-07-17` |
| [시간 블록 검증과 보정의 한계](reports/blocked_rolling_validation_2026-07-18.md) | `2026-07-18` |
| [오차 탐색과 성능 정체 원인](reports/bottleneck_eda_2026-07-14.md) | `2026-07-14` |
| [0.65 돌파 후속 검증 및 규정 감사 — 2026-07-26](reports/breakthrough_followup_2026-07-26.md) | `2026-07-26` |
| [분포 변화 보정의 검증 계약 감사](reports/covariate_shift_audit_2026-07-18.md) | `2026-07-18` |
| [그룹 간 결합과 특징 캐시](reports/cross_group_sprint_2026-07-13.md) | `2026-07-13` |
| [평가 구조·규정·신규 방법론 감사 — 2026-07-26](reports/evaluation_breakthrough_audit_2026-07-26.md) | `2026-07-26` |
| [평가·승격 로직 최종 감사와 0.65 판정 — 2026-07-29](reports/evaluation_promotion_final_065_2026-07-29.md) | `2026-07-29` |
| [그룹 3의 동일 계보 검증 예측 복원](reports/exact_group3_oof_2026-07-14.md) | `2026-07-14` |
| [동일 계보 검증과 정산 조건부 보정](reports/exact_oof_meta_gate_2026-07-17.md) | `2026-07-17` |
| [정산 보정 조건 탐색과 공개 전이](reports/exact_oof_meta_gate_sweep_2026-07-17.md) | `2026-07-17` |
| [외부 자료·사전학습과 가용 시점 검토](reports/external_data_pretrained_audit_2026-07-18.md) | `2026-07-18` |
| [그룹 3 KMA regime-conditioned power curve — 2026-07-26](reports/group3_regime_power_curve_2026-07-26.md) | `2026-07-26` |
| [기준 모델 잔차의 의사결정 보정](reports/incumbent_residual_decision_followup_2026-07-29.md) | `2026-07-29` |
| [독립 NWP 전문가 및 강건 라우팅 스프린트 (2026-07-18)](reports/independent_nwp_sprint_2026-07-18.md) | `2026-07-18` |
| [JMA·KMA 예보 결합 후보](reports/jma_msm_breakthrough_2026-07-26.md) | `2026-07-26` |
| [KMA 다년 G3 단독 risk probe 선정](reports/kma_g3_risk_probe_selection_2026-08-02.md) | `2026-08-02` |
| [KMA 다년 확장·G3 병목 재검증 보고서](reports/kma_multiyear_active_reconciliation_2026-08-02.md) | `2026-08-02` |
| [예측 전 관측을 이용한 전문가 선택](reports/kma_observation_router_2026-07-19.md) | `2026-07-19` |
| [KMA UM 10.4 컨텍스트 스프린트 (2026-07-19)](reports/kma_um_context_sprint_2026-07-19.md) | `2026-07-19` |
| [KMA 전문가의 연도 전이 검증](reports/kma_year_forward_blend_2026-07-26.md) | `2026-07-26` |
| [LDAPS 공간 방법론 전환 실험 — 2026-07-26](reports/ldaps_spatial_methods_2026-07-26.md) | `2026-07-26` |
| [메커니즘 다양성 블렌드 후속 검증](reports/mechanism_diversity_blend_followup_2026-08-02.md) | `2026-08-02` |
| [방법론과 규정 참고자료](reports/method_sources.md) | `od_sources` |
| [독립 멀티 NWP 및 JMA 2년 year-forward 후속 — 2026-07-28](reports/multisource_year_forward_followup_2026-07-28.md) | `2026-07-28` |
| [NOAA 다지점 issue-time 관측 후속 검증](reports/noaa_multistation_observation_followup_2026-08-02.md) | `2026-08-02` |
| [비교차 분포 코어 기준선 대칭성 감사 — 2026-07-29](reports/noncrossing_baseline_symmetry_audit_2026-07-29.md) | `2026-07-29` |
| [비교차 조건부 분포 코어 최종 검증 — 2026-07-29](reports/noncrossing_distributional_core_2026-07-29.md) | `2026-07-29` |
| [운전 국면별 결합의 공개 전이 실패](reports/phase_regime_cross_group_2026-07-17.md) | `2026-07-17` |
| [물리·예보 불확실성 신호 연구](reports/physical_signal_research_2026-07-18.md) | `2026-07-18` |
| [Pooled multi-source 돌파 실험 — 2026-07-26](reports/pooled_multisource_breakthrough_2026-07-26.md) | `2026-07-26` |
| [BARAM 2026 우선순위 실행 기록 — 2026-07-25](reports/priority_execution_2026-07-25.md) | `2026-07-25` |
| [정산 지표 보정과 강도별 공개 결과](reports/public_1508365_ficr_dose_response_2026-08-02.md) | `2026-08-02` |
| [공개 결과 방향을 이용한 정산 보정](reports/public_directional_ficr_2026-07-27.md) | `2026-07-27` |
| [공개 factor 병목 감사와 G1–G2 차등 reconciliation](reports/public_factor_and_group12_reconciliation_2026-07-28.md) | `2026-07-28` |
| [공개 요인 분해와 JMA GSM 후속 실험 — 2026-07-26](reports/public_factor_gsm_breakthrough_2026-07-26.md) | `2026-07-26` |
| [공개 양수 다중 요인 확장 후속 실험 — 2026-08-02](reports/public_positive_multifactor_followup_2026-08-02.md) | `2026-08-02` |
| [SCADA 대리 모델과 운전 국면](reports/regime_scada_sprint_2026-07-09.md) | `2026-07-09` |
| [잔차 시나리오와 JMA plateau 후속 연구 — 2026-07-28](reports/scenario_analog_and_plateau_breakthrough_2026-07-28.md) | `2026-07-28` |
| [SCADA 결합 비중과 점수 구성](reports/score_sprint_2026-07-09.md) | `2026-07-09` |
| [공개 사례 기반 병목 진단 및 발전 방향 (2026-07-18)](reports/source_driven_direction_2026-07-18.md) | `2026-07-18` |
| [시공간 다중 과제 후보의 당시 승격](reports/spatiotemporal_multitask_promotion_2026-07-18.md) | `2026-07-18` |
| [다기간 OOF 타당성·Temporal Smoothing·산출물 정리 — 2026-07-26](reports/temporal_smoothing_cleanup_2026-07-26.md) | `2026-07-26` |
| [평가 경계 보정 후보와 불확실성](reports/threshold_calibration_2026-07-17.md) | `2026-07-17` |
| [계층형 승격 감사와 후속 검증 — 2026-07-25](reports/tiered_promotion_followup_2026-07-25.md) | `2026-07-25` |
| [0.65+ / 상위 10% 후속 모델링 — 2026-07-27](reports/top10_trajectory_followup_2026-07-27.md) | `2026-07-27` |
| [터빈·그룹 3 결합 후보](reports/top_tier_sprint_2026-07-12.md) | `2026-07-12` |
| [학습·평가 파이프라인 감사](reports/training_evaluation_audit_2026-07-13.md) | `2026-07-13` |
| [궤적 평활화의 작은 공개 개선](reports/trajectory_smoothing_sprint_2026-07-14.md) | `2026-07-14` |
| [발전량 가중 정산 지표와 학습](reports/weighted_metric_sprint_2026-07-12.md) | `2026-07-12` |

## 공개본 실행 기준

원본 기상·발전량, 모델, 행별 예측과 캐시는 저장소에 없습니다. 연구 명령에 적힌 파일은 직접 생성·확보해야 하며, 문서에 경로가 있다는 이유로 공개된 자산으로 간주하지 않습니다. 규정·공지의 최신 확인은 별도 작업이며 여기서는 대회 당시 기록을 설명합니다.
