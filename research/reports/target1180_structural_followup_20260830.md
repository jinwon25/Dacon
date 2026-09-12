# 1180 목표 구조적 후속 연구 — 2026-08-30

## 현재 결론

- 공식 기준선은 v244 Public `1175.9746833121`이다.
- 같은 fallback 계열의 Public 결과에 맞춘 route/weight 재튜닝은 하지 않았다.
- 새 정보원 세 개, 조기결합, 배포형 pseudo-season 학습, 독립 LightGBM, 새 route를 검증했다.
- 처음 만든 v261/v264는 실행 감사에는 통과했지만 OOF 기본 피처와 실제 ZIP의 미래시즌
  변환이 달랐다. runtime-faithful 재감사에서 첫 source가 `-0.4598`로 반전되어 **제출
  부적격**으로 강등했다.
- v277은 runtime-faithful OOF에서 v244 대비 full-2022 `+9.5828`, late-2023
  `+2.1510`, full-2024 `+0.7917`이었지만, 공식 Public은 `1175.3125589868`로
  v244보다 `-0.6621243253` 낮았다. 따라서 **기각하고 승격하지 않았다**.
- 공식 champion은 계속 v244이며 목표 `1180`까지의 잔여 격차는 `4.0253166879`다.
  v277 결과에 맞춘 pseudo-deployment 계열의 route/weight 재튜닝은 하지 않는다.

## 중단 복구

전원 종료 시점은 v255 특성 캐시 생성 직후였다. 남아 있던 두 Parquet 캐시를 재사용해
2022/2023/2024 fold를 이어서 학습했으며 세 fold 체크포인트와 요약을 모두 복구했다.

## 독립 표현 실험

아래 단독 BSS는 동일한 v217 114피처 XGB 대비 비교용이다. v217 단독 BSS는 대략
`695.72 / 725.08 / 836.25`다.

| 버전 | 구조 | 2022 | 2023 | 2024 | 판정 |
|---|---|---:|---:|---:|---|
| v255 | 구종 구성 가중 TrackMan 릴리스·무브먼트 평균/분산 103개 | 691.18 | 726.76 | 833.90 | 전역 대체 실패, 제한 route 보완 신호 |
| v257 | 현재 카운트 조건부 prior-season TrackMan 프로필 63개 | 699.39 | 720.04 | 844.48 | 시즌 부호 반전, 단독 기각 |
| v258 | target-free 시퀀스 매핑 기반 타자 노출 프로필 89개 | 698.34 | 710.80 | 842.02 | 시즌 부호 반전, 단독 기각 |
| v262 | command 98개 + batter 89개 조기결합, 총 301피처 | 686.56 | 719.60 | 미개봉 | 두 source 실패로 조기 종료 |

v258은 과거 sequence alignment의 dominance `>=.99`, support `>=20`인 타자 677명만
사용했다. profile은 target-free TrackMan에서 만들었고 현재 행의 batter ID만 조회한다.
v257/v258이 전역 2023에서 실패했어도 v244 제한 route에서는 일부 양수였으므로, 전역
BSS만으로 결론내리지 않고 고정 route 감사까지 수행했다.

## 고정 route 및 전문가 결합

| 버전 | v244 대비 full-2022 | late-2023 | full-2024 | 핵심 판정 |
|---|---:|---:|---:|---|
| v256 | +5.4357 | +1.1230 | +0.8402 | v255 25% 고정 dose, p05 음수 |
| v259 | +5.7071 | +1.3227 | +0.7028 | 네 family/쌍 10개 중 source가 batter 선택, robust 실패 |
| v260 | +11.4543 | +2.4473 | +1.4514 | v254 command specialist + batter complement, 2024 월 4/8 |
| v261 | +11.0348 | +2.6197 | +1.6170 | 기존 v70의 4--9월 안정 구간 적용, 현 최선 |

v260/v261 공식은 다음처럼 역할을 분리한다.

1. v244의 네 route와 각 route weight는 그대로 둔다.
2. command 모델이 base XGB보다 0.5 쪽으로 이동하는 route 행은 command를 60% 사용한다.
3. 나머지 route 행은 batter 모델을 25% 사용한다.
4. v261은 4--9월에만 위 전문가를 켜고 3월·10월은 v244를 정확히 보존한다.

v261 full-2024 활성 구간 증분은 `+4.7583`이며 bootstrap 평균은 약 `+4.9`다.
하방은 pitcher `-0.6554`, crossed `-4.3769`, chronological `-1.3227`, Reality Check
`p=.09445`다. 평균 개선은 명확하지만 독립 confirmatory 2025 라벨이 없어 하방을 0 위로
확정하지 못했다.

## 배포 계약 재감사와 v264 기각

v252/v258의 TrackMan supplement는 과거시즌만 사용했지만, 두 전문가의 OOF 114개 기본
피처는 v217 historical training transform이었다. 실제 v264 ZIP은 v242에서 확인한 frozen
future-season runtime transform을 사용한다. 이 차이는 기본 XGB BSS를 연도별로 수백 점
바꿀 정도로 컸으므로 기계적 ZIP 감사만으로 제출할 수 없었다.

v265는 학습 행·가중치·트리 파라미터·TrackMan supplement를 모두 고정하고 audit 행의
기본 피처만 실제 runtime transform으로 교체했다. 2022 command/batter 단독 BSS는 각각
`455.1920/463.7719`였고, 고정 v261 공식의 v244 대비 증분은 `-0.4598`이었다. 사전
early-reject에 따라 진행 중인 2023 학습을 중단하고 2024는 열지 않았다. v264 폴더의
`MODEL_EVIDENCE_REJECTED.md`에 제출 금지 근거를 남겼다.

## pseudo-deployment 학습

v267은 각 과거 정규시즌을 그해의 미래 test처럼 변환했다. 예를 들어 2023 블록은
season `<2023`에서 고정한 lookup과 zero-anchor runtime branch로 만들었다. 다음 해 모델은
이런 pseudo-runtime 블록만 누적 학습하므로 train/test 변환 의미가 같다.

| audit | 기존 runtime XGB | pseudo-deployment XGB | 차이 |
|---:|---:|---:|---:|
| 2022 | 482.2314 | 483.8234 | +1.5920 |
| 2023 | 426.6679 | 502.9788 | +76.3109 |
| 2024 | 600.9391 | 779.8467 | +178.9076 |

단독 모델은 세 해 모두 개선됐다. v244 내부 결합은 original XGB와 pseudo XGB를 정확히
반반 평균하고, 기존 네 route와 4--9월 calendar를 그대로 보존한 v271 한 개로 고정했다.

| 축 | v244 대비 gain | 양수 월 | 최악 월 |
|---|---:|---:|---:|
| full-2022 | +9.5828 | 5/6 | -13.7841 |
| late-2023 | +2.1510 | 2/2 | +0.0357 |
| full-2024 | +0.7917 | 3/6 | -4.1285 |

full-2024 활성 구간 bootstrap 평균은 약 `+2.4~+2.6`이지만 p05는 pitcher `-2.9915`,
crossed `-6.6313`, chronological `-3.7826`, Reality Check `p=.2709`다. source에서 계산한
닫힌형식 최적 pseudo weight는 `1.8790/1.0970`, full-2024 최적은 `0.5081`이었다. 현재
고정 `0.50`이 잠금축 최적점과 사실상 같아 추가 weight 탐색은 중단했다.

v269/v270의 독립 LightGBM 다양성은 late-2023 월 안정성이 악화되어 기각했다. v272의
untouched R_CORE/anchor 새 route는 source-selected core complement가 full-2024 `-4.7089`로
반전되어 기각했다. v274 command/batter pseudo 전문가는 단독 BSS를 추가 개선했지만 v275
source maximin에서는 pseudo-base가 그대로 선택되어 최종 패키지에 넣지 않았다.

## 규칙과 검증 계약

- test.csv는 모델·route·calendar 선택에 읽지 않았다.
- 다른 test 행의 빈도, 평균, 순서, 선수 통계는 사용하지 않는다.
- TrackMan profile은 예측 연도보다 과거 season만 사용한다.
- full-2024는 오염된 잠금 진단이며 source 선택에는 쓰지 않았다.
- v261 calendar는 이번 월 결과에서 고른 것이 아니라 기존 v70에서 선언한 4--9월을 재사용했다.
- Public `1175.9747`은 recipe나 weight 선택에 사용하지 않았다.

## 재현 산출물

```text
artifacts/v255_pitchmix_release_variance_xgb_oof_20260829_01
artifacts/v256_pitchmix_release_fixed_audit_20260829_01
artifacts/v257_count_conditioned_trackman_xgb_oof_20260829_01
artifacts/v258_batter_trackman_exposure_xgb_oof_20260829_01
artifacts/v259_independent_feature_family_audit_20260830_01
artifacts/v260_mechanism_complement_experts_20260830_01
artifacts/v261_regular_calendar_expert_gate_20260830_01
artifacts/v265_runtime_command_oof_20260830_01
artifacts/v265_runtime_batter_oof_20260830_01
artifacts/v266_runtime_calendar_expert_audit_20260830_01
artifacts/v267_pseudo_deployment_xgb_oof_20260830_01
artifacts/v268_pseudo_deployment_route_audit_20260830_01
artifacts/v269_pseudo_deployment_lightgbm_oof_20260830_01
artifacts/v270_pseudo_deployment_diversity_audit_20260830_01
artifacts/v271_pseudo_deployment_calendar_audit_20260830_01
artifacts/v272_pseudo_deployment_new_route_audit_20260830_01
artifacts/v274_pseudo_command_oof_20260830_01
artifacts/v274_pseudo_batter_oof_20260830_01
artifacts/v275_pseudo_trackman_expert_route_audit_20260830_01
```

## 최종 적합과 제출 패키지

v276은 2021--2024 pseudo-runtime 정규시즌 블록 881,587행으로 114피처 XGB를 최종
적합했다. 원본 v244 ZIP은 변경하지 않고 아래 별도 v277 패키지를 만들었다.

```text
artifacts/v276_finalize_pseudo_deployment_xgb_20260830_01
artifacts/v277_pseudo_deployment_package_20260830_01/submit_v277_pseudo_deployment_calendar.zip
```

- 최종 모델 SHA-256: `9560464FCE645BBA8655D58B1023AF9E19D616B4FF0C04EB2CF524ACF94F8831`
- ZIP 크기: `86,539,171` bytes (`82.530 MiB`)
- 파일 수: `181`
- ZIP SHA-256: `CFC28C9C9EEB5D041CAA3D7120C472D97BD9133CA40F7D9A152276583D39CD57`
- CRC: 통과
- 정적 행 독립성: 금지 연산 0개, 통과
- 동적 행 독립성: singleton/shuffle/partition 최대 절대차가 모두 `0.0`, 통과
- 5행 실행: `11.389초`, finite/range 통과
- 245,789행 규모 실행: `237.366초`, finite/range 통과
- 전체 테스트: `610 passed, 21 skipped, 1 existing warning`

감사 환경의 scikit-learn은 1.6.1이고 v244 원본 전처리기는 1.7.2에서 직렬화되어
`InconsistentVersionWarning`이 발생했다. 예측과 모든 감사에는 통과했으며, 새 v276 XGBoost
모델에서 생긴 경고는 아니다.

## 승격 판단

v277은 2026-08-30 02:04:58에 `submit_v277.zip`으로 제출되었고, 공식 Public
`1175.3125589868`, 실행시간 `116초`를 기록했다. 이는 v244보다 `-0.6621243253` 낮고
목표 `1180`까지 `4.6874410132` 남은 결과다.

세 strict-forward 평균이 모두 양수였던 사전 증거가 Public으로 전이되지 않았으며,
full-2024의 월별 일관성 부족과 세 종류 cluster bootstrap p05 음수가 실제 경고로
확인되었다. v277은 **기각·미승격** 처리하고 v244를 공식 champion으로 유지한다. 이
결과를 이용한 같은 pseudo-deployment 계열의 Public 재튜닝은 중단하며, 이후 후보는
새 특성 표현·모델군·검증 계약 중 적어도 하나가 독립적인 경로에서만 탐색한다.
