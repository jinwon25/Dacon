# 리더보드 및 다음 제출 전략

> 2026-08-08 후속 실험으로 이 문서의 “incumbent 단독 유지” 판단은 갱신됐다. 최신 판단과 후보 ZIP은 [상위권 전략 보고서](top_rank_strategy_20260808.md)를 기준으로 한다. 기존 내용은 의사결정 이력으로 보존한다.

## 2026-08-09 공개 점수 반영

- `submission3 edit` / ID `39023` / `submit_v2.zip` / 제출 선택 `schedule`
- 제출 시각: `2026-08-08 16:08:48`, 실행 시간: **9초**
- Public Score: **763.2665303697**
- incumbent `749.5249490965` 대비 **+13.7415812732점 (+1.83%)**
- 해석: 5% Trackman + R 전용 최신성 RF 혼합 방향은 공개 평가 구간에서도 유효했다. 다만 한 공개 점수만으로 RF와 Trackman 각각의 기여를 식별할 수 없으므로, 다음 제출은 Trackman 비중만 5%→10%으로 바꾼 통제 실험이다.
- 다음 후보: `submit_v3.zip`, SHA-256 `6EE2CF45679F94C457A13B4FEFDCCADF3DF71E09B36F294CCF2A0EC6689C4579`

## 현재 기준

- incumbent Public Score: 749.5249490965
- package SHA-256: 77F96448A7149F385F3A800E9238957FD9940C434AD73F0EE14EB0DA6D47AB60
- actual server runtime: 5초
- submission2 Public Score: 704.1257475401
- 현재 권고: 이미 제출된 `submit_v2.zip`을 현 공개 기준점으로 삼고, `submit_v3.zip`을 다음 통제 제출 후보로 사용한다. game-type regime 후보는 계속 폐기한다.

## 이번 연구의 제출 판단

damped_3_0.8_fallback_raw가 네 fold 모두 개선했지만, 2024 delta는 -5.04e-6으로 사전 기준 -1e-5에 미달했다. 10,000회 재감사에서 결합 pitcher-season·pitcher bootstrap은 모두 음수였지만 이는 bootstrap 표본에서 관측한 비율이지 실제 개선 확률 100%라는 뜻이 아니다. 2024 단독 empirical P는 0.9167이고 delta의 95% bootstrap CI는 [-1.3e-5, +0.2e-5]로 0을 포함했다. damping을 2024 결과에 맞춰 미세조정하는 것은 validation overfit이므로 후보를 만들지 않았다.

phi=0.5와 0.8은 실행 전 이산 후보로 명시됐지만, 0.8의 최종 보유는 네 outer fold를 본 뒤 결정됐다. 다음 사이클에서는 rolling-origin 선택 규칙 하나를 실행 전에 고정하고, 2024 최소 효과 게이트를 다시 넘지 못하면 해당 방향을 종료한다. 현재 네 outer fold는 148개 실험에 반복 사용돼 사실상 개발 데이터로 취급한다.

CatBoost, hierarchical prior, direct-L2 GBDT, XGBoost와 RF 변형은 2024에서 모두 약했다. post-hoc 2024 convex blend에서도 이 모델들의 최적 weight는 0이었다.

후속 도메인 EDA에서 game_type F의 2022→2023 성공률이 0.708749→0.472904로 구조적으로 바뀌고 2024에도 전체 시즌 대비 음의 residual이 지속된 것을 확인했다. 고정 change-point 규칙은 2021~2023을 incumbent와 동일하게 유지하면서 2024 Brier를 -0.000125059 개선해 기존 제출 게이트를 모두 통과했다. 최종 2025 artifact는 F에 -0.1079835303 logit offset, R에 0을 적용한다.

`submit_gt_v1.zip`을 submission2로 한 번 제출한 결과 Public Score는 704.1257475401로 incumbent보다 45.3992015564 낮았다. 따라서 2023~2024의 F residual을 2025로 그대로 이월하는 가설은 배포 관점에서 기각한다. offset의 크기나 threshold를 이 결과에 맞춰 조정하지 않으며, 동일 계열 후보를 다시 제출하지 않는다.

이 결과는 game-type regime 보정이 실제 평가 기간으로 전이되지 않았다는 강한 분포 변화 증거다. 반대로 `submit_v2.zip`의 공개 점수 상승은 Trackman+recency hybrid 방향을 후속 탐색할 근거를 제공한다. `submit_v3.zip`의 공개 결과가 나오기 전까지는 v2를 안전한 기준점으로 보존한다.

## 다음 제출이 허용되는 조건

- 4-fold recency-weighted delta ≤ -1e-5
- 2024 delta ≤ -1e-5
- 어느 fold도 +5e-5보다 나쁘지 않음
- pitcher-season bootstrap P(improve) ≥ 0.90
- seed 변화 후 방향 유지
- 독립 행, batch invariance, row order, package 구조와 runtime 검증 통과

## 리더보드 운용

- 로컬 게이트를 통과한 명확한 가설만 제출한다.
- 한 제출은 한 가지 변화만 담는다.
- Public 점수로 damping, calibration 또는 blend weight를 연속 미세조정하지 않는다.
- 같은 package 재제출은 서버 재현성 확인 외에는 하지 않는다.
- 결과를 받으면 reports/submissions.csv에 package hash와 가설을 추가한다.

## 현재 얻은 정보

가장 중요한 정보는 복잡한 모델 부족보다 시즌별 base-rate와 calibration 관계가 불안정해 2024 단일 holdout 선택이 위험하다는 점이다. 복잡한 대안은 제거됐고 damped drift만 유망 신호로 남았다. 다음 제출은 모델 class를 늘리기보다 기존 RF/LGB resolution을 보존하면서 drift만 더 안정적으로 다뤄야 한다.
