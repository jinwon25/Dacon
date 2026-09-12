# Public 749.524949 후속 연구 결과

연구 사이클과 검증을 완료했으나 사전 정의한 통계 게이트를 모두 통과한 신규 후보는 없었다. Damped base-rate 후보는 다년 통합 성능과 모든 fold에서 개선됐지만, 2024년 최소 개선 기준에 미달해 승격하지 않았다. 따라서 신규 ZIP은 생성하지 않고 기존 submit.zip을 incumbent로 유지했다.

실행 기간: 2026-08-05~2026-08-06
incumbent: 제출 35748, submission1, Public 749.5249490965

## 1. incumbent 보존과 제출 기록

- 원본 submit.zip은 수정하지 않았다.
- artifacts/incumbent/submission1_submit.zip과 artifacts/incumbent/model/에 체크포인트를 만들었다.
- 원본과 복사본 SHA-256은 모두 77F96448A7149F385F3A800E9238957FD9940C434AD73F0EE14EB0DA6D47AB60이다.
- 제출 이력은 reports/submissions.csv, 체크포인트 메타데이터는 artifacts/incumbent/manifest.json에 기록했다.
- 새 commit과 push는 수행하지 않았다.

## 2. 문헌에서 실제 채택한 원칙

30건의 1차 문헌을 reports/literature_review.md에 검토했다.

- random split 모델 순위를 사용하지 않고 expanding-window season split을 사용한다.
- calibration과 blend는 평가할 outer year보다 앞선 순방향 OOF에서만 fit한다.
- Brier를 reliability, resolution, uncertainty로 분해하고 calibration intercept/slope를 기록한다.
- CatBoost ordered categorical, empirical-Bayes backoff, logloss 대 L2 objective를 비교한다.
- 6개 시즌 base rate에는 last value, 최근 평균, logit trend, damped trend만 사용한다.
- xCTRL은 개인화와 표본 신뢰도 원칙만 참고하고 현재 데이터에 직접 구현하지 않는다.
- TrackMan은 메인 선수 ID 직접 조인 0%라는 기존 감사 결과 때문에 사용하지 않는다.

## 3. validation

primary outer fold는 고정된 다음 네 개다.

1. train 2019~2020 → validate 2021
2. train 2019~2021 → validate 2022
3. train 2019~2022 → validate 2023
4. train 2019~2023 → validate 2024

2019 → 2020은 nested calibration/blend용 진단 fold로만 사용했다. exact game ID가 없으므로 game split을 했다고 주장하지 않는다. 모든 category map, entity 통계, annual rate, calibration 계수와 ensemble weight는 outer validation보다 앞선 데이터에서만 생성했다.

통합 비교는 fold 평균 delta, 고정 recency weight 1:2:3:4, 2024 delta, 최악 fold delta와 paired pitcher-season bootstrap을 사용했다.

## 4. Wave 0: incumbent 다년 재평가

| model | 2021 | 2022 | 2023 | 2024 |
| --- | ---: | ---: | ---: | ---: |
| LightGBM raw | 0.247065896 | 0.244261505 | 0.251331034 | 0.249005893 |
| RandomForest raw | 0.245607690 | 0.243987260 | 0.251651326 | 0.248768795 |
| fixed 0.35/0.65 raw blend | 0.245870442 | 0.243872783 | 0.251197992 | 0.248686180 |
| incumbent fixed blend + trend | 0.246945046 | 0.244169346 | 0.251137482 | 0.248460961 |

incumbent의 4-fold 평균 Brier는 0.247678209, recency-weighted Brier는 0.248254003이다.

- 3-season logit trend는 raw blend 대비 2021 +0.001074604, 2022 +0.000296563으로 악화했다.
- 2023에는 -0.000060510, 2024에는 -0.000225219 개선했다.
- RF/LGB 상관은 연도별 0.679~0.887로 크게 변했다.
- 2024 post-hoc 최적 LGB weight는 0.35였지만 2021, 2022, 2023은 각각 trend 기준 0.11, 0.47, 0.62였다.
- 2023 target rate는 0.499957, prediction mean은 0.524515였다. 이 fold의 local Skill은 0이었다.

## 5. Brier decomposition과 calibration 상태

20-bin 진단 기준이다. binning에 의존하는 근사 분해이므로 Brier 자체를 모델 선택 지표로 유지했다.

| year | Brier | reliability | resolution | uncertainty | intercept | slope |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.001797 | 0.003384 | 0.248927 | 0.135540 | 1.678147 |
| 2022 | 0.244169346 | 0.000621 | 0.005329 | 0.249164 | 0.062019 | 1.306756 |
| 2023 | 0.251137482 | 0.002191 | 0.000938 | 0.250000 | -0.031908 | 0.319452 |
| 2024 | 0.248460961 | 0.000068 | 0.001238 | 0.249807 | -0.004057 | 1.201088 |

calibration slope가 시즌마다 크게 달라 하나의 전역 map을 다음 시즌에 옮기는 전략은 불안정했다.

## 6. Wave 1: base-rate drift와 calibration

### Base-rate forecast

- raw/last는 2021과 2022에서 각각 0.001074604, 0.000296563 개선했다.
- damped trend 3-season, phi=0.8은 사용 가능한 2022~2024에서 모두 개선했다.
- 2021 raw fallback을 포함한 후보의 recency-weighted delta는 -0.000158763이었다.
- 그러나 2024 개선은 -0.000005043으로 사전 기준 -0.000010000에 미달했다.
- damping 미세조정 구간은 2024 pitcher-season bootstrap 개선확률이 약 78%여서 사용하지 않았다.

### Nested calibration

| method | 2021 delta | 2022 delta | 2023 delta | 2024 delta |
| --- | ---: | ---: | ---: | ---: |
| intercept-only | +0.002153041 | -0.000147165 | +0.000529914 | +0.000005953 |
| Platt | +0.002171718 | +0.000147336 | +0.000563282 | +0.000098229 |
| Beta, L2=0.01 | +0.002024062 | -0.000099872 | +0.000518978 | +0.000019485 |
| isotonic | 진단 자료 부족 | +0.000677906 | +0.000504136 | +0.000046457 |

모든 calibration family가 통합 성능을 악화했다. identity map을 유지한다.

## 7. Wave 2: hierarchical backoff

| model | 2021 | 2022 | 2023 | 2024 |
| --- | ---: | ---: | ---: | ---: |
| leaf 50 / parent 100 | 0.247991506 | 0.246950488 | 0.251389392 | 0.251521659 |
| leaf 100 / parent 200 | 0.247978707 | 0.246971373 | 0.251303208 | 0.251440704 |
| leaf 200 / parent 500 | 0.247992604 | 0.247062565 | 0.251196841 | 0.251356480 |

모든 standalone prior가 모든 fold에서 약했다. 평균 incumbent 상관은 0.70~0.72였지만 2024 constrained blend 최적 weight는 모두 0이었다. official as-of history가 entity signal 대부분을 이미 담고 있는 것으로 해석한다.

## 8. Wave 3: CatBoost native categorical

| model | 2024 Brier | delta | best iter | fit sec | peak RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth 6 | 0.249921424 | +0.001460463 | 5 | 427.999 | 4305.2 |
| depth 7, stronger reg | 0.249895795 | +0.001434834 | 4 | 526.274 | 5451.7 |
| depth 8 | 0.249842550 | +0.001381589 | 14 | 699.434 | 5845.4 |

세 설정 모두 극초반에 best iteration이 형성됐다. incumbent 상관은 0.57~0.58이었으나 convex blend 최적 weight가 0이므로 전체 연도와 seed 확장을 중단했다.

## 9. Wave 4: Brier-aligned objective와 RF

| model | 2024 Brier | delta |
| --- | ---: | ---: |
| LightGBM binary logloss | 0.250908500 | +0.002447539 |
| LightGBM regression L2 | 0.250908475 | +0.002447514 |
| XGBoost binary logistic | 0.250121179 | +0.001660217 |
| XGBoost squarederror | 0.250083266 | +0.001622305 |
| RF 300, depth 12, leaf 200 | 0.249322106 | +0.000861145 |
| RF 300, depth 12, leaf 100 | 0.249337544 | +0.000876583 |
| RF 300, depth 10, leaf 200, max_features 0.5 | 0.249490832 | +0.001029871 |

직접 Brier 최적화가 개선을 보장하지 않았다. 세 RF 설정도 개선되지 않았고 마지막 설정은 fit 1399.904초였다.

## 10. Wave 5: nested OOF ensemble

| candidate | d2021 | d2022 | d2023 | d2024 | recency delta | worst | 결합 pitcher-season bootstrap P | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| damped 0.8, raw fallback | -0.001075 | -0.000227 | -0.0000129 | -0.0000050 | -0.0001588 | -0.0000050 | 1.000 | fail: 2024 |
| raw blend | -0.001075 | -0.000297 | +0.0000605 | +0.0002252 | -0.0000585 | +0.0002252 | 1.000 | fail: 2024/worst |
| nested regime, L2=0.001 | 0 | -0.000097 | +0.0000113 | +0.0000347 | -0.0000019 | +0.0000347 | 0.970 | fail |
| nested trend, L2=0.01 | -0.000010 | +0.000011 | -0.0000037 | +0.0000150 | +0.0000060 | +0.0000150 | 0.066 | fail |
| nested logit stack, C=0.001 | +0.000907 | +0.000707 | +0.000359 | +0.000273 | +0.000449 | +0.000907 | 0.000 | fail |

규제 logit stacking은 calibration drift를 다시 학습해 악화했다. convex weight도 2021~2022 raw 우세에 끌려 2024 trend 이득을 충분히 보존하지 못했다. 표의 P는 2,000회 bootstrap resample에서 delta가 음수였던 비율이지 실제 개선 확률 100%를 뜻하지 않는다.

### Damped 후보 bootstrap 재감사

피드백 이후 고정된 phi=0.8 후보만 10,000회 paired cluster bootstrap으로 재감사했다.

| 범위 | cluster | 관측 delta | 95% bootstrap delta CI | 음수 resample | empirical P | P의 Monte Carlo 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | pitcher-season | -0.000005043 | [-0.000013, +0.000002] | 9,167 / 10,000 | 0.9167 | [0.9111, 0.9220] |
| 2024 | pitcher | -0.000005043 | [-0.000013, +0.000002] | 9,167 / 10,000 | 0.9167 | [0.9111, 0.9220] |
| 2021~2024 | pitcher-season | -0.000328261 | [-0.000405, -0.000255] | 10,000 / 10,000 | 1.0000 | [0.9996, 1.0000] |
| 2021~2024 | pitcher | -0.000328261 | [-0.000402, -0.000257] | 10,000 / 10,000 | 1.0000 | [0.9996, 1.0000] |

단일 연도에서는 pitcher-season이 pitcher와 같은 군집이므로 2024 두 행이 동일한 것이 정상이다. 결합 연도에서는 1,549 pitcher-season과 670 pitcher로 결과가 모두 개선 방향이지만, 2024 delta CI가 0을 포함해 최신 연도의 효과 크기는 불확실하다. 게임 식별자가 없어 경기 단위 상관은 반영하지 못했다. 세부 수치는 reports/damped_bootstrap_audit.csv에 보존했다.

phi 선택 절차도 재감사했다. 0.5와 0.8은 Wave 1 실행 전 configs/followup.json에 명시된 이산 후보였으므로 연속적인 사후 미세조정은 아니었다. 그러나 Wave 5에서 0.8을 남긴 결정은 네 outer fold 결과를 본 뒤 이뤄졌고, 각 outer fold에서 이전 rolling-origin 결과만으로 phi를 선택한 nested rule은 아니었다. 따라서 선택편향이 남으며, 위 bootstrap 수치를 untouched test 수준의 확증으로 해석하지 않는다. 148개 실험이 반복 평가된 네 outer year도 이제 개발 데이터로 취급한다.

## 11. 통계적 불확실성과 누수 점검

- recency weight 1:2:3:4를 결과에 맞춰 변경하지 않았다.
- outer target은 scoring, decomposition, bootstrap에만 사용했다.
- calibration과 blend fit에는 outer year target이 들어가지 않는다.
- test 행, 빈도, prediction mean, 순서와 batch 통계를 사용하지 않았다.
- batch independence, unknown category, nested split, calibration fit/apply 분리, 확률 범위, row order를 포함해 22개 테스트가 통과했다.
- exact game ID가 없어 game blocking 결과를 보고하지 않았다.
- 2024에서 사후 최적인 조합은 후보 승격 근거로 사용하지 않았다.
- phi=0.8의 최종 보유 결정에는 outer-fold 선택편향이 있으므로 다음 사이클에서는 사전 고정된 단일 rolling-origin rule만 평가한다.

## 12. TrackMan

TrackMan은 사용하지 않았다. 선수 ID 직접 조인 커버리지가 0%, 기존 league-context 모델도 약했으며, 공식 키 없이 선수별 xCTRL/GMM을 연결하면 규칙 위반과 잘못된 매핑 위험이 있다.

## 13. 제출 게이트와 패키지 결정

유망한 개선 신호는 있었지만 새 후보 중 다음 네 조건을 모두 통과한 모델은 없다.

1. recency-weighted mean delta ≤ -1e-5
2. 2024 delta ≤ -1e-5
3. 모든 fold delta ≤ +5e-5
4. pitcher-season bootstrap P(improve) ≥ 0.90

따라서 submit_candidate_*.zip은 생성하지 않았다. 이는 사전 정의된 제출 정책의 결과다. incumbent submit.zip은 그대로 유효하며 SHA-256도 변하지 않았다.

## 14. 자원

- Wave 0 전체 5-fold 재학습: 682.8초.
- Wave 0 관측 peak RSS: 약 3.5GB.
- CatBoost pilot peak RSS: 최대 약 5.85GB.
- XGBoost pilot peak RSS: 최대 약 3.36GB.
- RF 고비용 pilot peak RSS: 약 2.85GB.
- incumbent 실제 서버 추론: 5초, ZIP 4.055MB, peak RSS 약 718.4MB.
- 새 제출 모델이 없어 새로운 서버 시간이나 ZIP 크기를 만들어 보고하지 않는다.

재현성 스냅샷은 artifacts/research_snapshot_20260806/에 보존한다. code_snapshot.zip은 코드·설정·테스트·보고서의 결정적 ZIP이고, reports/reproducibility_manifest.csv에는 원본 데이터, config, incumbent/follow-up 모델, OOF cache, 제출 ZIP별 크기와 SHA-256을 기록한다. 프로젝트가 상위 저장소에서 아직 untracked 상태라 표준 git diff는 비어 있으며, 이 사실과 전체 untracked 목록을 git_diff.patch와 git_status.txt에 함께 보존한다. 원본 submit.zip, incumbent 체크포인트, 검증에 사용한 submit.zip은 동일 파일이며 두 경로의 SHA-256이 모두 77F96448A7149F385F3A800E9238957FD9940C434AD73F0EE14EB0DA6D47AB60인지 manifest에서 다시 검사한다.

## 15. 다음 실험 우선순위

| 순위 | 실험 | 기대 효과 | 구현 비용 | 누수 위험 |
| ---: | --- | --- | --- | --- |
| 1 | 한 번만 사전 정의한 rolling-origin damped forecast ensemble | 최신 drift 안정화; 2024 게이트 재미달 시 종료 | 낮음 | 중간 |
| 2 | 고정 RF 설정의 3-seed 확률 단순 평균 | Monte Carlo variance 감소 | 높음 | 낮음 |
| 3 | 2023 collapse의 game_type/team invariant ablation | 2023·2024 방향이 같을 때만 후보화 | 중간 | 낮음 |
| 4 | incumbent LGB에 count/uncertainty 최소 피처 최대 2개 변형 | 기존 분포와 resolution 보존 | 낮음 | 낮음 |
| 5 | 표본 크기별 RF/LGB weight | 여러 fold 상대우위가 일관될 때만 실행 | 중간 | 중간 |

CatBoost, hierarchical backoff, squared-error GBDT와 RF 하이퍼파라미터 대안은 다음 사이클에서 재개하지 않는다.
