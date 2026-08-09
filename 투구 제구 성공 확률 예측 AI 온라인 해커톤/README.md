# Aimers 9기: 투구 제구 성공 확률 예측

공식 데이터만 사용한 재현 가능한 확률 예측 베이스라인이다. 2019~2023 학습 / 2024 검증에서 모델을 비교하고, 모든 test 행을 서로 독립적으로 처리하는 `submit.zip`까지 생성한다.

## 2026-08-09 후속 상태

- incumbent `submit.zip`은 Public **749.5249490965**와 SHA-256 `77F96448...AB60` 상태로 보존되어 있다.
- 익명 투수의 연도별 구종 궤적을 이용한 Trackman 일대일 연결을 새로 구현했다. 직접 ID 교집합은 0%지만 고신뢰 연결의 2024→2025 유지율은 95.65%다.
- R 전용 반감기 1년 RF 95% + rolling-damped Trackman LGB 5% 후보가 2021~2024 네 fold를 모두 개선하고 고정 게이트를 통과했다.
- `submission3 edit` (ID 39023)이 `submit_v2.zip`으로 Public **763.2665303697**을 기록해 incumbent보다 **13.7415812732점** 높았다.
- 후속 통제 후보 `submit_v3.zip`은 v2에서 Trackman blend만 5%→10%으로 바꾼 파일이다. SHA-256은 `6EE2CF45679F94C457A13B4FEFDCCADF3DF71E09B36F294CCF2A0EC6689C4579`이며 기존 ZIP을 덮어쓰지 않는다.
- 최종 제출 ZIP 파일명은 40자 이내, 기본 형식은 `submit_vN.zip`으로 고정한다. 상세 규칙과 이력은 [제출 파일명·이력](reports/submission_naming.md)을 참조한다.

## 현재 결과

Primary validation은 실제 2025 평가를 모사한 season-forward holdout이다. 아래 값은 모두 동일한 2024 validation에서 실제 실행한 결과이며 다른 split의 점수를 섞지 않았다.

| 모델 | Brier Score | Local Brier Skill Score |
| --- | ---: | ---: |
| Train 성공률 상수 | 0.251875047 | 0.000 |
| 계층 prior (alpha=50) | 0.250168787 | 0.000 |
| 공식 RandomForest 재현 | 0.248768795 | 415.574 |
| LightGBM engineered L31 raw | 0.249005893 | 320.661 |
| LightGBM + train-only 3-season trend | 0.248770263 | 414.986 |
| RF + train-only 3-season trend | 0.248548983 | 503.566 |
| 최종 LGB/RF 0.35/0.65 blend | **0.248460961** | **538.802** |

최종 blend 가중치는 2024 OOF에서 0.05 간격으로 선택했기 때문에 보고 점수에 작은 선택 편향 가능성이 있다. Public LB 수료 점수 549.51과 local score는 평가 행이 달라 직접 비교하지 않는다.

## 데이터 결론과 누수 방지

- train은 1,475,092행 × 49열, Trackman은 1,793,078행 × 30열이며 target 비율은 0.523766이다.
- train의 최신 시즌은 2024, 형식 sample test는 2025다. 메인 데이터에는 정확한 날짜·경기 ID·경기 내 투구 번호가 없어 game-group split을 만들 수 없다.
- `asof_pitcher_success_rate`의 다음 행 갱신식은 확인 가능한 1,473,508건에서 100% 직전 target과 일치했다. 현재 행의 공식 `asof_*`만 사용하고 다음 행 또는 test 순서는 참조하지 않는다.
- 메인 `pitcher_id`/`batter_id`와 Trackman 선수 ID의 직접 교집합은 각각 0개다. 후속 연구에서는 target-free 연도별 구종 궤적과 손잡이별 일대일 최적 배정으로 투수만 보수적으로 연결한다.
- 과거 시즌 Trackman league-context 실험은 raw LightGBM보다 나빠 최종 패키지에서 제외했다.
- 범주 사전, global rate, calibration offset은 train/fold-train에서만 만든다. test의 빈도·평균·순위·분포·행 순서·다른 행을 사용하지 않는다.
- 2025 logit offset은 전체 공식 train의 최근 3시즌 연도별 target 비율에서 미리 계산해 `model/ensemble.json`에 저장했다. test batch 통계는 사용하지 않는다.

자세한 수치는 [데이터 감사](reports/data_audit.md), [초기 실험 결과](reports/initial_findings.md), [실험 원장](reports/experiments.csv), [보정 곡선](reports/calibration_curve.csv), [패키지 검증](reports/package_validation.md)에 있다.

## 디렉터리

```text
configs/default.json          # 목적 있는 LightGBM 후보 6개
src/data.py                   # 명시적 compact dtype 로딩
src/features.py               # row-local/과거 시즌 피처
src/validation.py             # season/group/stratified split 도구
src/metrics.py                # 공식 Brier Skill Score와 제출 검증
src/calibration.py            # Platt/isotonic/shrink/trend/blend
src/train.py                  # prior, 공식 RF, GBDT, 보정, 최종 학습
src/trend_experiment.py       # train-only 시즌 추세와 최종 blend 승격
src/audit.py                  # 전수 데이터·누수 감사
src/package.py                # submit.zip 생성/구조 검사
src/validate_package.py       # 실제 archive smoke/runtime 검사
train.py                      # 학습 진입점
script.py                     # 평가 서버 추론 진입점
model/                        # 로컬 최종 산출물(커밋 제외)
reports/                      # 감사·실험·검증 기록
tests/                        # metric/order/parity/row-independence 테스트
```

## 준비

Python 3.11 환경에서 다음 공식 파일을 배치한다.

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/trackman_history.csv
data_description.md
```

의존성은 실제 로컬 설치와 추론을 확인한 버전만 고정했다.

```powershell
python -m pip install -r requirements.txt
```

## 전체 재현

프로젝트 루트에서 순서대로 실행한다.

```powershell
python -m src.audit --project-dir .
python train.py --project-dir . --config configs/default.json
python -m src.trend_experiment --project-dir . --config configs/default.json
python -m pytest -q tests
python script.py
python -m src.package --project-dir . --output submit.zip
python -m src.validate_package --project-dir . --zip submit.zip
```

실험 하나가 끝날 때마다 `reports/experiments.csv`에 append하며 기존 행을 덮어쓰지 않는다. 이미 실험이 끝났고 최선 LightGBM만 재학습해야 한다면 다음 복구 명령을 사용할 수 있다.

```powershell
python train.py --project-dir . --config configs/default.json --final-only --num-rounds 54
```

복구 명령은 raw LightGBM 산출물만 복원한다. 최종 RF blend까지 재현하려면 이후 `src.trend_experiment`를 실행해야 한다.

## 제출 패키지

최종 산출물은 프로젝트 루트의 `submit.zip`이며 정확히 다음 최상위 구조만 포함한다.

```text
model/
script.py
requirements.txt
```

`script.py`는 자신의 위치를 기준으로 `./data/test.csv`를 읽고 `./output/submission.csv`를 쓴다. 공식 5행 sample과 ZIP을 별도 임시 경로에 해제한 smoke test가 통과했다.

대표 245,789행 반복 검증에서 CSV 로드+피처+앙상블 추론은 4.633~7.683초, peak RSS는 최대 718.4MB였다. 보수적으로 7.683초를 기준으로 삼았다. ZIP은 4.055MB, 압축 해제 후 5.032MB다. 이 수치는 현재 Windows/Python 3.11.2 환경의 실측이며 Ubuntu 평가 서버 속도를 보장하는 추정치는 아니다. 다만 600초·28GB 제한에 충분한 여유가 있다.

## 남은 위험

- 실제 2025 성공률이 최근 3시즌 선형 추세에서 벗어나면 고정 logit offset이 악화될 수 있다.
- 2024 한 시즌만 primary holdout으로 사용했고 정확한 game group을 구성할 컬럼이 없다.
- 최종 0.35/0.65 가중치는 동일 2024 OOF에서 선택·평가돼 선택 편향 가능성이 있다.
- 5행 test는 실제 평가의 신규 투수·타자 비율과 범주 분포를 대표하지 않는다.
- Trackman 투수 연결은 공식 매핑이 아니라 익명 궤적 기반 추론이므로 낮은 신뢰도 연결 오차와 검증 재사용 편향이 남는다.

## Public 749.524949 후속 연구

인컴번트는 artifacts/incumbent/에 보존되어 있다. 다년 walk-forward 후속 실험은 다음 순서로 재현한다.

~~~powershell
python followup.py --waves 0
python followup.py --waves 1
python wave234.py --hierarchical --years 2021 2022 2023 2024
python wave234.py --models lgb_hier_binary lgb_hier_l2 --years 2024
python wave234.py --models cat_d6 cat_d7_reg cat_d8 --years 2024
python wave234.py --models xgb_binary xgb_l2 --years 2024
python ensemble.py
python statistical_audit.py --project-dir . --resamples 10000
python -m src.finalize_followup --project-dir .
python -m pytest -q
python trackman_linkage.py --project-dir . --build-only
python hybrid_candidate.py --project-dir . --resamples 10000
python package_hybrid_candidate.py --project-dir . --output submit_candidate_hybrid_recency_trackman_v1.zip
python -m src.validate_package --project-dir . --zip submit_candidate_hybrid_recency_trackman_v1.zip
python research_snapshot.py --project-dir . --snapshot-name research_snapshot_20260806
~~~

Wave 0 cache는 artifacts/followup/oof/, 추가 모델 OOF는 artifacts/followup/models/에 저장되므로 완료된 fold는 재실행하지 않는다.

초기 follow-up에서는 통계적 제출 게이트를 통과한 모델이 없었지만, 2026-08-08 hybrid 후속 후보는 네 fold와 고정 게이트를 통과했다. 기존 `submit.zip`은 incumbent로 계속 보존하며 최신 판단은 `reports/top_rank_strategy_20260808.md`를 따른다.

현재 프로젝트가 상위 Git 저장소에서 untracked라 일반 git diff만으로 상태를 복원할 수 없다. research_snapshot.py는 코드·설정·테스트·보고서 ZIP, git status/diff, 데이터·config·모델·OOF·submission SHA-256 manifest를 artifacts/research_snapshot_20260806/과 reports/reproducibility_manifest.csv에 생성한다.

## 도메인 기반 제출 후보

한국 야구 제도 변화와 원데이터 game-type regime을 점검하는 명령은 다음과 같다.

~~~powershell
python domain_eda.py --project-dir .
python domain_drift.py --project-dir . --resamples 10000
python rf_seed_experiment.py --project-dir . --years 2023 2024
python domain_feature_experiment.py --project-dir . --years 2023 2024
python domain_feature_experiment.py --project-dir . --years 2021 2022 --variants lgb_domain_profile_v1
python -m src.finalize_domain --project-dir .
python package_candidate.py --project-dir . --output submit_candidate_game_type_regime_v1_clean.zip
python validate_candidate.py --project-dir . --candidate-zip submit_candidate_game_type_regime_v1_clean.zip --incumbent-zip submit.zip
~~~

`submit_candidate_game_type_regime_v1_clean.zip`은 Public 704.1257475401로 실패해 폐기했다. 최신 추천 후보는 `submit_candidate_hybrid_recency_trackman_v1.zip`이며, 자세한 근거는 `reports/top_rank_strategy_20260808.md`에 있다.
