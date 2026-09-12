# v25 post-break R_ANCHOR 연구 및 Public 결과 — 2026-08-17

## 결론

`submit_v25.zip`은 DACON 코드 제출 API에서 `isSubmitted=true`, `detail=Success`를 받았고 공식 Public **1155.8293405409**를 기록했다. 제출 ID는 `1535141`, 확인 당시 순위는 **12위**다.

| 항목 | 값 |
|---|---:|
| v25 Public | **1155.8293405409** |
| v22 Public | 1153.0436023798 |
| v22 대비 개선 | **+2.7857381611** |
| 10위 컷 | 1157.9594495591 |
| 10위까지 | **2.1301090182** |
| 11위까지 | **1.4938562805** |
| 1200까지 | **44.1706594591** |
| 제출 시각 | 2026-08-17 03:37:45 KST |
| 누적 제출 | 24회 |
| ZIP SHA-256 | `60CF13B28AF01B24A49E2B03E13161F2E04A1C09647D8DA2CFD95407ED9335B1` |

Public이 기존 챔피언을 명확히 넘었으므로 v25를 새 챔피언으로 승격한다. v22는 v25의 직접 부모로 루트에 보존한다.

## 모델 가설

F는 2022→2023에 reverse 실패율과 타깃 분포가 크게 바뀌었고, R_CORE는 기존 다단계 잔차 모델이 이미 강하게 설명한다. 반면 팀 13이 한쪽에만 있는 정규시즌 `R_ANCHOR`는 기존 v22 위에 독립적인 저자유도 직접확률 신호가 남아 있었다.

v25는 다음처럼 범위를 좁혔다.

- 학습: 2024년 행만 사용한 spline + logistic direct-probability model
- 제외: pitcher ID, batter ID, 현재 투구의 구종·TrackMan 결과, 다른 test 행 집계
- 적용: `game_type=R`이고 투수팀 또는 타자팀이 13인 `R_ANCHOR` 행만
- 혼합: `v22 + 0.075 × (direct - v22)`
- 보호: R_CORE와 F는 v22와 동일

최종 추론은 현재 행의 48개 공식 입력과 동봉된 train-only 모델만 사용한다.

## 분리된 시간축 검증

모델·도메인은 2023 전반→후반에서 선택했고, 선택을 고정한 뒤 2023 전체→2024 전체를 외부연도 감사로 열었다. 별도로 2024 전반→후반에 같은 구조를 재학습해 최근 시즌 적응성을 확인했다. 수치는 대회 Brier Skill Score 점수 차이다.

| 축 | v22 대비 gain | 적용 도메인 gain | 양수 active-month 비율 | 최악 active month |
|---|---:|---:|---:|---:|
| 2023 전반→후반 선택 | **+31.4866** | +181.0753 | 100.0% | +20.1913 |
| 2023 전체→2024 전체 외부감사 | **+1.7500** | +9.9353 | 85.7% | -3.7395 |
| 2024 전반→후반 재현 | **+0.8211** | +4.6961 | 100.0% | +0.0605 |

외부연도에서는 7개 active month 중 6개가 양수였고 9월만 `-3.7395`였다. 이 약점을 감안해 로컬 평균이 더 큰 강한 혼합 대신 7.5%를 고정했다. 최적화 반복 상한을 400→800으로 두 배 늘린 민감도 검사에서도 외부연도 `+1.7600`, 2024 후기 `+0.8283`으로 결과가 거의 같아 수렴 경고가 결론을 바꾸지 않았다.

## 시도했지만 기각한 계열

v22 이후 과거 결과에 얽매이지 않고 여러 독립 신호를 다시 검사했다.

- 실패유형 privileged profile: 2023 후기는 크게 개선됐지만 동결 2024에서 `-89.73`; 저자유도 3단계 버전도 `-51.52`
- latent pitch-type 원본 student: 2023 후기는 `+18.71`, 2024는 `-0.80`
- pitch-type empirical-Bayes student: 2023 후기는 `+2.13`, 2024는 `-1.34`
- pitcher/batter ID 추가 pitch-type LightGBM: 두 해 모두 분류 정확도·log-loss가 기존 student보다 악화
- 2,080개 실패 profile의 사후 강건 탐색: 세 시간축 최소 gain은 양수인 조합이 있었지만 세부 그룹 최소 gain `-0.3061`로 제출 금지

이 결과는 더 복잡한 teacher/student나 ID 모델보다, 도메인 변화를 격리한 작은 직접확률 보정이 현재 챔피언 위에서 더 잘 전이됐음을 보여준다.

## 패키지·규정 검증

`submit_v25.zip`은 v22 해시를 고정한 완전한 자식이다. 새 파일은 7.8KB 모델과 1.5KB 명세뿐이다.

- 전체 자동 테스트: **112 passed**
- 245,789행 실행: **58.862초**
- peak RSS: **1,446.5MB**
- 수식 재현 최대 오차: `2.22e-16`
- 배치 분할 최대 오차: `1.11e-16`
- R_CORE/F 부모 대비 최대 차이: 각각 `2.22e-16` / `0`
- 확률 범위·유한성, ZIP 계보, 행 순서, 네트워크 미사용, 외부 데이터 미사용: 모두 통과

상세 수치는 `artifacts/v25_postbreak_anchor_audit_20260817_01/summary.json`, `reports/v25_validation.json`에 있다.

## 참고 근거

- [DACON 대회 설명](https://dacon.io/competitions/official/236743/overview/description)
- [DACON 평가 산식](https://dacon.io/competitions/official/236743/overview/evaluation)
- [DACON 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [DACON TrackMan·ASOF·리더보드 활용 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082)
- [DACON test 행 독립 추론 공지](https://dacon.io/competitions/official/236743/talkboard/417123)
- [Menon et al., A statistical perspective on distillation, ICML 2021](https://proceedings.mlr.press/v139/menon21a.html)
- [Wang et al., Robust distillation for worst-class performance, UAI 2023](https://proceedings.mlr.press/v216/wang23e.html)
- [Karlsson et al., Using time-series privileged information for provably efficient learning, AISTATS 2022](https://proceedings.mlr.press/v151/k-a-karlsson22a.html)

문헌의 핵심은 teacher 확률 품질과 세부 그룹 calibration이 중요하다는 점이다. 이에 따라 latent teacher 계열은 평균 gain만으로 채택하지 않고 월·도메인·손·카운트 하위그룹을 함께 열어 기각했다. 최종 v25는 privileged 정보에 의존하지 않는다.

## 다음 사이클 원칙

v25의 Public 개선으로 R_ANCHOR post-break 축은 유효하다고 판정한다. 다만 강도를 리더보드에 맞춰 연속 조절하지 않는다. 다음 제출은 남은 `2.1301`을 겨냥하되, v25와 다른 생성 구조이거나 시간축·월·하위그룹에서 더 강한 독립 검증을 통과한 경우에만 허용한다.
