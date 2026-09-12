# v84 실제 제출 결과와 1170 후속 계획 — 2026-08-22

## 결론

v84는 DACON 코드 제출 API에서 정상 접수됐고 Public **1161.2020600422**를 기록했다.
이전 챔피언의 정확 점수 `1159.3352239501` 대비 **+1.8668360921**의 유의미한
개선이다. 제출 이력 ID는 `59988`, 공식 리더보드 내부 record는 `1543508`, 확인 당시
순위는 13위, 팀 제출 횟수는 29회다.

운영 판단은 **v84 승격, 단일 챔피언 1161로 교체**다. 1170까지 남은 격차는
`8.7979399578`이다.

## 새 단일 챔피언

- 파일: `artifacts/standalone_champion_1161/standalone_champion_1161.zip`
- 제출 당시 파일명: `submit_v84_probe.zip`
- SHA-256: `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`
- 크기: 34,629,670 bytes
- 루트: `script.py`, `requirements.txt`, `model/`
- 외부 부모 ZIP 의존성: 없음
- API 응답: `isSubmitted=true`, `detail=Success`
- 공식 Public: `1161.2020600422`
- 공식 제출 실행 시간: `41초`

파일명만 canonical champion 이름으로 바꾸고 ZIP 바이트는 제출본과 동일하게 보존한다.
직전 `standalone_champion_1159`는 감사·회귀용 부모로만 남긴다.

## 고정 레시피와 감사

v84는 Public 결과를 보고 조정하지 않은 기존 v56 레시피를 한 해 이동해 복원했다.

- source: full-2023 245,525구 + late-2024 76,896구
- source F 표본: 25,686구 + 9,570구
- model: main-effect-free pairwise FM, rank 16
- risk: source-period×domain 총 가중치 균등
- route: F만 활성
- 적용: parent logit에 `0.10 × clip(correction, -0.25, 0.25)` 추가
- 추론: PyTorch를 제거한 NumPy 임베딩 내적

학습 가중 로그손실은 4 epoch 동안 `0.70025466 → 0.68468368`로 감소했다. Torch와
NumPy 내보내기의 최대 오차는 `2.48e-07`이었다. 패키지 감사 결과는 다음과 같다.

| 감사 항목 | 결과 |
|---|---:|
| 수식 parity | `5.55e-17` |
| 보호 R 행 변화 | `0.0` |
| synthetic F 활성 | `5/5` |
| shuffle 독립성 | `0.0` |
| partition 독립성 | `0.0` |
| 245,789행 프록시 | `68.218초` |
| 전체 테스트 | `282 passed, 4 skipped` |

공식 600초 제한보다 엄격한 내부 120초 guard를 통과했다. 테스트 행 집계·순서·배치
통계를 모델 입력으로 사용하지 않는다.

## OOF에서 Public으로의 전이

v82 exact OOF 부모 위에서 v56 frozen shift는 full-2024 `+1.0181164`, late-2024
`+2.9165519`였다. 실제 Public gain은 `+1.8668361`로 두 진단축 사이에 위치한다.

- full-2024 대비 transfer ratio: `1.8336`
- late-2024 대비 transfer ratio: `0.6401`

이는 F 도메인이 실제 평가에 존재하고, main-effect를 제거한 공유 상호작용이 2025에도
전이됐다는 강한 증거다. 다만 이 한 건을 다른 family의 전이계수로 사용하지 않는다.
성공 레시피의 rank, risk, eta, clip, center, seed는 모두 동결하며 Public 기반 미세탐색을
하지 않는다.

## 야구 도메인 해석

F는 R과 투수 운용, 타자 수준, 카운트 분포, 이닝 압박이 다르다. 단순 F 평균 보정은 과거
Public에서 실패했지만, v84는 평균 효과를 제거하고 다음 상호작용의 공통 저차원 구조만
남겼다.

- 투수×타자
- 투수×카운트
- 투수×손잡이
- 투수×경기 도메인
- 타자×손잡이/도메인
- 팀 매치업
- 카운트×주자/압박
- 도메인×압박
- 이닝 구간×압박

두 source 기간의 F correction center가 크게 달랐는데(`-0.7193`, `-0.0892`), 두 기간 평균
center를 제거했기 때문에 season-level calibration drift를 직접 이월하지 않았다. 이 설계가
과거의 단순 F offset 실패와 v84 성공을 가르는 핵심으로 해석한다.

## 1170 후속 우선순위

남은 `8.79794`는 같은 F 레시피를 더 세게 적용해 메우지 않는다. 성공한 F 축은 보호하고,
서로 겹치지 않는 R_CORE/R_ANCHOR 신호를 추가한다.

### 1순위 — 팀원 모델 exact OOF 결합

팀 브랜치들의 최고 모델 중 현재
챔피언과 다른 예측을 만드는 모델에 대해 full-2023, late-2023, full-2024, late-2024의
`row_id/target/prediction`을 받는다. ZIP이나 Public 점수만으로 결합하지 않는다.

각 벡터에 대해 다음을 계산한다.

- v84 residual과 prediction delta의 상관
- full/late 축의 analytic headroom과 비음수 최적 가중치
- R_CORE/R_ANCHOR/F, 월, 투수, 타자 교차 블록 gain
- pitcher bootstrap, crossed pitcher×batter bootstrap, month block bootstrap p05

F를 건드리지 않는 R 후보를 우선하고, F가 포함된 팀원 모델은 v84와의 조건부 headroom이
명확할 때만 허용한다.

### 2순위 — R_CORE 동적 계층 불확실성 모델

현재 strict 모델이 이미 사용하는 평균 EB 효과를 반복하지 않고 posterior uncertainty
interaction을 만든다.

- pitcher × count leverage(3-ball, 2-strike, neutral)
- pitcher × batter hand × inning bucket
- 최근 표본수와 장기 표본수의 불일치
- 성공률 평균이 아니라 Beta-Binomial posterior SD와 변화 확률

각 outer origin 안에서 prior strength, half-life, calibration을 다시 학습하는 nested OOF만
primary로 인정한다. F와 R_ANCHOR를 기본 보호하고 R_CORE에서만 작은 logit correction을
검증한다.

### 3순위 — 다중 실패유형 일관 결합

단일 성공 이진분류 대신 실패를 middle/reverse/way-off 계열로 분해하고 multinomial 또는
Dirichlet 구조로 동시에 추정한 뒤 `P(success)=1-ΣP(failure type)`로 복원한다.
count·손잡이·주자·이닝 상호작용을 행 단위로 사용한다. TrackMan은 test 직접 join이 아니라
학습 시 투수 표현을 정규화하는 privileged teacher 또는 measurement-error weight로만
제한한다.

### 4순위 — F 성공 구조의 설명용 안정성 감사

점수 향상을 위한 재튜닝이 아니라 설명·재현을 위한 감사다.

- 2022→late-2023→2024→2025의 F 비중 및 correction coverage 정리
- unseen pitcher/batter/team 비율과 clipping 비율
- source-period별 center 제거 전후 gain 분해
- pair interaction ablation은 OOF 설명용으로만 수행하고 Public 후보를 만들지 않음

## 중단선

- v84 F eta `0.05/0.15`, rank, seed, center, clip Public line search 금지
- strict 가중치 추가 증량: v83 full/late 모두 음수이므로 금지
- v50 low-rank context: late 안정성과 월 안정성이 부족해 금지
- 전역 scalar calibration, domain×count lookup, 기존 TrackMan profile 강도 재탐색 금지
- 2024 label이나 v84 Public을 이용한 동일 family hyperparameter 선택 금지

## 다음 제출 게이트

다음 후보는 아래를 모두 충족할 때만 단일 ZIP으로 실제 제출한다.

1. exact v84 OOF 부모와 행·target·hash parity
2. full-2024와 late-2024 추가 gain 모두 양수
3. 두 축의 월 양수 비율 각각 75% 이상
4. 최악 월 gain `>-5`, 최소 적용 R domain gain `>=0`
5. pitcher, crossed pitcher×batter, month block bootstrap p05 모두 양수
6. final-family Reality Check `p<0.10`
7. singleton/shuffle/partition 오차 `<=1e-12`
8. 245,789행 내부 120초 guard 통과
9. 과거 ZIP 없이 `script.py + requirements.txt + model/`만으로 실행

## 공식 확인 경로

- 리더보드: https://dacon.io/competitions/official/236743/leaderboard
- 평가 산식: https://dacon.io/competitions/official/236743/overview/evaluation
- 평가 행 독립성 공지: https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent
- 코드 제출 주의사항: https://dacon.io/competitions/official/236743/talkboard/417157?dtype=recent&page=1
