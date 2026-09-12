# 0.65+ / 상위 10% 후속 모델링 — 2026-07-27

## 결론

공개 최고는 제출 `1502437`, 총점 `0.6461250914`로 유지한다. 방향성
사후보정의 공개 실패를 반영해 기존 계열의 강도 미세조정과 독립 24시간
trajectory 모델을 모두 감사했다.

다음 공개 probe의 1순위는 아래 파일이다.

`artifacts_final/candidates/public_probe_issue_tcn_expanding_g3_q65w05_20260727.csv`

- SHA-256:
  `0c527afaad8fe67a367c1b1ac252cfab03d373c7a92c846763bd0e4b6f8d19d7`
- 기존 공개 최고의 그룹 1·2를 그대로 유지하고 그룹 3만 변경
- 24시간 발행주기를 하나의 표본으로 학습한 3-seed quantile TCN
- 분위수 `0.65`, incumbent 방향 blend weight `0.05`
- 2024 expanding-origin G3 전체 score `+0.00293625`
- 2024 expanding-origin G3 전체 1-NMAE `+0.00106516`
- 2024 expanding-origin G3 전체 FiCR `+0.00480733`
- Q1·Q2·Q3·Q4에서 세 구성요소가 모두 양수
- 12개월 중 10개월 총점 양수
- H2 issue-block bootstrap q05 `+0.00123757`, 양수 비율 `99.8%`
- 2024 최대 이동은 설비용량의 `2.1168%`
- 2025 최대 이동은 설비용량의 `2.6363%`
- 2025의 72개 불완전 feature 행은 issue 간 보간 없이 incumbent 유지
- 단순 로컬 전이 가정 public score `0.64710384`

이 후보는 9월과 10월이 음수이고, `q=0.65, weight=0.05`가 반복 조회한
2024 안정성 감사에서 선택됐으므로 strict OOF 승격이 아니다. 공개 점수는
사용하지 않았지만 등급은 `controlled_exploratory`로 유지한다. 과거 신규
G3 요인이 공개에서 역전된 전력이 있으므로 예상 점수는 보장이 아니다.

2순위 보류 후보는 아래 파일이다.

`artifacts_final/candidates/public_positive_pooled_g2_w10_probe_20260727.csv`

- SHA-256:
  `139a6d4e93d9eaeec3731352f55c9f5e4b145805a5eb14ff0fdcbf978f6192a2`
- 공개에서 양수였던 pooled G2 factor를 weight `0.05`에서 `0.10`으로 확장
- 그룹 1·3은 공개 incumbent와 완전히 동일
- 로컬 G2 전체 score 증분: incumbent factor 대비 `+0.00085547`
- 공개 전이율 단순 외삽 점수: `0.64634108`
- incremental 월별 총점 양수: 8/12
- incremental issue-block bootstrap q05는 음수이므로 1순위가 아님

두 후보 모두 8,760행 형식 검증을 통과했다. 두 factor를 한 CSV에 먼저
합치지 않는다. 그룹 macro 가법성을 이용하려면 각 factor의 공개 부호를
먼저 독립적으로 확인해야 한다.

## 기존 요인 확장 감사

### 그룹 1 pooled + residual

공개 incumbent의 G1 구조는 primary weight `0.20`, residual weight `0.125`다.
연속 grid를 다시 계산한 결과 현재 조합이 2024 전체 score의 국소 최적이었다.
primary 또는 residual 강도를 올린 모든 유의미한 조합은 전체 score가 낮아지고
분기별 구성요소도 음수가 됐다. G1 강도 확장은 닫는다.

공개 효과와 로컬 효과를 비교하면 현재 G1 factor의 per-group 공개 증분은
`+0.00463682`, 로컬 증분은 `+0.01236424`로 전이율은 약 `37.5%`다.
로컬 개선치를 그대로 공개 점수에 더하는 기존 방식은 과대평가였다.

### 그룹 2 pooled

현재 weight `0.05`의 로컬 G2 score 증분은 `+0.00189987`, 격리된 public
per-group 증분은 `+0.00143902`로 전이율은 약 `75.7%`다.

weight `0.10`은 로컬 전체 score `+0.00275534`, 1-NMAE `+0.00067998`,
FiCR `+0.00483070`이다. 하지만 현재 factor 대비 Q1 1-NMAE와 H2 FiCR이
미세하게 음수이고 issue bootstrap이 약해 보류 probe로만 유지한다.

### 기존 KMA 그룹 3 power curve

현재 공개 양수 KMA G3 이동을 1.25배로 확장하면 로컬 전체 G3 score는
`+0.0046357`에서 `+0.0060757`로 증가한다. 그러나 공개에서 현재 factor가
1-NMAE를 낮추고 FiCR로 총점을 올린 구조여서, 추가 강도는 기대 public
증분이 약 `+0.00015`에 불과하고 component 위험이 있다. 별도 CSV를 만들지
않았다.

## 24시간 trajectory TCN

### 구조

- 한 NWP 발행주기의 24개 target hour를 하나의 독립 표본으로 사용
- provided LDAPS/GFS의 풍속·hub-height·그룹 IDW·공간 요약 159개 feature
- 네 개 dilated temporal residual block
- 그룹 1·2·3 공동 masked quantile loss
- quantile `0.50`, `0.65`, `0.80`
- seed `17`, `29`, `41`
- 모델당 60,873 parameter
- test actual generation 및 public score를 학습·정책 선택에 사용하지 않음

한 issue 안의 24시간 예보는 모두 제출 기준시각 이전에 공개된 동일 예보
run에 속한다. 다른 issue의 미래 관측이나 예측을 섞지 않는다.

### 정적 year-forward 결과

2022–2023만 학습해 2024 전체를 고정 예측한 최초 모델은 모든 그룹의 Q1
선택을 통과했지만 G1·G2는 Q2에서 즉시 역전했다. G3는 Q1·Q2가 양수였으나
H2 FiCR가 음수가 되어 승격하지 않았다. 정적 year-forward 결과로 CSV를
만들지 않았다.

### expanding-origin 결과

생산 대칭성을 위해 각 분기 직전 자료로 TCN을 새로 학습했다.

- 2022–2023 학습 → 2024 Q1
- Q1까지 학습 → Q2
- Q2까지 학습 → Q3
- Q3까지 학습 → Q4

Q1 최대점 정책은 G3 `q=0.80, weight=0.30`이었으나 Q4에서 score
`-0.005686`, FiCR `-0.010617`로 실패했다. 따라서 Q1 최대점 자동선택
계약은 폐기했다.

안정성 표면 감사에서는 G3 `q=0.65, weight=0.05`가 네 분기에서 모든
구성요소 양수였고 bootstrap도 통과했다. 이는 후반 fold를 본 뒤 선택한
정책이므로 strict 승격이 아니라 공개 단일-factor probe로만 사용한다.
G1·G2 trajectory 계열은 닫는다.

## 제출 프로토콜

1. 오늘 남은 한 슬롯을 쓴다면 G3 TCN 단일-factor probe만 제출한다.
2. 총점, 1-NMAE, FiCR를 모두 기록한다.
3. incumbent 대비 공개 macro 차이의 3배가 G3 factor의 per-group 효과다.
4. 총점과 두 구성요소가 모두 양수이면 다음 날 G2 `w=0.10` probe를 검토한다.
5. TCN G3가 음수이면 해당 family를 닫고 G2 확장 후보는 독립적으로 다시
   평가한다.
6. 두 factor가 각각 양수로 확인되기 전에는 합성본을 제출하지 않는다.
7. 0.65 미달을 이유로 같은 후보의 quantile·weight를 공개 점수에 맞춰
   재튜닝하지 않는다.

현재 0.65까지 남은 차이는 `+0.00387491`이다. 이번 TCN의 로컬 전이 가정
증분은 `+0.00097875`이므로 단독 0.65 후보가 아니다. 0.65는 TCN의 공개
양수 확인 이후에도 추가 독립 factor가 필요하다.

## 공개 제출 결과

2026-07-27 23:21:03에 G3 TCN 단일-factor probe를 수동 제출했다.

| submission | score | 1-NMAE | FiCR |
|---:|---:|---:|---:|
| `1503349` | `0.6454576156` | `0.8759443235` | `0.4149709077` |

기준 제출 `1502437` 대비 공개 macro 변화는 다음과 같다.

- score: `-0.0006674758`
- 1-NMAE: `+0.0001600758`
- FiCR: `-0.0014950275`

G1·G2를 그대로 두고 G3만 바꾼 제출이므로, 공개 G3 단독 효과는 macro
변화의 3배다.

- G3 score: `-0.0020024274`
- G3 1-NMAE: `+0.0004802274`
- G3 FiCR: `-0.0044850825`

즉, trajectory TCN은 절대오차를 소폭 줄였지만 FiCR을 크게 악화시켜
총점이 하락했다. 단순 전이 예상점 `0.64710384`와 실제 점수의 오차는
`-0.0016462244`다. 위 제출 프로토콜에 따라 이 family를 닫고, 공개
결과에 맞춘 quantile·weight 재튜닝은 하지 않는다. 공개 최고점은
제출 `1502437`의 `0.6461250914`로 유지한다.

## 재현

```powershell
python -m experiments.issue_trajectory_tcn
python -m experiments.issue_trajectory_tcn_expanding
python -m experiments.issue_trajectory_tcn_expanding `
  --seeds 29,41 `
  --base-raw-cache artifacts_final/lineage/issue_trajectory_tcn_expanding_seed17_20260727.npz `
  --raw-cache artifacts_final/lineage/issue_trajectory_tcn_expanding_ensemble3_20260727.npz `
  --report artifacts_final/diagnostics/issue_trajectory_tcn_expanding_ensemble3_20260727.json
python -m experiments.compose_issue_trajectory_tcn_candidate
python -m experiments.compose_public_positive_factor_expansion
```

관련 테스트:

```powershell
python -m pytest `
  tests/test_issue_trajectory_tcn.py `
  tests/test_issue_trajectory_tcn_expanding.py `
  tests/test_compose_issue_trajectory_tcn_candidate.py `
  tests/test_compose_public_positive_factor_expansion.py -q
```

공식 평가는 Public 40%, Private 60%로 분리되며 1차 평가는 Private
리더보드 상위 30팀이 산출물 검증 대상이다. 상위 10% 목표는 유지하되,
Public 한 점을 곧바로 Private 성능으로 간주하지 않는다.

- 공식 평가:
  https://dacon.io/competitions/official/236727/overview/evaluation
