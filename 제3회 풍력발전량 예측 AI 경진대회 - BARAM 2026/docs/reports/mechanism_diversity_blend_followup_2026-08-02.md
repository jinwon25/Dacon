# 메커니즘 다양성 블렌드 후속 검증

작성일: 2026-08-02

## 결론

현 공개 최고 `1502437`을 유지한다. 이번 검증에서는 새 제출 파일을 만들지
않았다. Q1·Q2만으로 고정한 가장 안정적인 블렌드조차 H2에서 score와
FiCR가 음수로 뒤집혔고, H2까지 본 뒤에야 살아남는 조합은 사후 선택이라
제출 근거로 사용할 수 없기 때문이다.

`1508186`의 G3 단독 probe 실패까지 합치면 다음 두 계열을 종료한다.

1. 외부기상·trajectory로 G3 전체 표면을 교체하는 계열
2. 현 active 주변에서 기존 tabular·spatial·trajectory 예측을 단순 convex
   blend하는 계열

## 모기 궤적 대회에서 옮긴 가설

이전 모기 비행 궤적 대회는 Private 2위 `0.703151`을 기록했다. 당시 같은
Kalman 잔차 계열은 상관 약 `0.99`에서 정체됐고, Neural ODE, Frenet/control
head, 회전물리 CREE처럼 서로 다른 메커니즘을 도입할 때마다 성능이
상승했다. 마지막에는 강한 base에 3-CREE 앙상블을 수동 주입했다.

이번 풍력 문제에도 같은 원리를 적용해 다음 네 메커니즘을 현 active 주변의
작은 변위로 섞었다.

- legacy exact driver: tabular tree ensemble
- legacy stack5: tabular stacking 변형
- LDAPS multiresolution: 공간·물리 NWP
- issue trajectory TCN: 24시간 시계열 trajectory

반복 공개 실패가 확인된 G3는 완전히 고정하고 G1·G2만 변경했다. 전체
304개 sparse 1~3-member 조합과 alpha
`0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20`을 평가했다.

## 핵심 EDA: 예측 변위와 실제 오차는 다르다

TCN의 active 대비 변위는 다른 메커니즘과 거의 직교했다.

- TCN vs exact driver movement correlation: `-0.0809`
- TCN vs LDAPS movement correlation: `-0.0754`

하지만 truth 대비 실제 오차 상관은 여전히 매우 높았다.

- TCN vs exact driver error correlation: `0.9832`
- TCN vs LDAPS error correlation: `0.9866`
- exact driver vs LDAPS error correlation: `0.9981`

따라서 예측을 서로 다르게 움직인다는 사실만으로는 모기 대회에서의
독립적인 오류 상쇄가 재현되지 않는다. 풍력 후보들은 공통 계절·FiCR
오류를 거의 같은 방향으로 가진다.

## 시간 분리 가설 검증

### 1. Q1-only 최고점 규칙

Q1에서 선택된 `0.25 exact + 0.75 TCN`, total alpha `0.20`은 Q1에서
score `+0.0011579`, 1-NMAE `+0.0002150`, FiCR `+0.0021009`였다.
그러나 잠근 Q2에서 score `-0.0009048`, FiCR `-0.0019148`, H2에서
score `-0.0007322`, FiCR `-0.0015465`로 역전됐다.

### 2. Q1 통과 후 Q2 최고점 규칙

Q1 조건을 통과한 후보 중 Q2 최고점을 택하면 `TCN 100%`, total alpha
`0.01`이 선택된다. Q1과 Q2는 각각 score `+0.0001545`,
`+0.0001716`이었지만 H2는 score `-0.0002432`, FiCR
`-0.0004947`로 실패했다.

### 3. Q1·Q2 최악 월 최대화 규칙

H2를 보지 않고 1~6월의 최악 score delta를 최대화하는 규칙도 별도로
고정했다. 선택된 `0.25 exact + 0.75 TCN`, total alpha `0.01`은 Q1
score `+0.0001615`, Q2 `+0.0001132`였으나 H2 score
`-0.0001594`, FiCR `-0.0003239`로 다시 실패했다. 11월 score
`-0.0007688`이 가장 큰 계절 붕괴였다.

### 4. 사후 진단

전체 304개 중 Q1·Q2·H2·full의 모든 component가 음수가 아닌 조합은
3개 있었다. 최선은 `0.5 exact + 0.5 TCN`, total alpha `0.01`이지만
이 사실은 H2까지 함께 사용해 발견했다. 독립 검증이 아니므로 promotion과
제출 파일 생성을 금지했다.

## 방법론 판단

모기 대회의 성공을 그대로 복사할 수 없는 이유는 평가 구조 차이와 오류
독립성 차이다. 모기 대회의 1cm hit 지표는 경계 샘플을 넘기는 직교 예측에
큰 보상을 줬지만, 이 대회는 1-NMAE와 FiCR를 함께 최적화하며 계절별 FiCR
부호 역전이 지배적이다. 같은 NWP와 같은 발전 이력을 공유하는 모델을 더
많이 섞어도 error correlation `0.98~1.00`을 벗어나지 못한다.

다음 실질적 돌파는 새 블렌더가 아니라 독립 오류원을 필요로 한다.

- 설비 가동·제약·정산 또는 출력제한 상태 데이터
- 완전한 이전 연도 NWP와 라벨로 만든 진짜 year-forward 검증
- 현 NWP power-curve와 다른 독립 생성 메커니즘

이 입력이 없는 상태에서 추가 후보 탐색은 2024 기간과 공개 리더보드에
대한 사후 적합 가능성이 더 크다.

## 재현

```powershell
python -m experiments.mechanism_diversity_blend_audit `
  --subset-repetitions 5000 `
  --bootstrap-repetitions 2000 `
  --output-report artifacts_final/diagnostics/mechanism_diversity_blend_audit_20260802.json

python -m pytest -q tests/test_mechanism_diversity_blend_audit.py
```

진단 JSON은 대용량·재생성 가능 산출물 정책에 따라 Git에는 포함하지 않고,
코드와 테스트 및 본 결정 기록만 추적한다.
