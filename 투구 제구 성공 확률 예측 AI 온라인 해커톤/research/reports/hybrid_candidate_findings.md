# 혼합 후보의 검증 결과

당시 고정 후보의 연도별 비교와 재추출 통과 조건을 기록합니다. 후보 선택에 사용한 정보가 있어, 통과 표시만으로 완전히 독립된 확증 결과라고 읽지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Hybrid recency/Trackman candidate

- Frozen recipe: incumbent with half-life-1 RF substituted only for `game_type=R`, then 5% rolling-damped Trackman LGB.
- Trackman linkage and profiles use only seasons before each forecast origin.
- No validation target enters linkage, feature construction, drift forecasts, or row-level application.

| year | incumbent | candidate | delta |
| ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.246865966 | -0.000079080 |
| 2022 | 0.244169346 | 0.244151170 | -0.000018176 |
| 2023 | 0.251137482 | 0.251047904 | -0.000089578 |
| 2024 | 0.248460961 | 0.248440533 | -0.000020428 |

- Recency-weighted delta: **-0.000046588**
- Latest-year delta: **-0.000020428**
- Worst-fold delta: **-0.000018176**
- Combined pitcher-season bootstrap P(improve): **1.0000**
- Fixed gate: **PASS**
- Selection caveat: all four outer years have already been reused extensively as development data; Public submission is a deployment probe, not independent proof.
