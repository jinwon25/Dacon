# v18 강건성·선택 편향 점검

v17과 관련된 보정과 F 경로 강도 차이를 기록합니다. 개발 구간을 반복 사용한 상태이며 실배포 점검과 독립 평가를 구분합니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Multi-season empirical-Bayes robust evaluation

- Selected recipe: `multi_pitcher_batter_hand_pressure_d1_a3200_w1.5`
- Correction group: `pitcher × batter × hand × pressure`; R_CORE fit/apply only.
- Historical decay / posterior alpha / correction weight: **1 / 3200 / 1.5**.
- Parent F-trend alpha: **0.35**.

| comparison | year | gain | month + | worst month | min p05 | min P(+) | leave-team min |
|---|---:|---:|---:|---:|---:|---:|---:|
| incremental_vs_v15 | 2023 | +13.8804 | 71.4% | -10.5998 | +1.1717 | 96.5% | +10.5609 |
| incremental_vs_v15 | 2024 | +12.6248 | 62.5% | -11.0073 | -2.3828 | 91.2% | +7.5817 |
| total_vs_v13 | 2023 | +57.5586 | 100.0% | +5.5541 | +30.6011 | 100.0% | +19.3911 |
| total_vs_v13 | 2024 | +18.4484 | 75.0% | -72.5056 | +1.7356 | 96.6% | +14.9592 |

- Final-shortlist (top five) Reality Check p-value: **0.0133**.
- Recipes screened before robust confirmation: **852**.
- Reality Check covers only the saved top-five multi-season shortlist; the larger search remains a source of selection bias.
- 2024 is development-contaminated, so this supports a deployment probe rather than independent confirmation.
