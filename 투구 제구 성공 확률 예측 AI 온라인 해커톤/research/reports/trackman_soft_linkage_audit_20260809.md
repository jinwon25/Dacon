# 프로필 연결의 불확실성 감사

연결 거리·순위·안정성이 선수 신원 확정과 같지 않음을 설명합니다. 손잡이 매핑과 누적 구종 수의 해석이 불명확한 경우 새 보정을 승격하지 않았습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Trackman soft-linkage audit

The existing linkage report provides target-free distance, assignment rank, margin, common-origin stability and confidence. The two hand mappings (1→Left/2→Right and its reverse) are not identifiable from the aggregate report alone; the full annual-fingerprint rerun is therefore marked ambiguous and no new correction is promoted. A reusable 200-repeat Dirichlet-perturbed Hungarian routine was executed as a smoke test on the cached candidate cost submatrix.

`asof_pitcher_pitchmix_n` must be audited for cumulative versus season-reset semantics before clipping negative deltas. Until that audit is run on the origin-specific annual table, medium/low/unmatched rows remain bitwise v2.

```text
         origin                                mapping confidence  n_rows  mean_distance  mean_margin  mean_rank                                                            status  rho_mean  rho_p05  entropy_mean  repeats
           2020 1->Left_2->Right (legacy hard mapping)       high     137       0.034734     0.035548   1.094891 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2020 1->Left_2->Right (legacy hard mapping)        low      25       0.151502     0.031367   1.400000 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2020 1->Left_2->Right (legacy hard mapping)     medium      68       0.081570     0.027051   1.455882 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2020 1->Left_2->Right (legacy hard mapping)  unmatched     125       0.155136     0.025255   6.912000 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2021 1->Left_2->Right (legacy hard mapping)       high     125       0.036587     0.056469   1.024000 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2021 1->Left_2->Right (legacy hard mapping)        low      77       0.147988     0.035578   1.480519 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2021 1->Left_2->Right (legacy hard mapping)     medium     138       0.084633     0.029524   1.297101 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2021 1->Left_2->Right (legacy hard mapping)  unmatched     125       0.184767     0.025691   5.432000 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2022 1->Left_2->Right (legacy hard mapping)       high     126       0.034680     0.068803   1.031746 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2022 1->Left_2->Right (legacy hard mapping)        low     101       0.153835     0.034907   1.366337 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2022 1->Left_2->Right (legacy hard mapping)     medium     170       0.088119     0.036968   1.170588 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2022 1->Left_2->Right (legacy hard mapping)  unmatched     163       0.204840     0.023479   5.736196 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2023 1->Left_2->Right (legacy hard mapping)       high     113       0.034357     0.085271   1.017699 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2023 1->Left_2->Right (legacy hard mapping)        low     142       0.153551     0.034267   1.380282 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2023 1->Left_2->Right (legacy hard mapping)     medium     159       0.088056     0.046402   1.132075 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2023 1->Left_2->Right (legacy hard mapping)  unmatched     234       0.231361     0.031399   6.012821 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2024 1->Left_2->Right (legacy hard mapping)       high     115       0.034018     0.090937   1.008696 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2024 1->Left_2->Right (legacy hard mapping)        low     173       0.157320     0.035570   1.439306 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2024 1->Left_2->Right (legacy hard mapping)     medium     158       0.087977     0.053558   1.088608 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2024 1->Left_2->Right (legacy hard mapping)  unmatched     265       0.237008     0.027780   6.275472 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2025 1->Left_2->Right (legacy hard mapping)       high     127       0.034837     0.095285   1.000000 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2025 1->Left_2->Right (legacy hard mapping)        low     189       0.156492     0.031941   1.375661 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2025 1->Left_2->Right (legacy hard mapping)     medium     163       0.089915     0.054878   1.104294 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
           2025 1->Left_2->Right (legacy hard mapping)  unmatched     313       0.236823     0.026173   6.166134 diagnostic_only; mapping alternatives require annual fingerprints       NaN      NaN           NaN      NaN
bootstrap_smoke             cost-matrix-soft-Hungarian        all      64       0.012139          NaN        NaN                     bootstrap utility smoke test; no outcome used  0.227031  0.13075      3.523331    200.0
```
