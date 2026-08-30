# 프로젝트 현황

마지막 갱신: `2026-08-31 KST`

최신 공식 결과: [`../reports/v335_public_result_20260831.md`](../reports/v335_public_result_20260831.md).
v335는 제출 ID `76835`, Public **1181.7100031613**으로 v290보다 **+4.9529314933**
개선돼 목표 1180을 달성했다.

## 현재 champion

| 항목 | 값 |
|---|---:|
| champion | `submit_v335_anchor_lowrank_complement.zip` |
| Public | **1181.7100031613** |
| 1180 초과분 | **1.7100031613** |
| 1185까지 | **3.2899968387** |
| 제출 ID / 확인 순위 | `76835` / 제출 이력 기준 확인 |
| 전달 경로 | `submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip` |
| SHA-256 | `A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A` |
| 실행 의존성 | ZIP 내부 `script.py`, `requirements.txt`, `model/`만 사용 |
| 행 독립성 | shuffle/partition 통과, 최대 `5.55e-17` |
| 현재 브랜치 기준 | `team/main`의 이 릴리스 커밋 |

이 파일은 현재 운영 상태의 단일 원본이다. v335는 2026-08-31 00:00:45 KST에 Public
`1181.7100031613`을 확인했다. v290의 R_CORE를 보존하고, F direct+low-rank 포트폴리오와
R_ANCHOR low-rank 보완을 결합한다. 중간 v320을 제출하지 않았으므로 공식 이득을 두 축에
나눠 귀속하지 않는다.

**정정 (사후 확정):** 애초 "리더보드 미갱신이라 개별 점수 미확인"으로 기록했던 것은
관측 시점의 한계였을 뿐이다. 이후 제출 이력 ID `61440`으로 v154의 실제 Public 점수가
**1168.4038526829**로 확인됐다 (v148 대비 `-1.8976170348`, v142 대비 `-1.2239405993`).
v154가 승격되지 않은 진짜 이유는 "리더보드 미갱신"이 아니라 **명확한 하락**이다.
v154와 그 exact 버전인 v152(`configs/v152_may_maturity_bridge_audit.json`,
`bridge_weight_toward_v138≈0.85`)가 공유하는 핵심 가설 — "v142→v138 브릿지 가중치를
0.15에서 0.85로 올리면 Public이 개선된다" — 은 이 실측으로 **기각**됐다. v152는 v154와
같은 축(같은 신호의 93.7982% 보존)을 쓰므로 함께 기각 상태로 취급하며, 새로운 독립
증거 없이 이 가중치·이 축을 재포장하지 않는다. `configs/v154_maturity_delta_submission_package.json`의
`public_estimate`(conservative 1172.6022 / centre 1173.1802 / optimistic 1173.6802)는
실측과 약 4.2점 어긋났고 방향(양수 예상)도 실제(음수 전이)와 반대였다 — 로컬 quadratic
곡률 + 단일 관측 델타(v142→v148, weight 0→0.15)로 weight 0.85까지 외삽하는 calibration
방법론 자체가 이 정도 외삽 거리에서는 신뢰할 수 없다는 뜻이다. 반면 v142(weight 0)에서
v148(weight 0.15)까지는 실측 기울기가 양수였다 — 즉 이 축의 진짜 최적점은 0.15보다
약간 위, 예컨대 0.2~0.35 부근 어딘가에 있을 가능성이 높다는 가설을 다음 세션·다음
제출권을 위한 메모로만 남긴다. 아직 코드로 만들거나 패키징하지 않는다.
과거 문서의 “champion” 표현은 작성 당시 기록으로만 읽는다.

## 계보와 전달 원칙

```text
v22 domain calibration + row-local ASOF
  → v25 post-break R_ANCHOR direct probability
    → v26 frozen signal weight 15%
      → 0819 TrackMan-ASOF gate
        → standalone_champion_1158.zip
          → EXP-021 strict 10% on R_CORE
            → standalone_champion_1159.zip
              → shared pairwise FM 10% on F
                → standalone_champion_1161.zip
                  → source-stability majority-two R_CORE
                    → standalone_champion_1162.zip
                      → v124 Public-quadratic stack
                        → v142 independent H1 + sign-stable C3
                          → submit_v148.zip (Public 1170.3014697177)
                            → submit_v167.zip (Public 1172.0772380321)
                              → submit_jy_runners_high_li_bridge027.zip (Public 1172.1373858439)
                                → submit_v244.zip (Public 1175.9746833121)
                                  → submit_v290.zip (Public 1176.757071668)
                                    → v320 F portfolio (미제출)
                                      → submit_v335.zip (Public 1181.7100031613)
```

계보는 연구 설명용이다. 실행할 때 과거 ZIP을 연쇄적으로 요구하지 않는다. 새 후보도 처음부터
standalone으로 만들며, 평가 v3를 통과하기 전에는 champion 파일을 덮어쓰지 않는다.

## 바로 읽을 문서

1. [`../reports/v335_public_result_20260831.md`](../reports/v335_public_result_20260831.md)
2. [`../reports/target1185_v328_v335_multiexpert_followup_20260830.md`](../reports/target1185_v328_v335_multiexpert_followup_20260830.md)
3. [`../reports/target1180_v320_futures_portfolio_candidate_20260830.md`](../reports/target1180_v320_futures_portfolio_candidate_20260830.md)
4. [`../reports/target1180_v290_evaluation_reaudit_20260830.md`](../reports/target1180_v290_evaluation_reaudit_20260830.md)
5. [`STANDALONE_CHAMPION.md`](STANDALONE_CHAMPION.md)
6. [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)
7. [`../reports/submissions.csv`](../reports/submissions.csv)

## 평가 상태

- 공식 점수 구현은 DACON BSS 식과 일치한다.
- 1180 목표는 v335의 공식 Public `1181.7100031613`으로 달성했다.
- 다음 목표 1185까지 공식 점수 `3.2899968387`이 남아 있다.
- 2024 label은 반복 연구에 사용돼 `development_contaminated`다. 독립 holdout으로 부르지 않는다.
- 새 후보는 `configs/evaluation_v3.json`과 `src/evaluation_contract.py`를 통과해야 한다.
- primary evidence는 서로 다른 `nested_outer` 또는 `locked_shadow` 축 최소 2개다.
- 전체 family trial ledger, White Reality Check, pitcher/crossed/block bootstrap을 요구한다.
- local→Public 단일 환산은 금지하며 clean transfer 3개 전에는 projection을 표시하지 않는다.
- 공식 추론 제한은 600초이고 저장소의 120초는 내부 soft guard다.

v84는 고정 v56 레시피의 조건부 full/late gain이 양수라 사용자 승인 탐색 정책으로 한 번
제출했고 Public `1161.2020600422`를 기록해 새 champion으로 승격했다. 일반 후보의 평가 v3
기준은 낮추지 않는다.

## 최신 실험 결정

- v61: 최종 TrackMan gate exact OOF 재현. full-2024 `+0.0138`, late-2024
  `-0.1612`; 확대 금지.
- v62~v64: ExtraTrees, count interaction, physical-teacher student 모두 기각.
- v65~v74: TrackMan 1,793,078행 전수조사. 현재 투구 물리·위치·의도 부재와 시간 전이
  실패로 IVB·고차원 profile·구종 학생 계열 기각.
- v75: full-2024 전역 calibration은 이미 양호. source-only additive/affine 보정은 게이트 실패.
- v76: domain×count source-only contrast가 exact late23→full24에서 `-45.486`; 기각.
- v77: 팀 exact OOF bundle 계약과 worst-domain constrained blend evaluator 완성. 독립 팀원
  OOF가 없어 weight 계산은 보류한다.
- v78: 환경 안정 trajectory residual은 primary 두 축 모두 source eta `0`; 2023에서 선택된
  eta `0.5`도 full-2024 `-7.5362`로 반전해 기각했다.
- v79: champion 고정 offset context/composition GLM은 primary 두 축 모두 eta `0`; 기각.
- v80: 여섯 exact OOF 방향의 forward constrained stack은 late23→full24 `-0.7267`,
  early24→late24 eta `0`. 기존 후보 same-axis oracle도 최대 `+2.4566`이고 대표 오차 상관이
  `0.999986~0.999997`이라 기존 계열 재혼합을 종료한다.
- v81: 안정 피처 shallow GBDT는 primary 두 축 모두 eta `0`; full23→full24 `-11.5698`로
  반전해 기각한다.
- v82: 2024 전에 고정한 EXP-021 strict 10% R_CORE 혼합을 최종 챔피언 위에서 재계산했다.
  full/late-2024 gain은 `+1.2597/+0.4875`, 최악 월은 `-9.1762/-9.0641`이다. 평가 v3
  승격 대상은 아니지만 단일 ZIP·행 독립성·245,789행 58.79초 감사를 통과했다. 실제
  Public gain은 `+1.2606643`으로 full-2024 OOF와 `+0.0009671` 차이였다.
- v83: 새 v82 OOF 부모 위 조건부 headroom을 재계산했다. strict 추가 증량은
  full/late `-0.0763/-0.7871`, v50은 late·domain gate 실패다. F-only v56은
  full/late `+1.0181/+2.9166`이라 다음 고정 recipe 복원 1순위다.
- v84: 고정 v56을 full-2023+late-2024 source로 재학습하고 NumPy-only FM으로 내보냈다.
  standalone 245,789행 `68.218초`, 수식 오차 `5.55e-17`, shuffle/partition `0`을 통과했다.
  Public `1161.2020600422`, v82 대비 `+1.8668360921`로 새 champion이다.
- v94: common wave0 full-2020~2024와 exact-v84 full22/late23/full24를 fidelity label,
  raw-index·target·parent hash와 함께 분리했다. common evidence를 exact champion gain으로
  오해하지 않는 multi-origin 계약을 완성했다.
- v95: 2020→2021·2021→2022만으로 선택한 `count/R_CORE+F/alpha5000/eta1`은 exact
  full22→late23 `+7.6946`이었지만 late23→full24 `-7.4283`, late24 `-4.4207`로 반전해 기각했다.
- v96: 같은 recipe의 common early→late gain은 2020~2024 모두 양수였으나 exact v84는
  2022 `-3.4258`, 2024 `+2.3382`로 부호가 달랐다. 2025 hidden label로 같은 시즌 residual을
  fit할 수 없으므로 diagnostic으로만 보존하고 패키징하지 않는다.
- v110: 동결된 v105 LGBM·MLP 합의 보정을 Public-1162 v104 exact parent 위에 재기준화했다.
  source alpha는 `0.87235`였지만 full/late-2024 gain이 `-0.6714/-3.7676`으로 반전해 기각했다.
- v111: 같은 보정을 v104 `majority_two` mask 비활성 행 5.96%~7.60%에만 적용했다.
  source alpha는 상한 `1.0`이었으나 late-2023 `-0.0822`, full/late-2024
  `+0.0477/+0.2543`에 그쳤고 모든 point·robust gate가 실패했다. v105 correction의
  전체/비활성 route와 강도 미세탐색을 종료한다.
- v116: 2022↔late-2023 교차 ridge로 동결한 세 가지 역사적 실패유형 prior가 exact v104 OOF에서
  full/late-2024 `+5.4305/+6.2139`였지만 Public `1161.5978`, v104 대비 `-1.0325`로
  반전했다. v104를 유지하고 v113~v116 계열을 Public 재튜닝 없이 종료한다.
- v118: 2020 Futures ABS 도입을 Regular 대조군과 비교한 정책 DID는 2021·2022 source에서
  `+12.2145/+5.0297`이었지만 동결 late-2024에서 `-8.0050`으로 반전했다. ABS·시즌
  regime 보정 계열을 종료한다.
- v119: 독립 구현한 투수별 희소 ridge 랜덤 기울기는 early22→late22 `+33.1162`였으나
  full22→late23 `-19.2524`, full-2024 `-6.6432`로 역전했다. direct-spline 미사용 경로도
  late-2024에서 음수였으므로 함께 종료한다. 상세 결과는
  [`../reports/target1170_v118_v119_post_v116_20260823.md`](../reports/target1170_v118_v119_post_v116_20260823.md)에 보존한다.
- v120: recent-game 행별 중심 spread는 source `+15.6201/+16.7467`이었지만 full-2024
  `-2.0130`으로 반전해 기각했다.
- v121: target-free 투수×카운트·타자 손 조건부 TrackMan arsenal ridge는 early22→late22
  `+60.0134`였으나 full22→late23 `-18.0876`으로 source에서 무너졌다. full/late-2024의
  `+3.6321/+2.4958`은 월·최악월 gate와 1170 증분에 미달한다. 상세 결과는
  [`../reports/target1170_v120_v121_post_v116_20260823.md`](../reports/target1170_v120_v121_post_v116_20260823.md)에 보존한다.
- 1180 목표 신규 가설 스크리닝(2026-08-23): `base_state` 세분화, win_expectancy 비대칭
  (`win_expectancy_gap`/`ctx_win_gap`), `asof_batter_*` 주신호화, `game_dayofweek` 패턴,
  구종 다양성/엔트로피 5개를 검토했다. 챔피언 계열 `feature_spec.json`을 직접 확인한 결과
  데이터 설명서의 모든 원본 컬럼과 주요 교차 카테고리(`count_state`/`platoon`/
  `pitcher_count`/`batter_count`/`team_matchup`/`situation_state`)가 이미 base feature로
  포함돼 있었고, 5개 가설 모두 기존 코드·리포트와 동일하거나(v78/v79의 `win_expectancy_gap`/
  `ctx_win_gap`은 변수명까지 일치, 양쪽 축 eta 0) 이미 반대 방향으로 판정(v107 타자 ASOF
  제거가 오히려 이득)됐다. 심화 검증 단계에 도달한 가설이 없어 1180 상당 후보를 확보하지
  못했다. 현재 3개 데이터 파일 기준으로는 추가 신규 가설이 사실상 소진됐다고 판단한다.
  상세 근거는 [`../reports/target1180_negative_screen_20260823.md`](../reports/target1180_negative_screen_20260823.md)에 있다.
- v152/v154 (사후 확정): v142→v138 브릿지 가중치를 0.15(v148)에서 0.85로 올리고 5월만
  v124로 되돌리는 May maturity 가설을 실측했다. 제출 이력 ID `61440`, Public
  `1168.4038526829`로 v148 대비 `-1.8976170348`, v142 대비 `-1.2239405993`이며 사전
  추정치(conservative `1172.6022`)와 방향·크기 모두 반전했다. exact 버전 v152(같은
  weight 0.85)도 같은 신호를 쓰므로 함께 기각한다. 로컬 quadratic 곡률 + 단일 실측
  델타(v142→v148, weight 0→0.15)로 weight 0.85까지 외삽하는 calibration 방법론은 이
  거리에서 신뢰할 수 없다고 결론짓는다. v142→v148 구간의 실측 기울기는 양수였으므로,
  이 축의 진짜 최적점이 0.15 근방보다 약간 위(예: 0.2~0.35)에 있을 가능성은 다음
  제출권을 위한 가설 메모로만 남기고 지금 패키징하지 않는다. 상세 근거는
  [`../reports/target1173_v149_v154_final_submission_20260823.md`](../reports/target1173_v149_v154_final_submission_20260823.md)의
  "실측 결과 갱신" 절에 있다.
- 로컬→Public 전이 메타 감사: 실제 Public 점수를 받은 제출 약 24건 전체를 표로 재구성했다.
  핵심 발견은 (a) 후보 자신의 weight/route를 고르는 데 쓰인 것과 같은 축의 큰 로컬
  숫자(v21 `+10.77`→실전이 `+0.04`, v25 selection `+31.49`→전이율 8.9%, v154
  `+11.4~17.6`→`-1.90`)는 체계적으로 과대평가되거나 반전됐고, 완전히 독립 구현된 모델의
  온건한 gain(v82, v142)은 44~100%로 신뢰성 있게 전이했다는 것, (b) v104/v116은 둘 다
  4개 독립 시간축 전부 양수였지만 v116만 사전에 "pitcher/crossed robust bootstrap p05
  negative" 경고가 있었고 그대로 반전했다 — 즉 축 개수보다 cluster bootstrap 하한이
  실제 반전을 더 정확히 예측했다. 이 감사를 근거로 GroupDRO 재검토를 시도했으나, 8/22
  P1 계획("ERM/environment-balanced/worst-group penalty 비교")이 이미 v78로 실행돼
  세 변형 모두 source eta 0으로 종료된 상태임을 확인해 재개하지 않았다. domain-profile
  LGB(8월 초 incumbent 기준, fold별 best_iteration 7~52로 불안정)와 TabM-mini(v45,
  full-2024 `-8.271`, OOF 미보존) 재검토도 재기준화·재학습 비용이 이번 라운드 예산을
  초과해 시도하지 않고 기각 기록만 남겼다. 상세 근거는
  [`../reports/target1180_transfer_meta_audit_20260823.md`](../reports/target1180_transfer_meta_audit_20260823.md)에
  있다.
- v148-flat: 챔피언 ZIP의 5세대 중첩(최대 경로 깊이 6, `main()` 6개, 동적
  `spec_from_file_location` 4곳, `requirements.txt` 3벌)을 단일 세대 평탄 구조
  (깊이 3, `main()` 1개, 동적 로딩 0, `requirements.txt` 1벌)로 리팩터링했다.
  모델 가중치 78개는 바이트 그대로 복사해 SHA-256 1:1 대조에서 불일치 0건이고,
  예측 수식·상수도 그대로다. 117,434행 동등성 감사에서 최대 절대차는 `3.33e-16`이며
  이는 **원본을 두 번 실행했을 때의 차이와 같은 값**이라 평탄화가 기여한 수치 차이는
  0이다. 세 도메인·데뷔 cold-start 2,000행·2019~2024 표본·3~10월을 모두 덮었고,
  shuffle/partition은 `3.33e-16`, 245,789행 런타임은 같은 세션 연속 측정에서
  원본 97.057초 대비 95.889초(0.988배)다. **공식 전달본은 계속 원본
  `submit_v148.zip`이며 평탄화본은 새 champion이 아니다** — 1170.3014697177은
  원본으로 획득한 점수이고 평탄화본으로 재제출해 확인한 바 없다. 부수 발견으로
  같은 이름의 `parent_script.py` 두 개가 실제로는 v84/v82 서로 다른 계층이라는 점,
  중첩 `requirements.txt`가 루트와 버전 충돌(scikit-learn 1.7.2 대 1.6.1 등)해
  합집합을 만들면 검증된 실행 환경이 바뀐다는 점을 기록했다. 상세는
  [`../reports/v148_flat_refactor_20260823.md`](../reports/v148_flat_refactor_20260823.md)에 있다.

실패 cache는 삭제하고 코드·테스트·compact 보고서만 보존한다.

2026-08-23 저장소 구조를 버전별 단독 실행 기준으로 정리했다. 일회성 실험 스크립트
안에 private 심볼로 파묻혀 있던 공용 인프라를 `src/core/`의 9개 모듈로 추출했고,
버전간 import 엣지가 `394`에서 `182`로 줄었다. 이어서 171개 버전 모듈을 제출 이력이
있는 버전에서 도달 가능한 `src/champion/` 37개와 종료된 실험 `src/archive/` 134개로
분리했다. `src/champion`은 `src/archive`를 import하지 않고 `src/core`도 양쪽을
import하지 않는다. hoo H1/C3 사슬, 공유 pairwise FM, post-break GAM·semantic screen
10개는 죽은 screen처럼 보였지만 실제로는 챔피언 의존성이라 아카이브하지 않고 승격했다.
함수 본문은 줄 단위로 그대로 옮겼고 전체 테스트는 `378 passed, 4 skipped`로 동일하다.
각 세대의 단독 실행 명령줄은 `src/champion/README.md`에 있다. 공식 전달본은 계속
`submit_v148.zip`이며 SHA-256도 변하지 않았다.

## 종료 상태와 후속 연구 조건

2026-08-23의 마지막 일일 제출권은 v154에 사용했고 실측 Public `1168.4038526829`로
기각됐다(v148 대비 `-1.8976170348`). 현재 세션의 제출 연구는 종료했다. 공식 전달
기준은 v148로 동결한다. 대회가 다시 열리거나 독립 OOF가 추가될 때만 다음 순서로
연구를 재개한다.

**다음 세션 가설 메모 (패키징 금지, 기록만):** v142(weight 0, Public `1169.6277932822`)
→ v148(weight 0.15, Public `1170.3014697177`)까지는 v142→v138 브릿지 가중치에 대해
실측 기울기가 양수였지만, weight 0.85(v152/v154)에서는 `-1.8976170348`로 크게
반전했다. 두 실측점(0, 0.15)만으로는 곡률을 특정할 수 없으므로, 이 축의 진짜 최적
가중치가 0.15보다 약간 위(예: 0.2~0.35 부근)에 있을 가능성과, 0.15 자체가 이미 국소
최적점 근처였을 가능성이 공존한다. 다음 제출권이 열리면 이 구간 안에서 단일 신중한
탐침(예: weight≈0.25)으로 세 번째 실측점을 확보하는 것이 무작정 0.85로 재도전하는
것보다 정보 가치가 크다. 지금 코드나 패키지를 만들지 않는다.

1. 팀원 모델의 동일 row exact temporal OOF와 동일 recipe test prediction을 확보한다.
2. analytic headroom과 residual correlation을 먼저 확인한 뒤 v77 constrained blend를 실행한다.
3. 평가 v3의 source 월·domain 비악화, bootstrap과 Reality Check를 통과할 때만 패키징한다.
4. 완전히 다른 합법적 정보원이나 새 locked shadow가 없으면 기존 ASOF·상황·TrackMan
   계열의 강도·규제·route 미세탐색을 재개하지 않는다.

TrackMan profile 강도, scalar center/spread, domain×count lookup, v78 환경 안정 Ridge,
TabM 크기·seed, v79/v81 강도·규제·tree 미세탐색은 종료 상태로 유지한다.

## Git·artifact 원칙

- 개인 feature branch에서 작업하고 팀 리뷰 후 squash merge한다.
- 원본 데이터·인증정보는 Git/LFS에 넣지 않는다. PRIVATE 팀 전달용 확정 v167 ZIP과 기존
  1161 LFS·1170 OOF 번들만 해시 고정 예외로 추적하며, v148·v154와 다른 대용량
  ZIP·모델·OOF는 승인된 비공개 팀 채널에서 SHA-256과 함께 관리한다.
- 팀원 전달 artifact는 공식 팀원만 접근 가능한 공간에서 SHA-256과 함께 관리한다.
- Public 제출은 팀 리뷰와 standalone 검증을 통과한 한 파일만 지정 담당자가 수행한다.
