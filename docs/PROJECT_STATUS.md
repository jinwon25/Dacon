# 프로젝트 현황

마지막 갱신: `2026-08-23 KST`

최신 최종 연구: [`../reports/target1173_v149_v154_final_submission_20260823.md`](../reports/target1173_v149_v154_final_submission_20260823.md). v154 마지막 제출 후에도 best-only 리더보드 기준 v148이 **1170.3014697177**, 공개 **9위** 챔피언이다.

## 현재 champion

| 항목 | 값 |
|---|---:|
| champion | `submit_v148.zip` |
| Public | **1170.3014697177** |
| 1170 초과분 | **0.3014697177** |
| 전달 경로 | `artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip` |
| SHA-256 | `7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337` |
| 실행 의존성 | ZIP 내부 `script.py`, `requirements.txt`, `model/`만 사용 |
| 행 독립성 | shuffle/partition 통과, 최대 `0.0` |
| 현재 브랜치 기준 | `team/main`의 이 릴리스 커밋 |

이 파일은 현재 운영 상태의 단일 원본이다. 직전 v142 이하 ZIP은 역사적 부모이며 현재
전달 파일이 아니다. v148 제출 API는 성공했고 공개 리더보드에서 9위, 누적 제출 34회를
확인했다. 제출 이력 ID는 `1544757`이다. v154 API 제출 후 누적 제출은 35회로 증가했지만
best-only 리더보드는 갱신되지 않아 v154를 챔피언으로 승격하지 않았다.
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
```

계보는 연구 설명용이다. 실행할 때 과거 ZIP을 연쇄적으로 요구하지 않는다. 새 후보도 처음부터
standalone으로 만들며, 평가 v3를 통과하기 전에는 champion 파일을 덮어쓰지 않는다.

## 바로 읽을 문서

1. [`../reports/target1170_v142_v148_public_result_20260823.md`](../reports/target1170_v142_v148_public_result_20260823.md)
2. [`../reports/target1170_v83_post_public_plan_20260822.md`](../reports/target1170_v83_post_public_plan_20260822.md)
3. [`../reports/evaluation_reaudit_20260822.md`](../reports/evaluation_reaudit_20260822.md)
4. [`PROJECT_MEMORY.md`](PROJECT_MEMORY.md)
5. [`STANDALONE_CHAMPION.md`](STANDALONE_CHAMPION.md)
6. [`../reports/trackman_deep_dive_20260822.md`](../reports/trackman_deep_dive_20260822.md)
7. [`../reports/target1170_v79_v81_followup_20260822.md`](../reports/target1170_v79_v81_followup_20260822.md)
8. [`../reports/target1170_research_update_20260822.md`](../reports/target1170_research_update_20260822.md)
9. [`../reports/submissions.csv`](../reports/submissions.csv)

## 평가 상태

- 공식 점수 구현은 DACON BSS 식과 일치한다.
- 1170은 테스트 양성률을 `0.4861`로 놓으면 평균 Brier 약 `2.1978e-05` 감소가 필요하다.
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

실패 cache는 삭제하고 코드·테스트·compact 보고서만 보존한다.

## 다음 연구 순서

1. 팀원 모델의 동일 row exact temporal OOF와 동일 recipe의 2025 test prediction을 받아
   constrained blend headroom을 계산한다.
2. 독립 OOF가 도착하면 analytic headroom과 residual correlation을 먼저 확인하고 v77의
   source 월·domain 비악화 constrained blend를 실행한다.
3. 비용이 허용되면 v94 common tier와 별도로 각 origin의 전체 champion ladder를 다시
   학습해 완전 exact multi-origin nested runner를 만든다.
4. 완전히 다른 합법적 정보원 또는 새 locked shadow가 생기기 전에는 같은 ASOF·상황·
   TrackMan 계열의 미세탐색을 재개하지 않는다.
5. blend가 평가 v3 선행 gate를 통과할 때만 bootstrap·Reality Check를 실행한다.
6. 새 직교 기반모형을 확보한 뒤에만 cross-fitted beta/logit calibration을 검토한다.

TrackMan profile 강도, scalar center/spread, domain×count lookup, v78 환경 안정 Ridge,
TabM 크기·seed, v79/v81 강도·규제·tree 미세탐색은 재개하지 않는다. 최신 상세 근거는
[`../reports/target1170_v79_v81_followup_20260822.md`](../reports/target1170_v79_v81_followup_20260822.md)에 있다.

## Git·artifact 원칙

- 개인 feature branch에서 작업하고 팀 리뷰 후 squash merge한다.
- 원본 데이터·인증정보는 Git/LFS에 넣지 않는다. 현재 1161 모델 ZIP과 선별 OOF
  evidence만 private Git LFS allowlist로 관리한다.
- 팀원 전달 artifact는 공식 팀원만 접근 가능한 공간에서 SHA-256과 함께 관리한다.
- Public 제출은 팀 리뷰와 standalone 검증을 통과한 한 파일만 지정 담당자가 수행한다.
