# 프로젝트 영구 메모리

이 문서는 투구 제구 성공 확률 예측 프로젝트의 현재 불변 조건이다. 새 작업 세션과
팀원은 실험 전에 이 문서를 먼저 읽고, 기준이 바뀌면 결과 근거와 함께 갱신한다.

## 1. 실행·전달 원칙

- 모든 새 모델은 처음부터 **standalone**으로 설계한다.
- 최종 ZIP 안의 `script.py`, `requirements.txt`, `model/`만으로 실행되어야 한다.
- 과거 `submit_v*.zip`, 저장소 `src/`, 다른 후보 artifact를 실행 시 요구하면 안 된다.
- 현재 단일 릴리스는
  `submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip`에
  보존한다.
- 후보는 현재 champion을 덮어쓰지 않는다. 자동 승격 게이트를 통과한 뒤에만 별도
  standalone 후보를 만들고, Public 결과가 확인된 뒤 champion 포인터를 바꾼다.

## 2. 현재 기준선

- Public: **1181.7100031613**
- 제출 ID: `76835`
- 계보: `v25 → v26 → TrackMan-ASOF → v82 → v84 → v104 → v124 → v142 → v148 → v167 → JY bridge027 → v244 → v290 → v320(미제출) → v335`
- v335: v290의 R_CORE를 보존하고 F direct 20%+low-rank 0.50, R_ANCHOR low-rank
  0.50을 적용한다.
- 전달 ZIP:
  `submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip`
- 전달 ZIP SHA-256:
  `A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A`
- v290 Public `1176.757071668`은 직접 공식 비교 기준이다. v335는 v290 대비
  `+4.9529314933`, 1180 대비 `+1.7100031613`이며 1185까지 `3.2899968387` 남았다.

## 3. DACON 규칙 불변 조건

[공식 독립 예측 공지](https://dacon.io/competitions/official/236743/talkboard/417123)를
항상 우선한다.

- 한 test 행은 그 행의 입력, 공식 train, train으로 미리 동결한 artifact만 사용한다.
- 다른 test 행의 평균·분포·빈도·순위·그룹 통계·rolling/lag/누적값을 사용하지 않는다.
- 파일 순서상 앞선 test 행도 과거 정보로 간주하지 않는다.
- 전체·singleton·shuffle·partition 실행에서 동일 행 확률 차이가 `1e-12` 이하여야 한다.
- test 크기나 분포로 threshold, center, alpha, weight를 다시 정하지 않는다.
- 외부 데이터는 공식 규칙과 운영진 답변으로 명시적으로 허용된 경우가 아니면 쓰지 않는다.
- 주요 feature, 후처리값, blend 비율, 모델 설정의 도출 근거와 선택 시점을 보고서에
  남긴다. Phase 3 검증은 점수 재현뿐 아니라 이 과정의 소명도 포함한다.

## 4. 연구·승격 원칙

- Brier 계열 proper score를 최적화하며 accuracy를 대리 목표로 사용하지 않는다.
- 모델 선택과 calibration은 outer fold 안에서 전체 선택을 반복하는 temporal nested
  cross-fitting으로만 수행한다.
- 2024 label은 이미 반복 연구에 사용된 `development_contaminated` 축이다. 진단에는
  사용하되 독립 holdout 또는 후보 승격 근거로 부르지 않는다.
- season, month, R_CORE, R_ANCHOR, F의 평균뿐 아니라 최악 집단을 함께 본다.
- 같은 모델의 scalar scaling보다 incumbent와 오차 상관이 낮은 직교 모델을 우선한다.
- Public 점수만으로 blend weight나 test 분포를 역산하지 않는다.
- local→Public 단일 환산을 금지한다. clean transfer 3개 미만이면 projection을 만들지
  않고, 3개 이상이어도 시나리오로만 사용한다.
- v82 full-2024 OOF→Public은 한 건의 clean transfer로 기록하지만 다른 family의 환산계수로
  사용하지 않는다.

standalone 후보 생성 최소 조건:

- 서로 다른 `nested_outer` 또는 `locked_shadow` primary 축 최소 2개
- 모든 primary 축 gain `> 0`, 양수 월 비율 `>= 75%`, 최악 월 `> -5`
- 모든 primary 축 R_CORE/R_ANCHOR/F 최소 gain `>= 0`
- pitcher, crossed pitcher×batter, 연속 block bootstrap은 위험 진단으로 기록하되 단독
  기각 기준으로 사용하지 않는다. v335는 crossed p05가 음수였지만 가장 큰 Public 개선을 냈다.
- final family 전체 trial ledger와 White Reality Check `p <= 0.10`
- exact parent parity와 각 축을 열기 전 recipe 동결
- 행 독립성 및 패키지 구조 검사 통과

## 5. 작업 정리 원칙

- 최종 판단은 `reports/`에 수치와 재현 명령으로 남긴다.
- 실패한 대형 cache, 임시 압축 해제본, 중복 ZIP, `__pycache__`, `.pytest_cache`는 삭제한다.
- 과거 코드·보고서는 재현 근거가 있으면 지우지 않고 역사 자료로 표시한다.
- 원본 데이터·개인정보·인증정보는 Git/LFS에 올리지 않는다. 모델·OOF·제출 ZIP은
  PRIVATE 팀 전달용으로 해시가 고정된 v167·과거 JY·최신 v335 릴리스, 기존 1161 private
  LFS allowlist와 `artifacts/oof_champion_1170/` 번들만 허용한다.

## 6. 현재 최우선 연구

1. v335가 보존한 R_CORE에서 기존 성분과 독립적인 정보축 탐색
2. v335 Public 결과로 low-rank dose·anchor team·F weight를 재튜닝하지 않기
3. 다음 목표 1185에 필요한 효과 크기를 전체행 RMS와 paired Brier로 함께 확인
4. 새 candidate family의 선택·tuning·calibration 전체를 outer fold 안에서 반복하는
   nested temporal runner와 trial ledger
5. 팀원별 독립 exact OOF가 확보될 때만 worst-domain constrained blend 재검토

완료·기각 기록:

- v61: 최종 gate exact OOF와 2025 profile 792개 exact 재현 완료. gate는 full-2024
  `+0.0138`, late-2024 `-0.1612`이므로 확대 금지.
- v62: group-balanced ExtraTrees는 full-2024 `-1.1415`, late-2024 `-0.0248`로 기각.
- v63: 순수 `pitcher × balls>strikes` 상호작용은 v50과 거의 무상관이지만 exact
  consensus `0/240`, full-2024 `+0.0860`, late-2024 `+0.0594`로 기각.
- v64: 동결한 물리 교사→ID 제거 학생은 full-2022 `-1.2240`, full-2024
  `-1.3183`, late-2024 `-1.5045`; 2024 세 seed 모두 음수로 기각.
- 팀 브랜치에는 독립 final OOF가 없다. `팀 브랜치`은 직전 1158 champion으로 이미
  병합됐고 `팀 브랜치`은 workflow 삭제뿐이므로 exact OOF 전달 전 blend를 금지한다.
- v44/v64 physical student와 v63 count interaction의 강도·seed·도메인 재탐색을 금지한다.
- 공개 대회 자료와 최신 문헌 감사는
  `reports/target1170_research_update_20260822.md`를 기준으로 한다.

최신 상세 결과는 `reports/target1170_research_update_20260822.md`, 실행 방법은
`docs/STANDALONE_CHAMPION.md`를 따른다.

## 7. TrackMan 전수조사 불변 결론 — 2026-08-22

- 공식 train은 1,475,092행, TrackMan은 1,793,078행이다. target을 읽지 않은 이벤트
  정렬로 1,217,598행(82.5439%)을 일대일 연결했다.
- test에는 현재 투구 물리량, plate location, intended target, TrackMan ID가 없다.
  따라서 현재 투구의 물리량을 직접 feature로 넣거나 다른 test 행에서 추정하지 않는다.
- TrackMan 파생값은 항상 `season < origin`인 공식 pitcher_id별 동결 profile만 허용한다.
  ID crosswalk도 target-free 정렬에서 support와 purity 기준을 사전 고정한다.
- 직접 map은 2025 proxy에서 행 99.05%, 투수 95.14%를 덮지만, 기존 Hungarian gate를
  교체한 v66은 full-2024 `-0.0155`, late-2024 `-0.1527`이라 기각했다.
- 127개 물리 profile을 조사한 뒤 저용량 Ridge(v68/v69), latent pitch type rebase(v71),
  TrackMan-label 학생(v73)을 모두 기각했다. 같은 family의 차원·weight 미세 탐색을 금지한다.
- v72에서 TrackMan 3분류 구종과 ASOF 재구성 label의 일치율은 93.3251%였다. TrackMan
  label 재학습은 분류 정확도를 약 1%p 높였지만 outcome Brier는 세 시간축 모두 악화됐다.
- v70은 과거 평균 IVB 한 개, ALL, 4~9월만 쓰며 late23→full24 `+0.3602`,
  early24→late24 `+2.8048`이었다. 그러나 recipe를 고정한 v74의 early22→late22와
  full22→full23에서 source-only eta가 모두 `0`, gain도 모두 `0`이었다. full23→full24
  확인축도 월·domain gate를 실패했다. **v70 최종 기각, 패키징·Public probe 금지**다.
- 이 조사 당시 1158 champion과 standalone SHA-256은 변함없었다. 이후 모든 후보도 단일 ZIP 실행,
  행 독립성, origin 이전 정보만 사용한다.

다음 우선순위는 (1) 팀원 독립 exact OOF constrained blend, (2) nested temporal runner,
(3) 환경 안정 feature-family residual이다. IVB·고차원 물리·repeatability·TrackMan 구종
학생 계열 재탐색은 금지한다. 상세 근거는
`reports/trackman_deep_dive_20260822.md`를 기준으로 한다.

## 8. 평가체계 v3 불변 결론 — 2026-08-22

- 공식 점수 구현은 `max(0, 100000 * (1 - Brier / (r*(1-r))))`와 일치한다.
- 1170까지 `+11.9254443249`이며 테스트 양성률을 `0.4861`로 가정하면 평균 Brier
  약 `2.9791e-05` 감소가 필요하다.
- `configs/evaluation_v3.json`과 `src/evaluation_contract.py`가 post-1158 후보의
  유일한 승격 계약이다.
- v75에서 full-2024 calibration intercept/slope는 `-0.00488/1.01172`였고 same-axis
  additive/affine gain은 `+0.829/+1.190`뿐이었다. source-only calibration은 월·domain
  gate를 통과하지 못했다. champion 단독 scalar calibration 재탐색을 금지한다.
- v75 same-axis domain×count oracle `+19.941`은 답을 본 진단이다. 이를 source-only로
  고정한 v76은 late23→full24 exact 부모에서 `-45.486`이므로 lookup 재탐색을 금지한다.
- 공식 전체 추론 제한은 600초다. 120초는 전달 여유를 위한 내부 soft guard로 유지한다.
- 이 재감사 당시 1158 champion, standalone 경로와 SHA-256은 변함없었다. v58~v76 중 새 Public probe
  자격을 얻은 후보는 없다.

상세 수치와 다음 실행 설계는 `reports/evaluation_reaudit_20260822.md`를 기준으로 한다.

## 9. v77 팀 OOF·v78 환경 안정 잔차 불변 결론 — 2026-08-22

- 팀 OOF 전달·비교는 `configs/oof_bundle_contract_v1.json`과
  `src/archive/v77_team_oof_constrained_blend.py`만 사용한다. 동일 `(axis,row_id)` 정렬, SHA-256,
  strict-forward 학습 경계, 행 독립성과 원본 데이터 parity가 없는 prediction은 혼합하지 않는다.
- 현재 팀 브랜치와 로컬 artifact에는 champion과 독립적인 exact final OOF가 없다. Public 점수나
  제출 파일만으로 blend weight를 정하지 않는다.
- v78 환경 안정 trajectory/reliability/context residual은 primary early22→late22와
  full22→full23에서 source eta가 모두 `0`이었다.
- full-2023 내부에서 통과한 uniform eta `0.5`는 full-2024 역사 감사축에서 gain
  `-7.5362`, 양수 월 `25%`, 최악 월 `-58.7291`, 최소 domain `-9.8566`으로 반전했다.
- v78은 최종 기각한다. 같은 feature family·환경 weight·eta·threshold 미세탐색, ZIP 생성과
  Public probe를 금지한다.
- 다음 유효 작업은 독립 팀원 exact OOF 확보 → analytic headroom/residual correlation →
  constrained blend → 평가 v3 순서다. 통과 결과만 새 standalone ZIP으로 만든다.

상세 근거는 `reports/target1170_v77_v78_followup_20260822.md`를 기준으로 한다.

## 10. v79~v81 불변 결론 — 2026-08-22

- v79 champion-offset context/composition GLM은 primary 두 축 모두 source eta `0`이라
  기각한다. 같은 interaction·ridge·eta 미세탐색을 금지한다.
- v80은 late-2023 source에서 v26 shift 25%가 `+110.3945`였지만 full-2024에서
  `-0.7267`로 반전했다. early-2024 source는 여섯 후보 모두 weight `0`이었다.
- 기존 후보의 same-axis oracle 최대 gain은 full-2024 `+2.4566`, late-2024 `+0.6418`이고,
  대표 후보의 champion 대비 오차 상관은 `0.999986~0.999997`이다. 기존 v17~v57 후보의
  blend weight 재탐색을 금지한다.
- v81 stable shallow GBDT는 primary 두 축 모두 eta `0`; full23→full24 historical에서
  `-11.5698`로 반전했다. feature threshold, 환경 risk, depth/tree/seed 재탐색을 금지한다.
- v79~v81은 test CSV·test 집계·Public 기반 weight를 사용하지 않았고 ZIP/Public probe를
  만들지 않았다. 대형 rejected prediction cache는 삭제하고 compact 판단만 보존한다.
- 다음 유효 입력은 v77 계약의 독립 팀원 exact OOF, 운영진이 명시적으로 허용한 새로운
  정보원, 또는 새 locked shadow다. 그 전에는 현재 standalone 1158.0746을 유지한다.

상세 근거는 `reports/target1170_v79_v81_followup_20260822.md`를 기준으로 한다.

## 11. v82 탐색적 Public probe — 2026-08-22

- 1170 승격 게이트는 유지하되, 사용자가 현재 Public 상회 가능 후보를 먼저 한 번 확인하는
  2단계 정책을 승인했다. 이 예외는 새 weight 탐색 권한이 아니다.
- v82는 2024 축을 열기 전에 고정된 v57 EXP-021 strict를 최종 챔피언 위 R_CORE에만
  probability `10%` 혼합한다. R_ANCHOR/F는 챔피언과 동일하다.
- 최종 부모 exact OOF gain은 full-2024 `+1.259697`, late-2024 `+0.487466`이지만,
  양수 월 비율 `50%/33.3%`, 최악 월 `-9.1762/-9.0641`로 평가 v3 승격 조건은 실패했다.
- 제출 파일은 `submit_v82_probe.zip`, 현재 전달 ZIP은
  `artifacts/standalone_champion_1159/standalone_champion_1159.zip`, SHA-256은
  `6F6B208EAA7D71BA8F9AC1E60C5CF0E23011F09FD35A639CD66B2912FCDB4AB1`이다.
- ZIP은 과거 ZIP 의존성이 없고 행 shuffle/partition 오차 `0`, 245,789행 규모 추론
  `58.7940초`, 전체 테스트 `276 passed, 4 skipped`를 통과했다.
- 인증 파일 재사용 위험을 고지하고 사용자 명시 승인 뒤 실제 제출했다. API는
  `isSubmitted=true`, `detail=Success`를 반환했다.
- 공식 Public의 사후 확인 정밀값은 **1159.3352239501**, 직전 대비 `+1.2606682750`,
  확인 당시 13위·누적 28회다.
  제출 ID는 API와 공개 페이지에서 제공되지 않았다.
- full-2024 OOF gain과 Public gain 차이는 `+0.0009671037`, 전이 비율은
  `1.000767727`이었다. 한 관측만으로 일반 환산하지 않는다.
- 새 champion은 `artifacts/standalone_champion_1159/standalone_champion_1159.zip`이며
  SHA-256은 그대로 `6F6B...4AB1`이다.
- Public 결과를 보고 strict weight·route를 역조정하지 않는다.

상세 근거는 `reports/target1170_v82_public_probe_20260822.md`를 기준으로 한다.

## 12. v83 새 챔피언 조건부 headroom — 2026-08-22

- v82 exact OOF를 새 부모로 놓고 기존 동결 shift를 additive rebase했다. test CSV·Public
  점수 기반 weight는 사용하지 않았으며 same-axis label을 쓴 진단이라 제출 근거 단독 사용은 금지한다.
- strict 같은 방향을 추가하면 full/late-2024 `-0.0763/-0.7871`이다. 10% 증량 금지다.
- v50은 full `+2.5390`이지만 late `+0.1340`, 양수 월 `33.3%`, 최소 domain
  `-3.4279`라 패키징 금지다.
- v56 F-only shared-horizon FM은 v82 R_CORE와 행이 겹치지 않는다. full/late gain은
  `+1.0181/+2.9166`, 양수 월 `75%/100%`, 최악 월 `-4.7126/+0.3596`, 최소 domain
  `0/0`이다.
- v56은 2022 선택 최악 월 `-6.5616`, consensus 통과 `0/32`, FM family 2024 재사용이라는
  위험이 있다. 따라서 기존 `rank=16/source×domain equal/F/logit .10`만 복원하고 어떤
  hyperparameter도 다시 선택하지 않는다.

해당 구현은 v84에서 완료됐다. 상세 조건부 수치는
`artifacts/v83_post_public_rebase_20260822_01/conditional_headroom.csv`에 있다.

## 13. v84 F shared FM 제출과 승격 — 2026-08-22

- v83에서 고정한 `rank=16/source_domain_equal/F/logit eta=.10/clip=.25`를 변경하지 않았다.
- source는 full-2023 245,525행과 late-2024 76,896행이며, 두 기간·domain 총 위험을 균등화했다.
- 제출 추론은 Torch가 아니라 ZIP 내부 vocabulary와 float32 embedding을 읽는 NumPy 내적이다.
- 제출 파일 `submit_v84_probe.zip`의 SHA-256은
  `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`이다.
- 수식 parity `5.55e-17`, R 보호행 변화 `0`, shuffle/partition `0`, 245,789행
  `68.2176초`, 전체 테스트 `282 passed, 4 skipped`를 통과했다.
- API는 `isSubmitted=true`, `detail=Success`; 제출 이력 ID는 `59988`이다.
- 공식 제출 실행 시간은 `41초`이고 로컬 245,789행 프록시는 `68.2176초`다.
- 공식 Public은 **1161.2020600422**로 v82 정확 점수 `1159.3352239501` 대비
  **+1.8668360921**이다. 확인 당시 13위·누적 29회다.
- canonical 단일 릴리스는
  `artifacts/standalone_champion_1161/standalone_champion_1161.zip`이다.
- 성공한 F 레시피를 Public에 맞춰 eta/rank/seed/clip/center로 재탐색하지 않는다.
- 다음 경로는 F를 보호하면서 팀원 exact OOF constrained stack, R_CORE 동적 계층
  uncertainty, 다중 실패유형 일관 결합 순서다.

상세 근거는 `reports/target1170_v84_public_result_20260822.md`를 기준으로 한다.

## 14. v85~v91 후속 검증 — 2026-08-22

- 공식 champion은 계속 `standalone_champion_1161.zip`, Public `1161.2020600422`,
  SHA-256 `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`이다.
- 팀 원격을 다시 fetch했다. `팀 브랜치`은 workflow 삭제뿐이고 `팀 브랜치`은 현재
  champion의 역사적 TrackMan-ASOF 부모다. 별도 독립 exact OOF는 없다.
- 공개 EXP-038~060의 temporal stack, exact TrackMan, workload, pitchmix, calendar,
  arsenal geometry는 모두 2022~2024 최저 연도 gate를 통과하지 못했다.
- v85 저랭크 source 정책 교체는 식별 가능한 pre-2024 정책축이 하나뿐이고 full-2024 월 양성
  50%라 기각했다.
- v86 R FM 신뢰도 gate는 full/late gain `+2.000/+3.028`이지만 월 양성 `50%/33%`,
  최악 월 `-10.646`이라 기각했다.
- v87 cross-season FM은 `+1.454/+1.894`, 월 양성 `75%/67%`, 최악 월 `-5.761`까지
  안정화했지만 source와 late gate를 통과하지 못했다. v88 coherence는 월 양성이 더 낮아졌다.
- v89 F 추가 dose는 full-2024 `-0.028`; 성공한 v84 eta `.10`을 더 키우지 않는다.
- v90 TrackMan 고압 count route는 late-2024 `-0.153`; TrackMan gate 강도·압력 route
  재탐색을 중단한다.
- v91 8월 이후 strict 증량은 source 두 축에서 강했지만 full/late-2024
  `-0.312/-1.030`으로 반전했다.
- v85~v91은 모두 test CSV·test 집계·Public 기반 recipe 선택 없이 row-local OOF로 평가했다.
  point gate를 통과한 후보가 없어 ZIP 생성과 DACON 제출을 하지 않았다.
- 이후에도 모든 배포 후보는 과거 ZIP 의존성 없는 단일 실행 ZIP으로만 만든다.
- 다음 새 family는 완전 nested temporal runner 안의 다중 실패유형 구조 모델이다. 기존 FM,
  F eta, TrackMan gate, strict month route의 강도·seed·threshold 미세탐색은 금지한다.

상세 근거는 `reports/target1170_v85_v91_followup_20260822.md`를 기준으로 한다.

## 15. 데이터·파이프라인 재감사와 v92~v93 — 2026-08-22

- 공식 champion은 계속 `standalone_champion_1161.zip`, Public `1161.2020600422`,
  SHA-256 `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`다.
- v92 temporal player command EB는 source 평균은 양수였지만 full/late-2024
  `+0.0932/-0.1024`, late 양수 월 `0%`로 기각했다.
- v93 conditional FM benefit gate는 full/late-2024 `+1.4791/+1.8530`이었지만 source
  full22→late23 전이에서 모든 recipe가 음수였고, 2024 월 안정성 `62.5%/66.7%`, 최악 월
  `-5.5383`으로 point gate를 통과하지 못했다.
- v87 5,000회 cluster bootstrap의 개선 비율은 full pitcher `78.84%`, late pitcher
  `72.18%`이며 모든 95% Brier-delta 구간이 0을 포함했다. v87/v93 Public probe를 금지한다.
- 현재 가장 큰 병목은 새 알고리즘 부족보다 full-2020~2024 exact champion analogue를
  반복 생성하는 multi-origin nested runner의 부재다. 이를 P0으로 완성하기 전 같은 FM,
  TrackMan profile, scalar calibration의 강도·seed·threshold 미세탐색을 재개하지 않는다.
- TrackMan은 current-pitch 위치·구종·물리량이나 직접 pitch key가 없으므로 train-only
  pitcher/pitch-type history profile로만 사용한다. 현재 투구 command를 직접 안다고 가정하지 않는다.
- 모든 후속 전달·제출 후보는 처음부터 `script.py + requirements.txt + model/`만으로
  실행되는 standalone ZIP이어야 하며 과거 ZIP 런타임 의존을 금지한다.
- 이번 사이클은 통과 후보가 없어 ZIP 생성과 DACON 제출을 하지 않았다.

상세 근거는 `reports/target1170_v92_v93_pipeline_audit_20260822.md`를 기준으로 한다.

## 16. v94~v96 multi-origin 계약·context 전이 — 2026-08-22

- v94는 common wave0 full-2020~2024와 exact v84 full-2022/late-2023/full-2024를
  fidelity label, raw-index·row-id·target·parent SHA-256과 함께 분리했다. common tier는
  메커니즘 안정성, exact tier는 champion marginal gain에만 사용한다.
- common 부모 평균 잔차는 2020 `-0.02373`, 2021 `+0.03305`, 2022 `+0.01806`,
  2023 `-0.02456`, 2024 `-0.00314`로 부호가 반복 전환했다. global/domain offset의
  단순 연도 운반을 재개하지 않는다.
- v95는 2020→2021·2021→2022만으로 270개 저자유도 context recipe를 선택했다.
  동결 선택 `count/R_CORE+F/alpha5000/eta1`은 exact full22→late23 `+7.6946`이었지만
  late23→full24 `-7.4283`, late24 `-4.4207`로 반전했다. count/context lookup의
  alpha·eta·bin·route 재탐색과 Public probe를 금지한다.
- v96에서 같은 recipe의 common early→late gain은 2020~2024 모두 양수였으나 exact v84는
  2022 `-3.4258`, 2024 `+2.3382`였다. 2025 hidden label로 early-season residual을 fit할 수
  없고 test 행 집계도 금지되므로 이 신호는 diagnostic으로만 보존한다.
- 2026-08-22 공식 코드공유·토크와 공개 GitHub 구현을 재감사했다. recent-ASOF,
  pre-2023 F 제거, CatBoost/LightGBM/MLP, TrackMan 상황 편차는 이미 현재 계보에서 검증됐으며
  새 exact row-aligned 독립 OOF는 발견하지 못했다.
- 이번 사이클은 ZIP 생성·DACON 제출 없이 Public `1161.2020600422` standalone champion을
  유지한다. 다음 최우선은 팀원 독립 exact OOF+2025 prediction bundle의 v77 constrained blend다.
- 모든 후속 후보는 처음부터 과거 ZIP 의존성 없는 standalone이어야 한다.

상세 근거는 `reports/target1170_v94_v96_multiorigin_20260822.md`를 기준으로 한다.

## 17. v142~v148 목표 1170 달성 — 2026-08-23

- v142는 v124 위 R_CORE에 독립 H1 15%와 네 과거 창의 부호가 모두 일치하는 C3 보정을
  결합했다. Public `1169.6277932822`, 제출 ID `1544738`, 당시 10위를 기록했다.
- v148은 v142와 mean-recent C3 고해상도 방향 사이의 15% bridge다. 2024 개발축에서
  v142 대비 full/late gain은 `+0.7192/+1.2433`, 증분 최악 월은 `-1.2404`였다.
- 최종 패키지는 245,789행 `88.890초`, 수식·비활성·shuffle·partition 최대 오차
  `1.11e-16` 이하로 감사를 통과했다.
- API 제출 후 Public **1170.3014697177**, 제출 ID `1544757`, 9위를 확인해 목표를
  `+0.3014697177` 초과 달성했다.
- 최종 전달 파일은 `artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip`,
  SHA-256은 `7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337`이다.
- 목표 달성 후 남은 일일 제출 quota 1회는 v154 최종 검증 후보에 사용했다.

상세 근거는 `reports/target1170_v142_v148_public_result_20260823.md`를 기준으로 한다.

## 18. v149~v154 마지막 1173 목표 연구 — 2026-08-23

- v149 공동 H1/C3 반응면, v150 계절감쇠·투수/타자 레벨 효과, v151 calibration은
  source/forward 위험 때문에 기각했다.
- v152의 5월 v124 성숙도 gate는 2022 source에서 같은 부호를 재현했고 85% bridge에서
  Public 중심 추정 `1173.2897`을 보였으나, 정확 v124 이중 추론은 120초 제한을 넘었다.
- v154는 결과 라벨 없이 `v124 - intermediate`만 distillation했다. 학습에서 제외한 2024년
  5월 44,078행에서 RMSE `0.0019841`, R² `0.73456`, 정확 gate 이득 보존율 `93.7982%`였다.
- v154 Public 보수/중심/상단 추정은 `1172.6022/1173.1802/1173.6802`였다.
- 최종 ZIP SHA-256은 `979904DB258A15DDEC65024359DB152269B103567D59C93F75E57DA3E39A5C42`다.
  API 제출은 성공했고 누적 제출이 35회로 증가했지만, 15:13 KST best-only 리더보드는
  v148의 `1170.3014697177`, ID `1544757`, 9위를 유지했다.
- 따라서 최종 공식 champion과 전달 기준은 v148이다. v154는 비승격 마지막 제출 기록이다.

상세 근거는 `reports/target1173_v149_v154_final_submission_20260823.md`를 기준으로 한다.

## 19. exact deployed-H1 계약과 v167 Public 1172 갱신 — 2026-08-25

- v157은 v148 ZIP의 실제 H1(depth 8, seed 42/43/44)을 strict-forward OOF로 재구축해
  과거 depth-6 단일-seed proxy와의 계약 오차를 확인했다.
- v158은 같은 exact H1 잔차 위에서 고정 C3 창을 다시 계산했다. 추가 C3 교체 자체는
  승격되지 않았지만 이후 H1 보정을 정확한 v148 기준에서 평가할 수 있게 했다.
- v160은 원 H1 패키지의 고정 affine `alpha=1.09`,
  `center=0.5854452601930041`만 복원했다. exact-v148 2024 gain은 `+2.1532287780`이었지만
  late-2023 `-7.4035640365`와 음수 bootstrap p05 때문에 사전 gate는 실패했다.
- 사용자가 개선 가능 후보의 실제 제출을 명시적으로 요청해, 새 저복잡도 후보 중 locked
  gain이 가장 큰 v160을 v167 독립 ZIP으로 패키징했다.
- v167은 v148의 89개 멤버 중 `model/h1/model/rf.pkl` 하나만 원본 H1 바이트로 바꾼다.
  두 번의 독립 재빌드 SHA가 일치했고 formula/inactive/shuffle/partition 오차는 모두
  `1.11e-16` 이하, 245,789행 실행은 `88.4978초`였다.
- 2026-08-25 01:39:32 KST API 제출 후 Public **1172.0772380321**, 제출 ID `1547707`,
  확인 시점 11위를 기록했다. v148 대비 **+1.7757683144**다.
- 새 챔피언 SHA-256은
  `30DD28F56723EC0F560C9101FC5A94EF78568F874DCA88BF808879831E61C8C1`이다.
- 실제 개선은 승격 근거로만 사용한다. 동일 affine을 Public에 맞춰 더 조정하거나 test 정답을
  역추정하지 않는다. 목표 1180까지 `7.9227619679`가 남았다.

상세 근거는 `reports/target1180_v167_public_result_20260825.md`를 기준으로 한다.
