# Codex 인수인계 — 목표 Public 1190+ (v335 챔피언 이후)

너는 이 프로젝트의 공동 시니어 야구 데이터 분석가이자 ML 리서치 엔지니어다.
목표는 DACON「투구 제구 성공 확률 예측 AI 온라인 해커톤」에서 **공식 Public 1190 초과**다.
보고서나 로컬 추정이 아니라 **실제 제출 결과로 1190+ 가 확인될 때까지** 연구·구현·감사를 계속하라.

작업 루트: `E:\학업\교외 활동\대외 활동\공모전\데이콘\.git-worktrees\pitch-control-target-1180`

---

## 0. 이 프롬프트의 가장 중요한 지시

**지금까지 한 것에 매몰되지 마라.** 최근 사이클은 전부 "v290/v320/v335 위에 작은 보정을
하나 더 얹기"였다. 그 방식의 단일 후보 상한은 관측상 Public **+1~5** 이고, 1190 까지 남은
격차는 **8.29** 다. 같은 계열을 미세 조정해서는 도달하지 못한다.

따라서 다음 세 가지를 **동시에** 굴려라.

1. **재고 회수** — §4 의 40건 reject-pile. 게이트 오보정(§3)으로 잘못 버려진 후보들.
2. **메커니즘 확장** — §5. v335 가 검증한 "frozen 성분을 미적용 route 로 이식"을 체계화.
3. **완전히 새로운 축** — §6. 위 둘로 8.29 가 안 나오면 결국 이것이 필요하다.

한 방향이 막히면 즉시 다른 방향으로 옮겨라. 하나에 사이클을 다 쓰지 마라.
**국소 최적화 금지. 넓게 탐색하라.**

---

## 1. 권위 있는 현재 상태

| 항목 | 값 |
|---|---|
| **챔피언** | submission `76835` / `submit_v335.zip` |
| **Public** | **1181.7100031613** (2026-08-31 00:00:45 KST, 111초) |
| SHA-256 | `A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A` |
| 패키지 | `artifacts/v335_anchor_lowrank_package_20260830_01/submit_v335_anchor_lowrank_complement.zip` |
| **1190 까지 격차** | **8.2900** |
| 1185 까지 격차 | 3.2900 |

제출 예산 **일 1회, 빠듯함**. 리더보드 **best-only** (실패해도 챔피언 순위 보존).

공식 지표: `max(0, 100000*(1 - mean((p-y)^2)/(r*(1-r))))`, `r = mean(y_test)`.
Public = 평가 test 100%, 종료 시 같은 값이 Private 로 확정. 추론 제한 600초.

### 1-1. v335 의 구성

```
v290 (Public 1176.757)
 └ v320 : F 경로만 변경 — direct expert 10%→20%, frozen v50 lowrank_s300_r2 0.50 추가
    └ v335 : R_ANCHOR(정규시즌 & pitcher_team_id 또는 batter_team_id == 13)에
             frozen v50 lowrank_s300_r2 0.50 추가.  F·R_CORE 는 v320 과 비트 동일.
```

50,000행 parity 로 확정: `v320 ⊂ v335 ⊂ v334`.
**v320 은 이제 챔피언의 진부분집합이므로 제출 가치가 없다.**

### 1-2. 확정 제출 n=13 전이율

로컬 full-2024 이득 → 실제 Public 이득의 비율:

```
-1.475 -0.837 -0.200 -0.190 +0.173 +0.331 +0.444
+0.768 +0.825 +0.894 +0.937 +1.001 +1.834
중앙 0.444 · 평균 0.347 · sd 0.86 · 누적 0.452 · 양수 9/13
```

**v335 는 0.894 (13건 중 4위).** 1190 에 필요한 로컬 이득:
중앙 전이율 가정 **+18.7**, v335 수준 전이율 가정 **+9.3**.

**로컬 이득에 고정 전이율을 곱해 "예상 점수"를 만들지 마라.** v296 이 그렇게 해서
low/center/high `1180.23/1185.67/1194.44` 가 실제 `1172.86` 으로 전부 반증됐다.
위 숫자는 필요 규모 감각용이지 예측이 아니다.

---

## 2. ★RMS 항등식 — 모든 후보의 1차 게이트★

단일 가산 신호를 최적 용량으로 주입할 때 `gain* = 100000 · RMS*² / D` 가 성립한다
(D = r(1-r) ≈ 0.2498, RMS* = 최적 용량에서의 **전체 행** 예측 이동량).

| 목표 | 필요 Public 이득 | 필요 RMS (최적용량) |
|---|---:|---:|
| 1185 | +3.290 | **0.002867** |
| **1190** | **+8.290** | **0.004551** |

**모든 후보 보고에 전체행 RMS 이동량을 반드시 기재하라.** 미달이면 신호 품질과 무관하게
목표 도달이 수학적으로 불가능하다. 참고: v320 의 전체행 RMS 는 `0.0017257` 였다.

**1190 을 단일 후보로 노린다면 RMS 가 v320 의 2.6배 이상이어야 한다.**
현실적으로는 **서로 직교하는 후보 3~5개의 누적**이 유일한 경로다. 그렇게 계획하라.

---

## 3. ★★이번 사이클 최대 발견 — 강건성 게이트가 오보정돼 있다★★

게이트 상태를 확인할 수 있는 확정 제출 7건:

| ver | Public | 게이트 | 근거 |
|---|---:|---|---|
| v104 | **+1.4282** | 실패 | crossed p05 −2.476, Reality p=.117 |
| v116 | −1.0325 | 실패 | pitcher/crossed p05 음수 |
| v167 | **+1.7758** | 실패 | source stability·bootstrap 게이트 실패 |
| v180 | −0.0387 | **통과** | 3개 bootstrap p05 전부 양수, Reality p=.0025 |
| v198 | −1.0831 | 실패 | Reality Check p=.1489 |
| v290 | **+0.7824** | 실패 | block p05 −0.233, crossed p05 −6.904 |
| **v335** | **+4.9529** | 실패 | crossed p05 **−5.958** |

- 게이트 **실패** 6건 → 양수 **4/6**, Public 합 **+6.82**
- 게이트 **통과** 1건 → 양수 **0/1**, Public **−0.04**
- **Public 이득 상위 4건이 전부 게이트 실패**
- `crossed pitcher×batter p05` 음수인 3건 → **3/3 양수**, 합 **+7.16**

**결론: crossed bootstrap p05 와 Reality Check 를 단독 기각 사유로 쓰지 마라.**
표본이 작고(n=7) 선택편향이 있으므로 "게이트 실패 = 좋다"로 뒤집지도 마라.
**진단으로만 쓰고, 기각은 locked 축 이득의 부호와 월별 안정성으로 하라.**

v335 는 crossed p05 −5.958, 이득의 96% 가 투수 5명 집중, late-2023 multiplicity 실패
상태로 제출됐고 전이율 0.894 를 냈다. 이것이 이 재보정의 직접 근거다.

---

## 4. ★재고: locked full-2024 이득이 양수인데 승격되지 않은 실험 40건★

`artifacts/*/summary.json` 전수 조사 결과 상위:

| artifact | status | locked 2024 | 양수월 |
|---|---|---:|---:|
| `v221_joint_role_h1_source_blend` | screen_reject | **+6.2976** | **1.00** |
| `v330_leave_one_origin_baseball_moe` | cross_origin_reject | +3.4852 | 0.88 |
| `v243_runtime_faithful_route_reaudit` | needs_confirmation | +2.3991 | 0.50 |
| `v50_reproduced` | — | +2.1194 | 0.38 |
| `v209_h1_workload_multiseed_audit` | robust_reject | +1.6365 | 0.75 |
| `v261_regular_calendar_expert_gate` | locked_or_robust_reject | +1.6170 | 0.67 |
| `v260_mechanism_complement_experts` | locked_or_robust_reject | +1.4514 | 0.50 |
| `v216_joint_workload_h1_multiseed_audit` | robust_reject | +1.4302 | 0.75 |
| `v206_augmented_h1_weight_scope_audit` | point_pass_robust_reject | +1.4010 | 0.75 |
| `v311_platoon_partial_pooling_strict_audit` | source_reject | +1.3393 | 0.62 |
| `v293_futures_population_lgb_diversity` | rejected | +1.2130 | 0.50 |

**주의: 이 이득들은 각기 다른 시점의 부모에 대해 측정됐다.** 대부분 v218~v290 시대
(~Public 1172~1177) 부모다. 챔피언은 그 뒤 v244(+0.70)·v290(+0.78)·v335(+4.95)를 얻었으므로
**상당수가 이미 중복 설명될 수 있다. 반드시 v335 부모로 재기준화한 뒤 다시 재라.**

### 4-1. 최우선 리드 — joint workload H1

`v221` 의 `+6.2976` 을 오독하지 마라. 그 실험의 결론은 `locked_incremental_gain: −0.0595`,
즉 **role 블렌드가 joint 단독보다 나쁘다**는 것이다. 기각된 건 role 성분이다.

진짜 신호는 그 아래에 있다.

| 실험 | 2022 | 2023 | 2024 | 비고 |
|---|---:|---:|---:|---|
| v214 screen (seed42, scale 0.1) | +5.79 | +6.24 | **+6.36** | 양수월 100% (2022·2024) |
| **v216 `strict_ensemble`** | **+3.50** | **+8.01** | **+3.79** | **양수월 100%, 9/9 seed-year 양수** |
| v216 배포 scope (`runners_or_high_li` @0.18) | — | — | **+1.053** | 양수월 4/8 → `robust_reject` |

**기각 사유는 좁은 배포 scope 이지 `strict_ensemble` 자체가 아니다.**
`strict_ensemble` 프로파일(3 origin 전부 양수, 100% 양수월, 다중시드 확인)은
**v335 가 제출될 때보다 강하다.**

**할 일**: joint workload H1 성분을 **v335 부모 위에서** 재기준화하고,
`runners_or_high_li` 가 아닌 더 넓은 scope(예: R 전체 또는 R_CORE)에서 dose 를
**source 축(full-2022, late-2023)으로만** 선택한 뒤 full-2024 를 1회 열어라.

자산: `artifacts/v214_.../joint_h1_year{2022,2023,2024}_seed42.npy`,
`artifacts/v215_.../joint_h1_year*_seed{43,44}.npy`,
`artifacts/v216_.../primary_axis.npz` (h1_parity_max_abs 가 세 축 모두 0.0 이므로
성분 적용 기계는 정확히 재현된다). 적용 함수는
`src/archive/v205_h1_workload_strict_forward_audit.py` 의 `apply_component_delta`, `strict_axis`.

---

## 5. 메커니즘 확장 — v335 가 검증한 것을 체계화하라

v335 가 Public 으로 증명한 것은 **"새 구조를 학습"이 아니라
"이미 검증된 frozen 성분을 아직 적용되지 않은 route 로 이식"** 이다.

이 프로젝트에는 frozen 성분이 많다 — v17/v21/v22 EB 맵, v20 TrackMan 증류,
v25 post-break direct spline, v50 lowrank, v84 rank-16 FM, v104 FM+조건부 LGBM,
v114/v131/v135/v160(v165 signed stack), v131/v133 H1, v142 C3 contrast 등.

**체계적 조사를 하라.**

1. 각 frozen 성분이 **현재 어느 route 에 적용돼 있는지** 인벤토리를 만들어라
   (R_CORE / R_ANCHOR / F / 기타).
2. **미적용 (성분 × route) 셀**을 전부 나열하라.
3. source 축 2개로 선별하고 locked full-2024 는 1회만 열어라.
4. RMS(§2)와 §3 의 재보정된 기각 기준을 적용하라.

**v50 lowrank 는 소진됐다** — 이제 F + R_ANCHOR 에 적용돼 있고, 나머지 R_CORE 에서는
음수다(`v314` ALL route locked **+0.630**, 3/8 양수월, source_reject).
**다른 성분으로 같은 일을 하라.**

### 5-1. 하지 말 것

- **Public 결과로 route/team/dose/seed 를 재선택하지 마라.** v335 가 team 13 에서 성공했다고
  team 16 을 추가하는 것은 금지다(full-2024 라벨 마이닝이자 Public 재튜닝).
- 이미 반증된 `|보정| >= tau` 크기 게이트를 다시 시도하지 마라 (§8).

---

## 6. 완전히 새로운 축 — §4·§5 로 8.29 가 안 나올 때

과거 진단(아래 §8)에서 확립된 사실을 출발점으로 삼되, **그 진단에 갇히지 마라.**
그 진단들은 "인컴번트 위의 가산 보정" 패러다임 안에서 얻은 것이다.

- 신호는 압도적으로 **투수**에 있다(투수 ID 상한 840). **타자 축은 거의 비어 있다**(상한 10.4).
- `asof_*` 30여 개 연속 피처는 사실상 **투수 지문**이라, 용량 있는 학습기는 시즌 고유 잔차를
  외운다: 잔차 학습 source(2022 OOF) **+495** → locked(2024) **−301**. 전이 최대치 **+4.4**.
- 따라서 **저용량·frozen·route 이식**이 통하고 **고용량 학습**은 통하지 않는다.

이 제약을 우회하는 방향을 찾아라. 예시(우선순위 아님, 지시도 아님):

- **표현 자체를 바꾸는 base 모델** — 가산 보정이 아니라 독립적인 base representation.
- **다중 목적함수** — 4-way outcome(성공/reverse/middle/wide) 복원은 이미 있다
  (`src/champion/v22_failure_mode_profiles.py`). 그 위의 구조는?
- **계층적/베이지안 부분풀링**을 route 가 아니라 **투수 유형**으로.
- **TrackMan 재검토** — 물리 신호 자체는 약했으나(strict-forward 최적 +0.737),
  링키지는 hyunku 맵(2024 커버 99.78%)이 정확하다(§9). 조합 방식이 문제였을 수 있다.
- **F 경로 심화** — F 는 30,010행(11.8%)인데 BSS 716 으로 R(1004)보다 크게 낮다.
  v320/v335 가 F 에서 큰 이득을 냈다는 사실이 여기 여지가 더 있음을 시사한다.

**공개 저장소는 이미 전수 감사했고 우리보다 앞선 곳은 없다**(§9). 거기서 찾을 것은
모델 성능이 아니라 **도메인 사실**이다. ABS 단서도 ttkkwan 의 용어집 한 줄에서 나왔다.

---

## 7. 반드시 지킬 계약

- 공식 지표 유지. 같은 검증 행에서의 **paired Brier** 비교.
- **모델 적합은 공식 train + 2019~2024 TrackMan 만.** 외부 야구 데이터 금지.
  공개 방법론은 공식 데이터로 **독립 재구현한 경우만** 인정.
- 각 test 행은 **독립 예측**. 다른 test 행의 빈도·분포·평균·순서·집계·rolling 금지.
- **리더보드 보정 프로브 영구 금지** — `data_description.md` §5 의
  "평가 데이터 전체를 보고 만든 사후 보정값" 에 해당한다. Claude 가 규정 검토 후 폐기했다.
- **Public 사용**: 새 계열의 가설 검증·챔피언 승격 판단에만. 같은 계열의 용량·route·seed·
  보정값 재선택에는 쓰지 마라.
- **챔피언 ZIP 과 기존 사용자 변경 보존.** 파괴적 git/filesystem 명령 금지. 새 버전·새 경로.

### 7-1. 패키지 감사 게이트 (v335 에 적용한 절차를 그대로)

1. 행 singleton / shuffle / partition 불변성
2. 비활성 경로 exact-parity (챔피언과 최대 절대차 0)
3. **245,789행 프록시 런타임** (제한 600초) + 정적 금지연산 스캔
4. finite, `[0,1]`, row_id 순서, CRC, SHA-256, requirements 고정

**⚠ 경로 커버리지에 주의하라.** 공식 `data/test.csv` 와 `f_route_format_test.csv` 에는
**team_id 13 행이 0개**여서 R_ANCHOR 경로가 한 번도 스케일 테스트되지 않았었다.
Claude 가 만든 3경로 커버 프레임을 재사용하라:
`artifacts/c340_v335_scale_audit_20260830_01/route_coverage_test.csv` (9행, 48열, BOM 유지).
`scale_proxy` 가 격행을 F 로 강제하므로 **홀수 길이**로 두어야 모든 base 행이 두 경로를 거친다.
대규모 parity 스크립트: `artifacts/c340_.../parity.py`, `artifacts/c341_.../parity.py`.

**기존 조건 1건**: `requirements.txt` 는 `scikit-learn==1.6.1` 핀인데 일부 estimator 가
1.7.2 로 pickle 되어 `InconsistentVersionWarning` 이 난다. v290·v320·v334·v335 가 동일하고
v290·v335 모두 공식 채점을 통과했으므로 신규 위험은 아니다. 재패키징 시 정렬하면 좋다.

---

## 8. 닫힌 방향 (재시도 금지) 및 철회된 주장

### 8-1. 닫힌 방향

| 방향 | 결과 |
|---|---|
| 표준 GBDT 정면 재구축 | oracle level 보정 후에도 154.7~201.5 (당시 인컴번트 1008.5) |
| 전역 intercept/slope 재보정 | oracle Platt 이득 +3.41 이 상한 |
| 조건부 재보정 (pred_bin/bn_bin/inning) | 순열 잡음 이하 |
| 시즌 상대화 featurization | 원시 201.5 vs 상대화 195.9 — 무효 |
| 강건 단순모델과의 블렌드 | 2022·2024 양쪽 단조 음수 |
| F-1군경력 투수 psr 기울기 교정 | 모델 0.2997 vs 참값 0.3007 → 교정량 0 |
| **2024 R ABS regime 전문가 (v296)** | **Public 1172.86, 계열 폐쇄** |
| v50 lowrank ALL route (v314) | locked +0.630, 3/8 양수월, source_reject |
| **`\|보정\| >= tau` 크기 게이트 (C344)** | 2022 가 고른 q80 이 locked **−1.117**. team13 을 못 이김 |
| F direct dose 30% (v332) | locked −0.641, 20% 포화 |
| 4-class 실패유형 student rebase (v333) | locked −0.143 |
| 리더보드 global-shift 프로브 | 규정 위반, 영구 폐기 |

### 8-2. Claude 가 철회한 주장 (믿지 마라)

1. **"기존 trackman 실험이 26% 오염된 링키지로 평가됐다" — 틀렸다.** 실험들은
   `reports/trackman_linkage.csv` 가 아니라 외부 `hyunku/assets/pitcher_map.csv`
   (2024 커버 99.78%)를 썼고, 그건 Claude 신규 맵·kyungjun·seokjin 맵과 **100% 일치**한다.
   v252/v255/v257/v258 의 기각은 유효하다.
2. **"인컴번트가 late-2023 에서 붕괴하니 과적합" — 틀렸다.** late_2023 축은 v288 프로토콜상
   2023 상반기만으로 학습(~15만행)한다. 학습량 차이일 뿐이다.
3. **"인컴번트가 F-1군경력 투수의 psr 을 2배 과대반영" — 틀렸다.** 기울기 회귀로 분리하면 정확하다.

---

## 9. 재사용 가능한 자산

- **저장된 예측 축**
  - `artifacts/v288_futures_multiseed_fixed_audit_20260830_01/combined_axes.npz`
    (`baseline_full_2024`=v244, `candidate_full_2024`=v290, `*_late_2023` 동일)
  - `artifacts/v318_.../selected_axes.npz` (v320 의 parent/candidate/direction, late23·full24)
  - `artifacts/v314_.../selected_axes.npz` (v50 lowrank 의 origin별 strict-forward 보정)
  - `artifacts/v285_.../selected_axes.npz` (full_2022 부모)
  - `artifacts/v173_.../*.npz` (full_2022 챔피언 계열 예측)
- **부모 성숙도** (같은 축 내부 5-fold OOF 로 회수 가능한 잔차):
  full_2022 **453.67** / late_2023 **1218.04** / full_2024 **124.89**.
  source 축 부모가 미성숙할수록 큰 용량이 선택되고 성숙한 부모에 과다투여된다.
  보정 규칙 `w_locked ≈ w_source × (여유_locked/여유_source)` (2022→2024 계수 **0.275**).
  **late_2023 은 반시즌 학습이라 dose 선택축으로 부적합하다.**
- **팀 매핑** (Claude 확정): `artifacts/c321_.../team_map.json` —
  12=DOO_BEA, 13=LG_TWI, 14=KIW_HER, 15=LOT_GIA, 16=KIA_TIG, 17=HAN_EAG,
  18=SAM_LIO, 19=NC_DIN, 20=KT_WIZ, 21=SK_WYV→SSG_LAN.
  **main `pitcher_hand=1` = 좌완.** trackman `MIN_*`=퓨처스, `KBO_/ACE_`=특수팀.
  main R = 팀ID 12~21, F = 12~21 + 22,23,25.
- **도메인 지식** (train-only, 5시즌 안정): `logit(shrunk psr)` → 성공의 로지스틱 기울기
  R **1.1335** / F-F전용 **0.7145** / F-1군경력 **0.3007**.
- **ABS regime**: 1군 2024, 퓨처스 2023 도입. R 시즌고유 `middle_rate` 2023 .15318 →
  2024 .17602(시계열 최대폭). F prevalence 2022 .70875 → 2023 .47290.
  **2025 는 양쪽 모두 ABS.** (단 v296 의 "최근 시즌 절대 R 전문가"는 실패했다.)
- **공개 저장소 감사**: 12개 전수 확인, 실제 Public 은 swimmer 1167.92, hoo 1157.81,
  calico 1126.45, thddydgnl 1090.91, kyungjun 1054.34, ttkkwan 1038.
  **우리보다 앞선 곳은 없다.** thddydgnl 의 4-way outcome 복원은 이미 우리 것이다.
- **Claude 분석 전체**: `artifacts/c3xx_*` (c300~c345). `vNNN_` 과 충돌하지 않는다.
- **조율 문서**: `reports/claude_codex_coordination_20260830.md` (1,371줄 + 이번 갱신)

---

## 10. 실행 방식

- 각 실험에 가설, 데이터 범위, 학습/검증 origin, 부모 SHA, 선택 축, 잠금 축, 결과,
  승격/기각 이유를 `artifacts/vNNN_.../summary.json` + 짧은 보고서로 남겨라.
- 진행 상황은 `reports/claude_codex_coordination_20260830.md` 에 이어서 갱신하라.
- **후보 보고에는 반드시 전체행 RMS 이동량을 포함하라**(§2).
- 로컬 양수만으로 제출을 권고하지 마라. 일 1회 예산에서 기대 Public 이득 1점 미만은
  제출 가치가 없다. 단, §3 에 따라 crossed bootstrap·Reality Check 단독 실패로 기각하지 마라.
- 사용자 결정이 필요한 진짜 blocker 가 아니면 멈추지 말고 합리적 가정 아래 계속 진행하라.
- **1190+ 가 공식 확인되기 전에는 완료라고 선언하지 마라.**

### 10-1. 현실적 계획 감각

관측된 단일 후보 상한은 Public **+1~5** 이고 격차는 **8.29** 다.
**서로 직교하는 후보를 3~5개 확보해 누적하는 계획**을 세워라.
후보 하나가 제출 대기 중일 때 다음 후보 연구를 멈추지 마라.
매 제출 후 실제 전이율을 기록해 §1-2 표를 갱신하고 다음 후보의 규모 요건을 역산하라.
