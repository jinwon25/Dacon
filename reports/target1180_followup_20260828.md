# Target 1180 후속 연구 및 재현 보고서 — 2026-08-28

## 결론

현재 공식 챔피언은 그대로 유지한다.

- Public: `1172.1373858439`
- 파일: `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip`
- SHA-256: `4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E`
- 1180까지 남은 점수: `7.8626141561`

이번 라운드에서는 1180 상당의 승격 가능한 후보를 확보하지 못했다. 여러 후보가
2024 한 축에서는 양수였지만, 두 독립 소스 축의 부호 일치·월 안정성·cluster bootstrap을
동시에 만족하지 않았다. 따라서 새 ZIP을 만들거나 기존 챔피언을 덮어쓰지 않았다.

대신 다음의 재현 가능한 결과를 확보했다.

1. 공식 JY 계약을 실제 런타임 순서로 정확히 재구성했다.
2. 과거 sequential gate 감사가 공식 Brier Skill Score가 아니라 log loss 합을 사용한 문제를
   분리하고 BSS 재감사기를 추가했다.
3. v131 비핵심 도메인 감사가 예측을 바꾸면서도 지표 함수에서 `R_CORE`만 활성 도메인으로
   고정해 `F`/`R_ANCHOR` 효과를 0처럼 기록한 사각지대를 발견하고 재검증했다.
4. 공식 손실함수에 맞춘 RMSE-H1과 train-only 선수 노출 수준 방향을 새로 검증했다.
5. 최신 공개 구현의 8월 24일 이후 변경도 확인했지만, 리더보드로 정한 계수를 재사용하지
   않고 strict-forward OOF에서 독립 검증해 기각했다.

## 1. 저장소 및 작업 상태 감사

### 원본 작업 폴더 보존

원본 작업 폴더는 `codex/target-1185-followup`의 과거 커밋 `0d2ae60`에 있었고,
팀원 변경과 미추적 파일이 다수 존재했다. 해당 작업 트리의 소스·데이터·아티팩트는 수정하지
않았다.

실험은 다음 별도 worktree에서만 수행했다.

```text
E:\학업\교외 활동\대외 활동\공모전\데이콘\.git-worktrees\pitch-control-target-1180
branch: codex/target-1180-followup-20260828
base: team/main 02a8fac5dddfadf1bbbeaf4ab0928a4ef6762f76
```

공식 데이터는 새 worktree에 hardlink로 연결했고 원본과 SHA-256이 일치함을 확인했다.

| 파일 | 크기(byte) | 검증 |
|---|---:|---|
| `train.csv` | 368,527,723 | 원본 SHA 일치 |
| `trackman_history.csv` | 353,823,031 | 원본 SHA 일치 |
| `test.csv` | 1,894 | 5행 smoke 파일 |
| `sample_submission.csv` | 112 | 원본 SHA 일치 |

### 현재 단일 버전 실행 구조

현재 구조를 과거 방식으로 되돌리지 않았다.

- 공식 후보 생성 진입점: `scripts/build_jy_next_variants.py`
- 순차 gate 감사: `scripts/audit_jy_sequential_gates.py`
- 최종 JY 감사: `scripts/audit_jy_next_final.py`
- standalone 공통 감사: `python -m src.audit_standalone_release`
- 현재 릴리스는 ZIP 루트의 `script.py`, `requirements.txt`, `model/`만으로 실행한다.
- 추론 입력은 `data/test.csv`, 형식 기준은 `data/sample_submission.csv`, 출력은
  `output/submission.csv`다.
- 모델 및 고정 테이블은 ZIP 내부 `model/` 아래에 포함한다. Git 추적 OOF 계약은
  `artifacts/oof_champion_1161/`, `artifacts/oof_champion_1170/`에 있다.
- 일반 버전 코드는 `src/champion/`과 `src/archive/`, 공통 구현은 `src/core/`에 분리된 현재
  구조를 유지했다.

공식 릴리스는 172개 member, 83,675,308 byte이며 CRC와 저장소 기록 SHA가 일치했다.
245,789행 전체 규모 standalone 감사는 `210.18초`로 공식 600초 제한 안에 완료됐다.
singleton/shuffle/partition 최대 오차는 모두 `0`, 금지 패턴도 `0`이었고 출력은 전부
유한한 `[0, 1]` 범위였다.

최종 검증 결과는 다음과 같다.

- `python -m pytest -q`: `397 passed, 21 skipped`
- `python scripts/audit_repository.py --include-untracked`: 844개 파일, 금지 경로·대형 파일·
  비밀정보·LFS 오설정·고정 바이너리 불일치 모두 0
- 신규 스크립트 `compileall`: 통과
- 챔피언 ZIP SHA-256: 저장소 고정값과 일치

### 재현성 위험

릴리스 요구사항은 `scikit-learn==1.6.1`인데 일부 pickle은 `1.7.2`에서 생성된 흔적이 있다.
현재 환경에서는 정상 추론되지만, 다음 챔피언을 만들 때는 1.6.1 격리 환경에서 재직렬화와
standalone 감사를 반복해야 한다. 이번 라운드는 챔피언 ZIP을 수정하지 않았다.

## 2. 평가 계약 재감사

공식 평가는 Brier Skill Score이므로 후보 간 순서는 paired squared-loss 감소로 판단해야 한다.
기존 `audit_jy_sequential_gates.py`는 log-loss 합을 사용했다. 새
`scripts/audit_jy_sequential_gates_brier.py`는 동일 후보군을 paired unclipped BSS gain으로
다시 계산한다.

결과:

- 기존 선택 `bridge027`은 동일 후보군에서 여전히 1위였다.
- 다만 정확한 2024 BSS 증분은 약 `+0.5006`, 최악 월은 `-0.1073`이었다.
- 따라서 과거 log-loss 수치를 BSS 점수 증가로 해석하면 안 된다.

`v168_jy_exact_contract_reaudit`에서 실제 런타임 순서를 재구성했다.

- identity 공식 parity 최대 오차: `1.11e-16`
- 과거 v160 감사와 실제 런타임 순서 최대 오차: `0.0002830`
- 순서 수정 자체의 2024 증분: `+0.07693 BSS`
- 공식 runners/high-LI JY 전체 효과: `+0.61242 BSS`
- bridge-only: `+0.53767 BSS`
- H1/C3-only: `+0.08942 BSS`

현재 JY 미세조정만으로 7.86점을 메울 근거는 없다.

## 3. 신규 및 재기준화 실험

아래 값은 모두 현재 정확 부모 또는 실제 JY 계약 대비 paired BSS 증분이다. 2024는 잠금
진단이며 반복 연구로 오염된 축임을 전제로 해석했다.

| 실험 | 소스 근거 | 2024 잠금 결과 | 판정 |
|---|---|---|---|
| v169 독립 CatBoost/DCNv2 잔차 | capped-CB disagreement gate가 2022 `+0.77`, late23 `+7.59` | 최대 `+1.41`, 최악 월 `-10.50` | 강건성 미달 |
| v170 현재 잔차 CatBoost | late23 최고도 `-39.55` | pre-refit `-13.30`, refit `-14.58` | 소스 기각 |
| v171 비핵심 도메인 잔차 | `R_ANCHOR`/`F`에서 연도별 최적 부호 반전 | 일관 방향 없음 | 기각 |
| v172 H1 시드 부분집합 | source 선택 seed42는 점 안정성 실패 | `+0.145`, bootstrap p05 모두 음수 | 기각 |
| v173 H1 비핵심 확장 | `F`: 2022 `+3.75`, late23 `-9.18` (w=.05) | `+1.46`, 강건하지만 소스 부호 반전 | 기각 |
| v174 RMSE-H1 | 2022 `+0.618`, late23 `-0.009` | 소스 gate 실패로 미개봉 | 기각 |
| v175 RMSE-H1 압박 카운트 | 2022 `+0.472`, late23 `+0.421`, 월 비율 미달 | `+0.056`, bootstrap p05 음수 | 기각 |
| v176 train-only 선수 노출 수준 | 최선도 2022 `-0.749`, late23 `+1.198`; 타자 노출도 같은 부호 반전 | `-0.072` | 기각 |

### v131 도메인 감사 사각지대

과거 v131의 `ALL` H1 후보는 실제로 비핵심 행을 바꿨다.

- full-2022 `F` 변경 행: 30,448
- late-2023 `R_ANCHOR` 변경 행: 16,577
- late-2023 `F` 변경 행: 8,140

그러나 공통 `_axis_metrics`가 `_diagnostics(..., ("R_CORE",), ...)`로 고정되어 비핵심
도메인 이득이 0처럼 요약됐다. v173은 이를 실제 BSS로 다시 계산했다. 2024 `F`는 양수였지만
2022와 late-2023 사이 부호가 반전하므로 배포하지 않는다.

### 공식 지표에 맞춘 H1

현재 H1은 `Logloss` CatBoostClassifier다. v174는 동일 82피처·깊이 8·1,200트리 조건에서
`RMSE` CatBoostRegressor를 strict-forward로 학습했다. 전체 경로는 late-2023에서 거의
중립이었고, 압박 카운트로 제한한 v175도 2024 전달 이득이 소멸했다. 이 결과는 단순히
분류 손실을 squared-error로 바꾸는 것만으로는 새 해상도를 만들지 못한다는 근거다.

### 최신 공개 구현의 선수 노출 수준 방향

공개 `hoo743-ui/LG_Aimers09`는 8월 24일 스냅샷 이후 타자 노출 수 방향을 리더보드에서
탐색했다. 사용자 제약에 따라 그 계수와 리더보드 정점은 재사용하지 않았다.

v176은 다음과 같이 독립 재정의했다.

- 감사 연도 이전 두 시즌의 공식 train 행만 집계
- batter/pitcher별 노출 수 표의 center/scale도 과거 train에서만 계산
- 미등록 선수는 0 보정
- test 행 간 groupby, test 분포 center/scale, Public 점수 선택 없음

이 방향은 full-2022에서 음수, late-2023에서 양수로 반전했다. 공개 리더보드 개선을 현재
모델에 이식하면 안 된다는 직접 근거다.

## 4. 누수 및 규칙 준수

모든 신규 스크립트는 다음을 결과 JSON에 명시한다.

- 공식 train만 fitting에 사용
- 현재 audit target은 feature나 보정 적합에 사용하지 않음
- `test.csv` 미열람
- test 집계·순위·빈도·분포·center/scale 미사용
- 다른 test 행과 정보 공유 없음
- Public 점수로 recipe/weight 선택 없음
- row-local inference 가능

TrackMan은 과거 공식 제공 기록만 사용한다. 현재 audit/test 투구의 위치·구종·물리량은 읽지
않는다. 공개 저장소 검토에서도 외부 KBO 데이터 경로와 평가행 집계 기반 보정은 제외했다.

## 5. 재현 명령

공통 경로 변수는 실제 old research artifact 경로로 바꿔야 한다. 생성 아티팩트는 Git에서
제외된다.

```powershell
python -m pytest -q

python scripts/audit_jy_sequential_gates_brier.py --help
python -m src.archive.v168_jy_exact_contract_reaudit --help
python -m src.archive.v169_independent_residual_gate_audit --help
python -m src.archive.v170_current_residual_catboost --help
python -m src.archive.v171_independent_domain_residual_audit --help
python -m src.archive.v172_h1_seed_subset_audit --help
python -m src.archive.v173_h1_noncore_extension_audit --help
python -m src.archive.v174_brier_h1_regressor --help
python -m src.archive.v175_brier_h1_pressure_audit --help
python -m src.archive.v176_train_only_exposure_level_audit --help
```

주요 결과 디렉터리:

```text
artifacts/jy_brier_metric_reaudit_20260828_01
artifacts/v168_jy_exact_contract_reaudit_20260828_01
artifacts/v169_independent_residual_gate_audit_20260828_01
artifacts/v170_current_residual_catboost_20260828_01
artifacts/v171_independent_domain_residual_audit_20260828_01
artifacts/v172_h1_seed_subset_audit_20260828_01
artifacts/v173_h1_noncore_extension_audit_20260828_01
artifacts/v174_brier_h1_regressor_20260828_01
artifacts/v175_brier_h1_pressure_audit_20260828_01
artifacts/v176_train_only_exposure_level_audit_20260828_01
```

## 6. 다음 승격 조건

현재 증거로는 작은 gate/weight 조정보다 독립 예측 축이 필요하다. 다음 후보는 최소한 다음을
만족할 때만 ZIP으로 만든다.

1. full-2022와 late-2023 정확 현재 부모 대비 둘 다 양수
2. 잠금 2024에서도 양수이며 월별 최악값이 허용 범위 안
3. pitcher, crossed pitcher-batter, chronological block bootstrap p05가 모두 양수
4. 제한된 후보군에 대한 Reality Check 통과
5. 현재 JY standalone에 row-local하게 구현 가능
6. 격리된 공식 요구사항 환경에서 singleton/shuffle/partition 및 runtime 감사 통과

현 시점의 가장 큰 병목은 새 피처 하나가 아니라 **2025로 전달되는 독립 OOF 방향의 부재**다.
새 팀원 모델 OOF나 완전히 다른 학습 구조가 생기기 전에는 2024 양수만 보고 제출을 늘리는
것보다 1172.137 챔피언을 보존하는 편이 합리적이다.

## 7. 추가 후속 결과: v178/v180

위 6절의 판단 뒤 독립 OOF 축을 하나씩 더 만드는 대신, 이미 서로 다른 오차를 가진 고정
컴포넌트의 **signed stack을 현재 JY 부모 위에 재기준화**했다. 계수는 v165에서 이미 동결된
값을 그대로 사용하고, source-only 두 축(full-2022, late-2023)에서 강도 `0.25`를 선택했다.
잠금 2024는 선택에 사용하지 않고 최종 감사에만 사용했다.

### v178 다축 OOF 감사

| 축 | 현재 JY 대비 BSS gain | 세부 안정성 |
|---|---:|---|
| full-2022 | `+0.748373` | 7개월 중 6개월 양수, 최악 `-0.718773` |
| late-2023 | `+0.617614` | 3개월 모두 양수, 최악 `+0.306289` |
| locked 2024 | `+0.193590` | 8개월 중 5개월 양수, R_CORE `+0.274818` |

잠금 2024 행을 대상으로 한 2,000회 재표집 결과도 모두 양수였다.

- pitcher cluster bootstrap p05: `+0.203039`
- crossed pitcher-batter bootstrap p05: `+0.006789`
- chronological moving-block bootstrap p05: `+0.251453`
- White Reality Check: 최종 후보 `0.25`와 no-op 부모만 비교, `p=0.002499`

v179 ablation에서는 Jiyun 컴포넌트를 제거하면 이 안정성이 사라졌다. 따라서 v180은 Jiyun을
포함한 v178 공식을 그대로 배포하는 것으로 고정했고, Public 점수에 맞춰 계수를 다시 조정하지
않았다.

### v180 최종 재학습과 독립 실행 패키지

Jiyun LightGBM/CatBoost의 iteration은 2019--2023 학습/2024 검증으로만 정한 뒤,
2019--2024 전체에 각각 `193`/`324` iteration으로 최종 재학습했다. 2025 test의 다른 행,
전체 분포, 예측 평균 또는 Public 점수는 학습·보정·선택에 사용하지 않았다.

- ZIP: `artifacts/v180_signed_stack_package_20260828_01/submit_v180_signed_stack.zip`
- 크기: `86,084,812 bytes`
- 파일 수와 루트: 179개, `model/`, `script.py`, `requirements.txt`
- SHA-256: `91CA020EFA776BA12BF50630FCE533A06EE6E6984FD89C26E5C20BB5FFB5AAC4`
- 현재 챔피언 재구성 오차: `5.55e-17`
- v178 공식 재현 오차: `5.55e-17`
- singleton/shuffle/partition 최대 오차: `1.67e-16`
- 245,789행 환산 실행: `177.248초` (공식 제한 600초 이내)
- 상태: `eligible_for_api_submission`

재현 진입점은 현재 저장소 구조를 유지한다.

```powershell
python -m src.archive.v178_jy_signed_stack_rebase --help
python -m src.archive.v179_deployable_signed_stack_ablation --help
python -m src.champion.v180_finalize_jiyun --help
python -m src.champion.v180_build_signed_stack_package --help
python -m src.audit_standalone_release --package artifacts/v180_signed_stack_package_20260828_01/submit_v180_signed_stack.zip --test-csv data/test.csv --sample-rows 5 --scale-rows 245789 --timeout-seconds 600
```

최종 회귀 검증은 `402 passed, 21 skipped`, 저장소 감사 852개 파일 전체 통과,
신규 모듈 `compileall` 통과다.

## 8. DACON 제출 결과

`2026-08-28 14:52 KST`에 v180 ZIP을 API 제출하려 했으나, 서버의 사전 검증이 저장된
토큰을 유효하지 않은 토큰으로 거부했다.

- API 응답: `isSubmitted=false`
- 업로드: 시작되지 않음
- 일일 quota: 차감되지 않음
- v180 Public 점수: 아직 없음
- 기존 챔피언: Public `1172.1373858439` 및 release ZIP 그대로 보존

새 DACON API 토큰을 git 비추적 `.env`에 갱신한 뒤 같은 SHA의 ZIP을 그대로 제출해야 한다.
그 전에는 v180을 챔피언으로 승격하거나 `reports/submissions.csv`에 점수를 기록하지 않는다.

이후 사용자가 Downloads에 복사된 동일 ZIP을 `submit_v180.zip`으로 직접 제출해 결과를
확인했다.

| 항목 | 결과 |
|---|---:|
| 제출 ID | `71754` |
| 제출 시각 | `2026-08-28 14:59:16 KST` |
| v180 Public | `1172.0987352738` |
| 현재 챔피언 Public | `1172.1373858439` |
| 챔피언 대비 | `-0.0386505701` |
| v167 대비 | `+0.0214972417` |
| 1180까지 gap | `7.9012647262` |

결론은 **기각·미승격**이다. v180은 v167보다 소폭 높았지만 직접 부모인 현재 JY 챔피언보다
낮았다. 작은 차이를 근거로 같은 signed-stack scale을 Public에 재적합하지 않는다. 현재 공식
챔피언과 release ZIP은 변경하지 않고, v180 코드·OOF 감사·재학습 패키지는 재현 가능한 음성
결과로 보존한다.

## 9. v180 Public 이후 독립 후속 연구: v181--v192

v180의 Public 하락을 계수 재적합 신호로 사용하지 않았다. 대신 기존에 고정된 공식
strict-forward OOF 계약을 유지하면서, 계층 잔차·상황별 제구 프로필·연도 완전 분리
stacking을 새로 검증했다. 공개 저장소의 예측은 어떤 배포 ZIP에도 포함하지 않았다.

| 실험 | 핵심 방향 | 판정 근거 |
|---|---|---|
| v181 | 공개 구현 독립 OOF screen | 최신 공개 저장소에 명시적 라이선스가 없어 연구 전용 |
| v182 | v178 + 공개 shallow-context 보완축 | 세 축에서 양수였지만 라이선스·로컬 재현 계약 미충족으로 배포 금지 |
| v183 | all-month signed-stack 재기준화 | locked 2024 `-0.602`, 기각 |
| v184 | 선수 계층 base + Hoo 잔차 CatBoost | 2022 `+8.968`, late-2023 `+5.381`이나 2024 `-1.541`, 기각 |
| v185 | 공개 보완축 성분 분해 | hierarchy가 아니라 얕은 상황 잔차가 보완 신호라는 진단만 채택 |
| v186 | 선수 identity-free shallow adaptive residual | early-2022→late-2022 `-14.6`, 소스 기각 |
| v187 | 전년도 선수별 압박 상황 프로필 | 최선 소스도 `-0.337/+0.344`로 부호 반전 |
| v188 | 압박 프로필 soft transition | locked 2024 총 `+0.255`지만 월 4/8, crossed p05 음수 |
| v189 | 압박 프로필 route screen | locked 2024 총 `+0.322`지만 active subset 하방 음수 |
| v190 | context-adjusted season-stable profile | 소스 평균이 모두 음수 |
| v191 | leave-one-year-out minimax stack | 세 held-out 전체 이득은 양수, 월 안정성 strict gate 미달 |
| v192 | 사전 고정 soft-LOO gate + 3개년 최종 stack | 세 연도 전체 이득 양수, 2024 강건성 통과; 탐색 제출 후보로 승격 |

v184처럼 단일 연도에서 큰 수치를 만드는 모델은 다른 해로 전달되지 않았다. 선수의 제구는
고정된 능력뿐 아니라 시즌별 구종 운용, 부상·피로, 역할 및 공인구/ABS 환경 변화가 섞인
상태이므로, 과거 선수 효과를 강하게 외삽하는 후보보다 서로 다른 기존 축의 작은 signed
조합이 더 안정적이었다.

### v192 연도 완전 분리 결과

`v192`는 각 held-out 연도를 전혀 보지 않고 나머지 두 연도에서 L1 budget을 선택하는
leave-one-year-out 절차를 사용했다. 사전에 고정한 soft gate를 유일하게 통과한 budget은
`0.15`였다. 최종 계수는 Public 점수를 사용하지 않고 세 공식 연도 OOF만으로 다시 적합했다.

| 검증 축 | 현재 JY 대비 BSS gain | 양수 월 | 최악 월 |
|---|---:|---:|---:|
| full-2022 | `+0.621225` | 5/7 | `-0.730234` |
| late-2023 | `+1.031696` | 3/3 | `+0.526391` |
| full-2024 | `+0.219884` | 5/8 | `-0.574667` |

2024의 2,000회 강건성 감사도 모두 양수였다.

- pitcher cluster bootstrap p05: `+0.396169`
- crossed pitcher-batter bootstrap p05: `+0.101447`
- chronological moving-block bootstrap p05: `+0.487605`
- White Reality Check: no-op 포함 최종 2개 후보, `p=0.0009995`

다만 full-2022의 양수 월이 기존 strict point gate `6/7`이 아니라 `5/7`이어서 자동 챔피언
승격 조건은 통과하지 못했다. v180보다 leave-one-year-out 하방과 2024 cluster bootstrap이
개선됐고 세 held-out 전체 이득이 모두 양수라는 점을 근거로, 사용자가 직접 Public을 확인할
수 있는 **탐색 제출 후보**로만 승격했다. 현재 1172.137 챔피언은 그대로 유지한다.

최종 순계수는 다음과 같다.

```text
v135_c3_recent_window:                  +0.0724597608649307
v114_independent_source_stability_mask: +0.015483964114884825
v131_hoo_h1_independent_oof:            -0.062056275020207705
v160_original_h1_affine:                 0.0
```

## 10. v193 독립 실행 제출 패키지

`v193`은 `v192`의 고정 계수를 실제 단일 실행 ZIP에 이식한 버전이다. `v180` 패키지의 179개
모델·테이블·요구사항 파일을 그대로 보존하고, fail-closed 빌더가 루트 `script.py`의 stack
상수와 후보 라벨만 정확히 한 번씩 치환한다. 원본 챔피언과 v180 ZIP은 수정하지 않았다.

| 항목 | 결과 |
|---|---:|
| ZIP | `artifacts/v193_triyear_stack_package_20260828_01/submit_v193_triyear_stack.zip` |
| 크기 | `86,084,836 bytes` |
| SHA-256 | `AEF4FA45CDF0908ACEEDCE06135C04151CC3BAD0A7FCEB5CD851097DDC126DB4` |
| 파일 수 / 루트 | 179 / `model`, `requirements.txt`, `script.py` |
| 현재 챔피언 재구성 최대 오차 | `5.55e-17` |
| v192 공식 재현 최대 오차 | `5.55e-17` |
| shuffle 최대 오차 | `0` |
| partition 최대 오차 | `0` |
| singleton 최대 오차 | `1.67e-16` |
| 금지된 test 행 집계 연산 | 0건 |
| 245,789행 실행 시간 | `197.306초` / 제한 600초 |
| 출력 유한성·범위 | 통과 |

`scikit-learn 1.7.2`에서 직렬화된 일부 기존 객체를 공식 패키지 요구 버전 `1.6.1`에서
읽을 때 호환성 경고가 발생한다. 이는 v180에도 존재한 알려진 위험이며, 이번 패키지는 같은
객체를 바이트 그대로 유지했다. 1.6.1 환경의 챔피언 parity와 전체 규모 추론은 모두
통과했다.

재현 명령:

```powershell
python -m src.archive.v192_soft_loo_triyear_stack --help
python -m src.champion.v193_build_triyear_stack_package `
  --source-package artifacts/v180_signed_stack_package_20260828_01/submit_v180_signed_stack.zip `
  --champion-zip submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip `
  --v192-summary artifacts/v192_soft_loo_triyear_stack_20260828_01/summary.json `
  --data-dir data `
  --output-dir artifacts/v193_triyear_stack_package_20260828_01 `
  --timeout 600
python -m src.audit_standalone_release `
  --package artifacts/v193_triyear_stack_package_20260828_01/submit_v193_triyear_stack.zip `
  --test-csv data/test.csv --sample-rows 5 --scale-rows 245789 --timeout-seconds 600
python -m pytest -q tests
python scripts/audit_repository.py --include-untracked
```

최종 검증은 `425 passed, 21 skipped`, 저장소 감사 879개 파일 전체 통과다. 전역
`pytest -q`는 ignored 공개 연구 복제본 아래의 타 저장소 테스트까지 자동 수집하므로 공식
회귀 명령은 `pytest -q tests`로 고정한다.

## 11. v194--v196 야구 상황별 hierarchy transport 감사

v184의 투수 hierarchy + 상황 잔차 신호가 2022와 2023에서는 강했지만 2024 전체로는
하락했던 원인을 다시 분해했다. 투수-수비팀 점수 차, 이닝, 주자, 아웃카운트, 타석 좌우,
레버리지 등 사전에 정의한 저카디널리티 야구 상황 게이트 130개와 고정 가중치 7개를
검토했다. 테스트 데이터와 Public 점수는 선택이나 보정에 사용하지 않았다.

`v194`의 strict leave-one-origin-out 선택은 2022·2023에서 고른 후보가 2024에서
`-4.251`로 무너져 기각했다. 세 공식 OOF origin을 모두 사용했을 때는
`score__li=close|li_mid`, 가중치 `0.05`가 선택됐다. 이는 투수팀 관점 점수 차가
`-1~+1`이고 `0.7 < li <= 1.5`인 중간 레버리지 접전만 hierarchy 잔차를 적용한다.

`v195`는 같은 130개 게이트 전체를 White Reality Check에 넣어 선택 편향을 감사했다.
leave-one-origin-month-out 18개 블록에서 같은 게이트가 18회 모두 선택됐고 held month의
`83.3%`에서 이득이었다. 2024년 3--6월만으로 선택한 별도 forward 감사에서는
초반 이닝·중간 LI 게이트가 7--10월에 `+0.181`을 냈다. 그러나 v192 위에 더해지는
순수 증분의 강건성 하방과 다중검정은 통과하지 못해 단독 증분은 기각했다.

`v196`은 최종 합성 예측 `v192 + 접전·중간-LI hierarchy` 전체를 원래 JY 챔피언과
직접 비교했다.

| 검증 축 | JY 대비 BSS gain | 양수 월 | 최악 월 |
|---|---:|---:|---:|
| full-2022 | `+2.722121` | 7/7 | `+0.412541` |
| late-2023 | `+2.320557` | 3/3 | `+0.369931` |
| full-2024 | `+0.734403` | 6/8 | `-0.151612` |

2024 pitcher, crossed pitcher-batter, chronological bootstrap p05는 각각
`+0.699033`, `+0.203465`, `+0.735069`이었다. 선택 게이트는 2024에서도 130개 중
1위였지만, no-op과 v192를 포함한 132개 후보 Reality Check가 `p=0.148926`으로
사전 기준 `0.10`을 넘었다. 따라서 point gate는 통과했지만 robust gate는 실패했고,
공식 챔피언은 교체하지 않는다. 다만 여러 연도에서 합산 효과가 일관되고 v180보다
Public 기대 개선 폭이 큰 점을 근거로 탐색 제출 후보의 최종 학습·패키징은 진행했다.

## 12. v197--v198 최종 학습과 독립 실행 패키지

`v197`은 공식 train의 2019--2024년 1,475,092행만 사용해 2025용 투수 hierarchy
스냅샷과 101개 피처 CatBoost 잔차 모델을 학습했다. 테스트 CSV를 읽지 않았고,
다른 테스트 행이나 테스트 전체 분포가 필요하지 않은 행 단위 추론이다.

| 항목 | 결과 |
|---|---:|
| 학습 행 / 시즌 | `1,475,092` / `2019--2024` |
| 피처 / 범주형 | 101 / 15 |
| 투수 스냅샷 | 792명 |
| 2023 opening snapshot → 2024 hierarchy parity | `0.0` |
| 학습 시간 | `792.335초` |
| 모델 SHA-256 | `B3D57DB9D64F7EEAF78AC799BB2C51544774AA05383116468569DC83ABFC5DBE` |
| 스냅샷 SHA-256 | `EC6A3F7A1166B85538112A0C5AABB491929F25E8643BB7248B377B25038E256F` |

첫 v198 내부 빌드에서는 잔차 출력을 확률로 해석한 결함이 패키지 감사에서 발견됐다.
해당 빌드는 즉시 폐기·덮어썼고 Downloads로 복사하거나 제출하지 않았다. runtime을
`hierarchy base + residual`로 수정하고 회귀 테스트를 추가한 뒤 최종 ZIP을 다시 만들었다.

| 항목 | 최종 결과 |
|---|---:|
| ZIP | `artifacts/v198_total_context_stack_package_20260828_01/submit_v198_total_context_stack.zip` |
| 크기 / 파일 수 | `169,594,297 bytes` / 182 |
| SHA-256 | `1EA2B7AE928FD1FF2F725ACC98D444613C21C97E85D0B005B45143538187865E` |
| 챔피언·공식 수식 parity 최대 오차 | `5.55e-17` / `5.55e-17` |
| shuffle·partition·singleton 최대 오차 | `1.67e-16` / `1.67e-16` / `0` |
| 245,789행 실행 시간 | `217.798초` / 제한 600초 |
| 금지된 test 행 집계 연산 | 0건 |
| 출력 유한성·범위 | 통과 |

최종 전체 검증은 `435 passed, 21 skipped`, 저장소 감사 890개 파일 전체 통과다.
Public 제출 전 사전 추정은 중심 `1172.7`, 주 구간 `1172.5--1173.1`, 현실적 넓은
구간 `1171.8--1173.6`으로 기록한다. 이는 2024 OOF `+0.734`와 v180에서 관찰한
OOF→Public 낙관 편차를 함께 반영한 값이며, 1180을 기대한다는 의미는 아니다.

재현 명령:

```powershell
python -m src.champion.v197_finalize_hier_context --help
python -m src.champion.v198_build_total_context_package --help
python -m src.audit_standalone_release `
  --package artifacts/v198_total_context_stack_package_20260828_01/submit_v198_total_context_stack.zip `
  --test-csv data/test.csv --sample-rows 5 --scale-rows 245789 --timeout-seconds 600
python -m pytest -q tests
python scripts/audit_repository.py --include-untracked
```
