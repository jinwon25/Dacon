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
