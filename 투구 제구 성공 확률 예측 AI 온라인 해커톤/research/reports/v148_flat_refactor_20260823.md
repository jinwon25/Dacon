# v148 단일 세대 평탄화 리팩터링

기록 시각: 2026-08-23 KST

## 결론

`submit_v148.zip`의 5세대 중첩 구조를 단일 세대 평탄 구조로 리팩터링했고,
**예측 동등성을 실측으로 확인**했다. 117,434행 비교에서 최대 절대차는
`3.33e-16`이고, 이는 **원본 패키지를 두 번 실행했을 때의 차이와 정확히 같은
값**이다. 즉 남은 차이는 리팩터링이 만든 것이 아니라 원본이 원래 가지고 있던
실행 간 부동소수점 비결정성(LightGBM/CatBoost 스레드 리덕션)이다.

**공식 전달본은 여전히 원본 `submit_v148.zip`이다.** 평탄화본은 동등성이
검증된 유지보수용 리팩터링본이며 새 챔피언이 아니다. 리더보드 점수
1170.3014697177은 원본으로 획득한 기록이고, 평탄화본으로 재제출해 점수를
재확인한 적은 없다.

## 구조 before / after

원본은 자족 패키지이지만 계보가 디렉터리 깊이로 박제돼 있었다.

```text
[before]  최대 경로 깊이 6
script.py                                                  138줄  v148 진입점
 ├─ model/c3_sign_all.joblib
 ├─ model/h1/script.py                                     575줄  H1 (자체 main 보유)
 └─ model/v124/script.py                                   228줄  v124 (자체 main 보유)
     ├─ model/v124/model/components/v104_features.py       226줄
     ├─ model/v124/model/components/parent_script.py       182줄  v84 (자체 main 보유)
     │   └─ .../parent/components/parent_script.py         161줄  v82 (자체 main 보유)
     │       ├─ .../parent/parent/components/champion_script.py  2,175줄 (자체 main)
     │       └─ .../parent/parent/components/strict_script.py      873줄 (자체 main)
     ├─ model/v124/model/{conditional,r_fm}/
     ├─ model/v124/model/parent/v56_fm/
     └─ model/v124/model/parent/parent/{champion,strict}/
```

```text
[after]   최대 경로 깊이 3
script.py            유일한 진입점 (main 1개)
requirements.txt     1벌
model/
  lib/
    __init__.py  paths.py
    base_ensemble.py  strict_asof.py  strict_overlay.py
    futures_fm_overlay.py  conditional_overlay.py
    row_local_features.py  form_context_rf.py
  ARCHITECTURE.md    레이어 파이프라인과 최종 혼합식
  base_ensemble/  strict_asof/  conditional/  regular_fm/{older,recent}/
  futures_fm/  form_context_rf/rf.pkl  c3_sign_all.joblib
```

| 항목 | before | after |
|---|---:|---:|
| 최대 경로 깊이 | 6 | **3** |
| 파일 수 | 89 | 90 (`ARCHITECTURE.md` 추가) |
| `main()` 진입점 | 6 | **1** |
| 동적 로딩(`spec_from_file_location`) | 4곳 | **0** |
| `requirements.txt` | 3벌 | **1벌** |
| 모듈·디렉터리의 계보/상태 이름 | 12개 | **0** |
| 패키지 크기 | 46,352,820 B | 46,687,840 B |

4,558줄을 한 파일로 합치지는 않았다 — 가독성이 목적이기 때문이다.

### 공식 제출 구조 제약과 `model/lib/`

대회 공식 규정은 제출 ZIP **최상위에 `model/`, `script.py`, `requirements.txt`
세 가지만** 허용한다. "추가 최상위 폴더가 zip 구조 내 존재하는 경우 등 구조가
불일치하는 경우 설치 오류가 발생합니다."

**원본이 6단계로 중첩됐던 이유가 바로 이 제약이다.** 최상위에 셋만 둘 수 있으니
컴포넌트 코드를 전부 `model/` 안으로 밀어 넣어야 했고, 세대가 쌓일수록 깊이가
늘어난 것이다. 즉 중첩은 게으름이 아니라 제약에 대한 대응이었다.

첫 평탄화본은 이 규정을 모르고 `lib/`와 `ARCHITECTURE.md`를 최상위에 두어
**설치 단계에서 거부됐을 구조**였다. 현재 빌드는 컴포넌트 패키지를 `model/lib/`로
옮겨 최상위를 계약대로 맞췄다. 평탄 구조(깊이 3), 동적 로딩 0개, 기능명은 그대로다.

`script.py`는 컴포넌트를 import하기 **전에** `model/`을 `sys.path`에 넣는다.
경로는 `Path(__file__).resolve().parent` 기준이며 작업 디렉터리를 쓰지 않는다 —
평가 서버가 어느 디렉터리에서 실행할지 보장되지 않기 때문이다. 모듈 이름은
`lib.*`로 유지되므로 컴포넌트 내부 import는 한 줄도 바뀌지 않았다.

빌드 스크립트는 ZIP을 쓰기 직전에 최상위 엔트리 집합을 `ROOT_CONTRACT`와
대조하고 불일치 시 빌드를 중단한다. `tests/test_v148_flat_parity.py`는 같은 검사를
평탄화본과 **원본(대조군)** 양쪽에 걸어 회귀를 막는다.

## 이름 정책

이름은 **기능 또는 도메인**을 가리킨다. 계보(`v82`/`v84`/`v104`/`v124`),
상태(`champion`/`legacy`), 리더보드 목표(`target1160`)는 이름에 쓰지 않는다.
`champion`은 챔피언이 바뀌는 순간 거짓이 되는 이름이라 함께 제거했다.
부수 규칙으로 **아티팩트 디렉터리는 그것을 로드하는 모듈 이름**을 따른다.

| 모듈 | 역할 | 가중치 |
|---|---|---|
| `model/lib/base_ensemble.py` | LGB+RF+CatBoost+joint state-mode 기반 앙상블 | `model/base_ensemble/` |
| `model/lib/strict_asof.py` | exact as-of 규율 모델 (group/team/lowrank) | `model/strict_asof/` |
| `model/lib/strict_overlay.py` | strict를 R_CORE에 10% 오버레이 | — |
| `model/lib/futures_fm_overlay.py` | F(퓨처스) 도메인에 FM 보정 eta 0.10 | `model/futures_fm/` |
| `model/lib/conditional_overlay.py` | regular FM + 조건부 보정, stability gate | `model/regular_fm/`, `model/conditional/` |
| `model/lib/row_local_features.py` | 행 독립 피처 생성 | — |
| `model/lib/form_context_rf.py` | 최근 폼·카운트 컨텍스트·platoon RF | `model/form_context_rf/` |

`R`/`R_CORE`/`R_ANCHOR`/`F`는 공식 도메인 코드(1군 정규시즌 / 퓨처스)이고
`h05`/`l15`/`b075`는 하이퍼파라미터 인코딩이므로 그대로 두었다. 계보가 아니다.

### 가중치 파일명을 바꾸지 않은 이유

당초 계획은 `model/base_ensemble/` 안의 가중치 16개(`v14_*`, `v20_*`,
`legacy_cb_*` 등)도 기능명으로 바꾸는 것이었다. **실행하지 않았다.** 조사 결과
그 파일명들이 파이썬 문자열 리터럴에만 있는 게 아니라 **모델 아티팩트 자신**에
값으로 들어 있었다.

| 참조하는 아티팩트 | 참조된 파일명 |
|---|---|
| `hybrid.json` | `v14_refinement_spec.json`, `v16_residual_spec.json`, `v20_target1160_spec.json`, `v21_context_state_eb_spec.json`, `v22_low_variance_spec.json`, `v25_postbreak_anchor_spec.json` |
| `v20_target1160_spec.json` | `v20_pfd_control.txt`, `v20_pfd_soft_l050.txt` |
| `v25_postbreak_anchor_spec.json` | `v25_postbreak_anchor.joblib` |

즉 파일명을 바꾸려면 이 세 아티팩트를 편집해야 하고, 그러면 **78개 전부
바이트 동일**이라는 보증이 깨진다. 그 보증은 "가중치는 1비트도 건드리지
않았다"를 증명하는 가장 강한 안전장치이므로, 덜 보이는 곳의 이름 정리와
맞바꾸지 않았다. `v20_pfd_control.txt`와 `v25_postbreak_anchor.joblib`은
파이썬 코드에서 아예 참조되지 않고 JSON을 통해서만 로드되므로, 이름 변경이
사실상 순수한 아티팩트 편집 작업이라는 점도 판단 근거였다.

내부 함수명 `apply_fixed_v56`과 상수 `H1_WEIGHT`도 같은 이유로 두었다. 전자는
외부에 드러나지 않고, 후자는 값 보존이 명시적으로 요구된 상수다.

### 핵심 메커니즘 치환

원본은 각 계층이 자식을 `importlib`으로 새로 실행한 뒤 그 모듈의 전역
`MODEL_DIR`을 밖에서 덮어써서 아티팩트 경로를 주입했다.

```python
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.MODEL_DIR = MODEL_DIR / "parent"     # 경로 주입
```

평탄화본은 `model/lib/paths.py`의 `MODEL_ROOT` 하나를 각 모듈이 직접 참조한다.

```python
MODEL_DIR = MODEL_ROOT / "base_ensemble"     # 정적, 주입 없음
```

정적 import로 바꾸면 모듈이 1회만 실행되고 캐시된다(원본은 매 호출 재실행).
안전성을 사전 확인했다: 7개 컴포넌트 전부 `global` 문, 모듈 수준 가변 컨테이너,
`lru_cache`가 없어 예측에 영향을 주는 모듈 상태가 존재하지 않는다.

## 동등성 검증

`src/champion/v148_flat_parity_audit.py`. 두 패키지는 컴포넌트 모듈명이 겹쳐 한 인터프리터에
공존할 수 없으므로 **각각 별도 서브프로세스**에서 실행하고 확률 벡터를 비교했다.

| 항목 | 값 |
|---|---:|
| 비교 행 수 | **117,434** |
| **최대 절대차** | **3.3306690738754696e-16** |
| 평균 절대차 | 3.29e-18 |
| 완전 일치(비트 동일) 비율 | 97.19% |
| 합격 기준 | ≤ 1e-12 |
| **원본 vs 원본 (재실행) 최대차** | **3.3306690738754696e-16** |
| 원본 예측 평균 | 0.516231886056267 |
| 평탄화본 예측 평균 | 0.516231886056267 |

마지막 두 줄이 이 검증의 핵심이다. 원본을 두 번 돌린 차이와 원본↔평탄화본의
차이가 **같은 값**이므로, 평탄화가 기여한 수치 차이는 0이다.

도메인·cold-start 분포:

| 구간 | 행 수 | 최대 절대차 |
|---|---:|---:|
| R_CORE | 80,795 | 3.33e-16 |
| R_ANCHOR | 21,897 | 2.22e-16 |
| F | 14,742 | **0.0** |
| cold-start(데뷔 선수) | 2,000 | 2.22e-16 |

원본 시즌 2019–2024 전체와 3–10월 8개 월에서 표본을 뽑았다.

### 검증 프레임 구성 근거

동결 패키지는 2025 평가행만 받는다. `strict`가 `season > 2024`를 강제하고,
챔피언의 TrackMan profile 테이블은 `season=2025`로만 키가 잡혀 있다. 또
`asof_*_n` 세 카운터가 각 선수의 2024년 말 누적치 이상이어야 하고, 재구성된
시즌 내 성공 수가 `[0, 시즌 내 투구 수]` 안에 있어야 한다. 실제 2025 행은 이를
자동으로 만족하지만 2019–2024 학습행은 만족하지 못한다.

따라서 카운터를 `동결 누적 + 시즌 내 진행분`으로 재구성해 계약을 충족시키고,
진행분과 비율은 원본 행에서 유도해 행별 다양성을 보존했다. 두 패키지에 **동일한
프레임**을 넣으므로 이 재구성은 동등성 비교에 영향을 주지 않는다.

cold-start는 train에서 뽑을 수 없다(모든 선수가 이미 동결 테이블에 있다).
2025 데뷔 선수를 모사해 미등록 ID와 0 카운터를 부여했고, 이로써 fillna/전역
비율 fallback 경로를 실제로 통과시켰다.

## 행 독립성과 런타임

| 검사 | 원본 기록 | 평탄화본 실측 |
|---|---:|---:|
| shuffle 불변성 | 1.11e-16 | **3.33e-16** |
| partition 불변성 | 0.0 | **3.33e-16** |

두 값 모두 위에서 확인한 부동소수점 잡음 바닥(3.33e-16)과 같은 수준이다.

245,789행 런타임은 **같은 세션에서 연속 측정**했다(이 머신은 부하에 따라 최대
2.76배까지 느려지므로 절대 시간 단독 비교는 무의미하다).

| 패키지 | 245,789행 | 비고 |
|---|---:|---|
| 원본 | 105.624초 | |
| **평탄화본** | **105.761초** | 원본 대비 **1.001배** |
| 같은 프레임 최대 절대차 | 4.44e-16 | 전체 규모에서도 동등 |

구조 변경에 따른 런타임 비용은 없다(비율 1.001). 절대 시간은 측정 시점의 머신
부하에 좌우되므로 비율만 의미가 있다 — 최상위 구조 수정 전 측정에서는 같은
방식으로 0.938배가 나왔다. 공식 제한 600초를 크게 밑돈다.

## 모델 가중치 무결성

78개 모델 아티팩트를 **바이트 그대로** 복사하고 원본과 SHA-256을 1:1 대조했다.
불일치 **0건**. 재학습·변환·형식 변경 없음. 예측 수식과 상수
(`H1_WEIGHT=0.15`, `C3_WEIGHT=0.5`, `MEAN_RECENT_WEIGHT=0.15`,
`ANCHOR_TEAM=13`, clip `0.001/0.999`)도 그대로다.

## 산출물

| 파일 | 값 |
|---|---|
| 패키지 | `artifacts/v148_flat_20260823_01/submit_v148_flat.zip` |
| SHA-256 | `83CA670CF68F329B3CC1BAC2B452E5FB4D2F3D0BE39BD49BA71526E9FDC1C8E8` |
| 크기 | 46,687,840 bytes (90 files) |
| 빌드 | `src/champion/v148_flat_build_package.py` |
| 런타임 | `src/champion/v148_flat_runtime_script.py` |
| 동등성 감사 | `src/champion/v148_flat_parity_audit.py` |
| 테스트 | `tests/test_v148_flat_parity.py` (16개) |

전체 테스트: **384 passed, 4 skipped** (기준선 368 + 신규 16).

## 리팩터링 중 발견한 사항

1. **두 `parent_script.py`는 서로 다른 계층이다.** 파일명이 같아 혼동하기 쉽지만
   `model/v124/model/components/`(6,443 B)는 docstring상 **v84**
   (`V84_FIXED_V56_F_ROUTE`, F 라우트에 고정 v56 FM 적용)이고,
   `model/v124/model/parent/components/`(6,234 B)는 **v82**
   (champion + 10% EXP-021 strict on R_CORE)다. 각각
   `model/lib/futures_fm_overlay.py`, `model/lib/strict_overlay.py`로 분리했다.
   상위 v124 스크립트가 전자를
   `"v104_parent_component"`라는 이름으로 로드하고 있어 이름과 실제 계층이
   어긋나 있었다.

2. **`requirements.txt` 3벌은 합집합을 만들면 안 된다.** 세 파일이 서로 충돌한다.

   | 패키지 | 루트 (유효) | `model/h1/` (사문) |
   |---|---|---|
   | scikit-learn | 1.6.1 | 1.7.2 |
   | joblib | 1.5.1 | 1.5.3 |
   | pandas | 2.2.3 | 2.3.3 |
   | catboost | 1.2.8 | 1.2.10 |

   ZIP 루트 파일만 평가 서버가 사용하며, 그것이 1170.3014697177을 만든 실제
   환경이다. 중첩본은 h1/v124가 독립 패키지였던 시절의 잔재로 런타임에 아무
   영향이 없다. 합집합을 만들면 검증된 실행 환경을 바꾸게 되므로 **루트 파일을
   그대로 유지**했다.

3. **참조되지만 존재하지 않는 아티팩트가 있다.** `strict_script.py`의 `main()`은
   `exact_pitchtype_control.json`, `lowrank_rspecific_effects.json`,
   `lowrank_recency_effects.json`, `trackman_physical_ridge.json`을 읽지만 이
   파일들은 패키지에 없다. v82는 `main()`을 호출하지 않고 개별 함수만 쓰므로
   원본에서도 실행되지 않는 죽은 경로였다. `main()` 제거로 함께 사라졌다.
   `champion_script.py`의 `game_type_offsets.json`은 `.exists()` 가드가 있어
   부재 시 무보정으로 통과한다(원본과 동일 동작).

4. **사용되지 않는 3.1MB 아티팩트가 있다.** `strict/pitcher_count_effects.json`은
   `main()`에서만 참조된다. 삭제하면 예측은 안 바뀌지만 "개선하지 않는다" 원칙에
   따라 **그대로 보존**했다. 향후 용량이 문제될 때만 별도 검토 대상이다.

5. **줄바꿈이 혼재한다.** champion/strict/h1은 CRLF, v124/v84/v82는 LF였다.
   평탄화본은 LF로 통일했다. Python 소스의 줄바꿈은 예측에 영향이 없다.

6. **빌드는 앵커가 어긋나면 즉시 실패한다.** 모든 소스 변환은 정확한 문자열
   앵커에 `count == 1` 단언을 걸었다. 실제로 첫 빌드가 CRLF 때문에
   `champion:paths: expected exactly 1 anchor, found 0`으로 멈췄고, 조용히
   잘못된 패키지를 만드는 대신 원인을 드러냈다.

## 규칙 준수

- 모델 재학습 없음, 예측 수식·상수 변경 없음.
- 원본 `artifacts/v148_v142_v138_blend_package_20260823_01/` 무수정
  (SHA-256 재확인 테스트 포함).
- 2025 결과 라벨·외부 데이터·test 집계/분포/순서 미사용.
- 행 독립 추론(shuffle/partition 검증 완료).
- commit·제출 없음.
