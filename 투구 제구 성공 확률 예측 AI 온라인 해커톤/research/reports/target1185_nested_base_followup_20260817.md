# 1185 목표 nested base·독립 예측 자산 후속 연구 — 2026-08-17

## 결론

이번 사이클에서도 `submit_v27.zip`을 대체할 제출 후보는 만들지 않았다. 현재
champion은 Public **1157.9736407889**, 확인 당시 **10위**이며 1185까지
`+27.0263592111`, 1200까지 `+42.0263592111`이 남아 있다.

브랜치·인수인계 문서·로컬 artifact를 감사했지만 다른 팀원이 만든 독립적인
row-level OOF/test 예측은 없었다. 초기 LightGBM/RandomForest `wave0` OOF는
v30 bank에 포함되지 않았으므로 별도 감사했지만, late-2023에서 선택한 레시피가
full-2024에서 반전했다. 이어 v19 state/failure-mode 경로를 각 origin 이전 연도만으로
완전히 다시 선택했다. 이 fully nested 버전은 2023과 2024에서 모두 실패해, 기존
v19의 2023·2024 동시 선택 성능이 개발 상한임을 정량적으로 확인했다.

마지막으로 과거 시즌 TrackMan pitcher arsenal·릴리스 프로필을 privileged
distillation student에 추가했다. 프로필 student도 late-2023에서 양수였지만 official
열만 보는 student보다 약해 사전 규칙에 따라 2024 라벨을 열지 않고 종료했다.

## v42 — fully forward-nested state/mode route

### 검증 설계

- 2023 레시피: 2022만으로 state config, failure-mode signal, 두 multiplier를 선택
- 2024 레시피: 2022·2023의 최소 gain을 최대화해 선택
- 각 감사 연도는 레시피가 고정된 뒤 한 번만 평가
- 2022 failure-mode OOF는 `latent_failure_mode_state_20260817_02`, 이후 연도는
  기존 strict season-forward cache 사용
- domain별 월 안정성 및 pitcher/batter/crossed bootstrap 동시 기록

### 결과

| 감사 연도 | 선택 연도 | gain vs incumbent | 양수 월 비율 | 최악 domain | 판정 |
|---:|---|---:|---:|---:|---|
| 2023 | 2022 | **-172.2176** | 0.0% | -767.5869 | 기각 |
| 2024 | 2022·2023 | **-5.3946** | 62.5% | -90.8161 | 기각 |

2023은 모든 월과 모든 domain에서 악화했다. 2024는 R_CORE `+7.8806`, F
`+42.9166`이었지만 R_ANCHOR `-90.8161`이 전체 개선을 상쇄했다. pitcher,
batter, pitcher×batter bootstrap의 2024 개선확률도 `0.308~0.3365`에 불과했다.

기존 v19 개발 경로는 2023·2024를 동시에 선택에 넣었을 때 2024 `+46.3960`,
모든 월 양수를 기록했다. fully nested 결과와의 차이는 이 수치를 독립적인 미래
성능 추정치가 아니라 개발 상한으로 취급해야 한다는 직접 근거다.

## v43 — 남은 독립 base 방향 감사

### 자산 감사

- Git branch에는 코드·문서만 있고 OOF/test prediction은 추적되지 않았다.
- Drive 인수인계 문서의 v13/v17 asset은 현재 champion의 선조 계보이므로 독립
  모델이 아니다.
- 로컬 `wave0`는 2020–2024 strict prior-season LightGBM/RF OOF이고 v30의 106개
  signal bank에는 포함되지 않았다.

### late-2023 선택 후 frozen 2024 감사

| family | 2023 선택 레시피 | 선택 gain | full-2024 | late-2024 | 판정 |
|---|---|---:|---:|---:|---|
| wave0 | `lgb_raw`, ALL, 10% | +81.8575 | **-7.3113** | +3.4915 | 기각 |
| nested gap | `nested-v19`, ALL, -20% | +77.9596 | +0.0693 | **-3.4455** | 기각 |

`wave0`는 2024 8–10월에는 양수였지만 full-2024의 3–7월 중 다수에서 반전했고
R_CORE도 `-9.9274`였다. 후반기만 보고 경로를 다시 고르는 것은 외부 감사 라벨의
사후 사용이므로 하지 않았다. nested gap은 full-2024 평균이 거의 0이었고
late-2024의 세 달이 모두 음수였다.

## v44 — prior-only TrackMan profile augmented PFD

### 가설과 누수 통제

기존 v38 teacher는 현재 투구의 구종·구속·회전·무브먼트·릴리스 정보를 사용했지만
student는 official inference-safe 열만 보았다. v44는 student에만 다음 origin 이전
시즌의 target-free pitcher profile 66개를 추가했다.

- linkage 신뢰도와 pitcher별 표본수
- 구속·회전·무브먼트·릴리스 평균·표준편차
- 구종별 arsenal 평균과 mix
- 최신 시즌 물리값과 장기 평균의 차이
- origin `S`의 profile에는 TrackMan season `< S`만 포함
- pitcher/batter/team ID는 두 student 모두 제거
- 2022 aligned teacher/student를 late-2023에 적용해 recipe 선택
- profile student가 선택에서 이길 때만 2024를 열도록 코드에서 강제

### 결과

| student | 최선 경로 | late-2023 gain | 최악 월 | 최소 domain gain |
|---|---|---:|---:|---:|
| official without IDs | ALL, 40% | **+12.2269** | +9.1202 | +8.2168 |
| profile without IDs | ALL, 40% | +10.4848 | +7.3053 | +8.1281 |

두 student 모두 세 달과 세 domain에서 양수였지만 profile student가 official
student보다 `1.7421` 낮았다. 물리 profile의 추가 설명력이 선택 자료에서 확인되지
않았으므로 2024 outer audit과 패키징을 실행하지 않았다. 현재 투구 TrackMan은
teacher에서만 쓰였고 audit/test student에는 들어가지 않는다.

## 재시도 금지 목록 추가

1. v19 route를 2022 하나 또는 2022·2023 maximin으로 다시 선택하는 방식
2. `wave0` LightGBM/RF를 v27에 단순 확률 혼합하는 방식
3. nested state/mode route와 기존 v19 차이를 양·음의 작은 가중치로 보정하는 방식
4. 66개 prior-only TrackMan profile을 PFD student에 일괄 추가하는 방식
5. 2024 후반 결과를 보고 위 family의 domain·월·가중치를 다시 고르는 방식

## 현재 판단과 다음 우선순위

현재 저장소의 exact-ASOF, residual lookup, state/mode, calibration, direct spline,
TrackMan teacher/student, 초기 tree base는 실질적으로 포화됐다. 같은 계보의 미세
라우팅으로 1185를 기대할 근거는 없다. 다음 유효 작업은 다음 둘 중 하나다.

1. 실제로 독립적으로 학습된 팀원 모델의 동일 행 OOF/test prediction을 확보해
   v27과 오차 공분산을 검증한다.
2. 새로운 표현 구조를 쓰되 2022→2023 선택, 2023→2024 감사가 가능한 저용량
   tabular ensemble/retrieval 모델을 만든다. 단순 MLP, CatBoost, XGBoost 재실행은
   이미 실패했으므로 동일 구조는 제외한다.

새 제출 우선순위는 계속 `submit_v27.zip` 단독 1순위, `submit_v25.zip` 백업이다.
이번 사이클 후보는 모두 OOF/연도 안정성 gate에서 탈락했으므로 ZIP 생성이나 Public
제출을 하지 않았다.

## 재현 명령

```powershell
python -m src.archive.v42_forward_nested_state_mode --project .
python -m src.archive.v43_independent_base_audit --project .
python -m src.archive.v44_profile_augmented_pfd --project .
python -m pytest -q
```

모델·OOF·CSV prediction·제출 ZIP은 모두 `artifacts/` 또는 `submissions/` 아래에만
있으며 Git 커밋 대상이 아니다.
