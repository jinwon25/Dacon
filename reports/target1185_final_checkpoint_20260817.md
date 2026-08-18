# Target 1185 후속 연구 최종 체크포인트 (2026-08-17)

## 결론

- 최종 제출 우선순위 1위는 계속 `submit_v27.zip`이다.
- 확인된 Public 점수는 **1157.9736407889**이며, 1185까지 **27.0263592111**, 1200까지 **42.0263592111**이 남았다.
- 이번 후속 주기에서 v42-v57을 검증했지만, 사전 정의한 시간순 OOF 안정성 게이트를 모두 통과한 새 후보는 없다.
- 따라서 Public 점수 확인을 위한 추가 제출은 만들지 않았다. `submit_v25.zip`(Public 1155.8293405409)은 복구용 2순위로만 유지한다.

## 검증 원칙

- 선택: 2022 및 late-2023만 사용한 exact-recipe consensus.
- 감사: 레시피를 고정한 뒤 full-2024와 late-2024를 확인.
- 금지: 현재 검증 연도 라벨의 학습 사용, 테스트 행 집계/공유, 외부 추정 성공률, Public 점수 기반 계수 선택.
- 최종 반영 기준: 두 선택 원점 합의, full-2024 gain 5 이상, 월별/도메인 안정성, late-2024 재현을 모두 요구.

## 실험 결과 요약

| 실험 | 핵심 아이디어 | full-2024 gain | 판단 |
|---|---|---:|---|
| v42-v44 | 중첩 상태/모드, wave0 독립 기반, TrackMan PFD | 최대 0 이하 또는 선택 실패 | 제외 |
| v45 | TabM-mini 잔차 | -8.271 | 제외 |
| v46-v48 | sparse logistic, 투수 역할, 리그 전이 contrast | -0.192, -0.126, +0.151 | 이득 부족/late 불안정 |
| v49 | 시간순 convex OOF stack | -1.745 | 제외 |
| v50-v52 | low-rank 투수 맥락, 동적 투수 상태, failure-logit 보조축 | +2.119, +0.033, -4.685 | 기준 미달/불안정 |
| v53-v56 | FM 계열 상호작용과 두 원점 공유 구조 | 최대 +1.821 | 기준 미달, 월별 손실 |
| v55 | train-only prototype retrieval | -0.641 | 제외 |
| v57 | 독립 EXP-021 strict 전체 모델 블렌드 | **+1.260** | 합의 통과, 감사 안정성 실패 |

상세한 v49-v55 결과는 `reports/target1185_cycle_checkpoint_20260817.md`, v56 결과는 `reports/target1185_shared_horizon_fm_20260817.md`에 기록돼 있다.

## v57 독립 모델 감사

### 재현 범위

- 출처: <https://github.com/mk-isos/lg-aimers-9-pitch-control>
- 재현 기준 커밋: `5d8b60622f742e21f4416300c826b8162f3cbf35`
- 고정 구성: R-LightGBM/HGB 50:50, 과거 OOF team EB smoothing 1000, 투수×24 count/hand low-rank EB smoothing 300 및 rank 6.
- 2022/2023/2024 OOF의 길이와 타깃 순서를 로컬 `train.csv`에 대해 exact equality로 검증했다.
- 생성된 OOF·모델·외부 복제본은 모두 `artifacts/` 아래에만 두었고 Git에 포함하지 않았다.

### 선택 및 감사

- 탐색 공간: probability/logit blend × ALL/R_CORE/R_ANCHOR/F × 8개 작은 가중치.
- 선택 결과: `probability`, `R_CORE`, weight `0.10`.
- 2022 gain: **+39.674**, worst month **+3.281**.
- late-2023 gain: **+7.195**, worst month **+5.479**.
- exact consensus 통과 레시피: 16/64.
- full-2024 gain: **+1.260**, positive month fraction **0.50**, worst month **-9.176**.
- late-2024 gain: **+0.487**, positive month fraction **0.333**, worst month **-9.064**.
- v27과 challenger의 오차 상관은 축별 **0.9963-0.9994**였다. 독립 구현이지만 실질적인 오차 구조가 너무 유사해 앙상블 이득이 작았다.

결론적으로 v57은 두 선택 원점 합의에는 성공했으나 `full-2024 gain >= 5`, full-2024 월 안정성, late-2024 월 안정성 게이트를 통과하지 못했다. 추론 패키지와 제출 파일을 만들지 않는다.

## 재현과 산출물 정책

```powershell
python -m src.v57_public_strict_blend
python -m pytest -q
```

- v57 상세 결과: `artifacts/v57_public_strict_blend_20260817_01/summary.json`
- OOF 및 SHA-256 목록: 위 artifact의 `summary.json`
- Git 제외 대상: `data/`, `artifacts/`, 모델 파일, OOF 배열, `submissions/`, 제출 ZIP, 인증정보.

## 제출 우선순위

1. `submit_v27.zip` — 현재 유일한 챔피언, Public 1157.9736407889.
2. `submit_v25.zip` — 복구용 백업, Public 1155.8293405409.
3. v42-v57 — 제출 금지. 검증 게이트 미통과.

추후 연구를 재개한다면 v27과 0.996 이상 상관인 잔차 미세조정보다, 다른 정보·목적함수·표현을 쓰는 genuinely orthogonal 기반 모델을 먼저 확보해야 한다.
