# 1185 목표 독립 자산·모델 계보 후속 감사 — 2026-08-17

## 결론

`submit_v27.zip` / Public **1157.9736407889**를 유지한다. 이번 사이클에서
독립 팀 예측 자산, 희소 선형 확률모형, 투수 역할 계층 prior를 순서대로
검증했지만 승격 가능한 후보는 없었다. 새 제출 ZIP을 만들거나 Public 점수를
추가로 조회하지 않았다.

## 외부·팀 자산 감사

Google Drive 공유 루트 `LG Aimers`는 2026-08-15 이후 갱신되지 않았으며
직접 포함하는 폴더는 기존 `v13_과거기준선_1068`, `v17_현재최고점_1093`
두 개뿐이었다. Drive 전역의 `OOF`, `submit_v`, `LG Aimers`, `투구 제구`
교차검색에서 별도 `corrected_cb_oof_20260814_01` 폴더가 검색됐지만, 부모를
추적하면 v13의 `03_모델_재학습/참고용_압축해제본`이었다. 함께 있던
`advanced_domain_residual_20260814_01`, `legacy_cb_diversity_o24_20260814_01`
역시 저장소가 이미 사용하는 같은 계보다. 따라서 독립 오차를 제공하는 팀
OOF/test prediction으로 간주하지 않았다.

로컬 Git 원격에는 기존 `main`, `codex/score-1093-team-handoff`만 보였고,
GitHub connector 계정은 private 원격 `jinwon25/pitch-control-probability`에
권한이 없어 원격 PR·브랜치 추가 감사를 수행하지 못했다. 로컬 remote ref
기준으로는 새 팀 커밋이 없다.

대회 공식 코드 공유 페이지의 공개 링크는 다음 두 운영진 베이스라인뿐이었다.

- [Baseline/Inference RandomForest](https://dacon.io/competitions/official/236743/codeshare/14146)
- [Baseline/Train RandomForest](https://dacon.io/competitions/official/236743/codeshare/14147)

현재 공개된 대회 특화 상위권 해법은 확인되지 않았다.

## v46 — 희소 선형 확률모형

트리·lookup·신경망과 다른 오차면을 만들기 위해 다음 사전 고정 구조를
검증했다.

- 최근 두 시즌만 학습, 이전 시즌 가중치 0.60
- averaged SGD logistic, L2 `alpha=2e-5`, 최대 15 epoch
- 75개 row-local numeric/ASOF 상태
- 선수, 팀, 카운트, 손잡이, domain, 이닝, 주자·점수·LI 및 명시적
  선수×카운트/손잡이/리그 상호작용 one-hot
- category 최소 빈도 25, 13.8k\~14.4k 총 피처
- 2022와 late-2023에서 완전히 같은 route·weight만 선택 가능
- full/late-2024는 recipe 동결 뒤 단 한 번 감사

| 축 | gain vs v27 analogue | 양수 월 비율 | 최악 월 |
|---|---:|---:|---:|
| 2022 선택 방향 | +0.3002 | 75.0% 미만 | -0.6911 |
| late-2023 선택 | +1.0228 | 100.0% | +0.2155 |
| full-2024 감사 | **-0.1915** | 50.0% | -1.2889 |
| late-2024 감사 | +0.0468 | 66.7% | -0.7243 |

동일 recipe consensus gate는 0개였다. fallback `R_CORE / 0.25%`도 full-2024가
음수여서 기각했다. late-2024의 미세 양수만 보고 epoch, 규제, route를 다시
조정하지 않는다.

## v47 — 투수 역할·이닝 계층 prior

명시적 선수×이닝 효과가 기존 저차수 상태모형에서 부족할 가능성을 확인했다.
최근 두 시즌의 label만 사용하여 다음 순서로 empirical-Bayes backoff를 만들었다.

`domain -> pitcher×domain -> pitcher×domain×inning×(상대손/pressure)`

검증한 네 신호는 pitcher-domain, pitcher-inning, pitcher-inning-hand,
pitcher-inning-pressure다. 다른 평가 행, 게임 블록, 현재 경기 집계는 사용하지
않았다.

| 축 | 선택 신호/route/weight | gain | 양수 월 비율 | 최악 월 |
|---|---|---:|---:|---:|
| 2022 | inning-hand / R_CORE / 0.25% | +0.7801 | 75.0% 미만 | -0.1589 |
| late-2023 | 동일 | +0.7719 | 100.0% | +0.3545 |
| full-2024 | 동결 감사 | **-0.1263** | 25.0% | -0.2882 |
| late-2024 | 동결 감사 | **-0.1165** | 0.0% | -0.1186 |

두 선택 기원에서 전체 gain은 양수였지만 월별 consensus gate는 0개였고,
2024의 8·9·10월이 모두 음수였다. 역할 계층의 alpha·세부 context 재탐색은
재시도 금지 목록에 추가한다.

## 누수·규칙 판단

- 두 실험의 모든 모델/lookup은 audit origin 이전 두 시즌에서만 학습했다.
- 당해 시즌 상태는 현재 행의 공식 ASOF 누적값에서 이전 labelled 시즌의 고정
  누적값만 차감했다.
- audit label은 signal/route/weight 선택과 학습에 사용하지 않았다.
- 다른 평가 행의 빈도·순서·예측·선수 상태는 사용하지 않았다.
- train이 게임 순서로 정렬돼 있더라도 평가 행 간 공유가 필요한 현재 게임
  pitch count, game block, 선발/불펜 피로도 복원은 구현하지 않았다.

## 다음 우선순위

현재 로컬 입력 계보의 모델·lookup·보정·TrackMan student는 반복적인 시간 반전을
보였다. 가장 기대값이 높은 다음 입력은 **팀원이 독립적으로 생성한 동일 행 순서의
season-forward OOF와 2025 test prediction**이다. 입수 시 먼저 provenance,
row_id/target 정렬, 학습 cutoff, v27 오차 공분산을 검사한 뒤 고정 저가중치
블렌드만 봉인 감사한다. 그 전까지 제출 우선순위는 v27, 비상 백업은 v25다.

## 재현

```powershell
python -m src.archive.v46_sparse_logit_screen --project . --output-dir artifacts/v46_sparse_logit_20260817_01
python -m src.archive.v47_pitcher_role_hierarchy --project . --output-dir artifacts/v47_pitcher_role_20260817_01
python -m pytest -q
```

모델, OOF, 데이터, 제출 ZIP은 `artifacts/`, `data/`, `submissions/` 아래에만
두며 Git에 포함하지 않는다.
