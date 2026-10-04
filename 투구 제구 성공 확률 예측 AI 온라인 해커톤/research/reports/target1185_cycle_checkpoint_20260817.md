# 1185 목표 후속 사이클 체크포인트 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../../notebooks/v26_champion_reproduction.ipynb`](../../notebooks/v26_champion_reproduction.ipynb).

## 운영 결론

이번 후속 사이클에서 `submit_v27.zip`을 대체할 제출 후보는 만들지 않았다. 현재 확인된
Public champion은 다음과 같다.

| 항목 | 값 |
|---|---|
| 파일 | `submit_v27.zip` |
| Public | `1157.9736407889` |
| 마지막 확인 순위 | 10위 |
| 제출 ID | `1535195` |
| SHA-256 | `CEFC91F4F8EBBA32EE8025816319F32C95023CB59EA70741176A98FB7A172385` |
| 245,789행 runtime | 56.882초 |
| peak RSS | 약 1.4GB |
| 배치 불변성 최대 오차 | `1.67e-16` |

1185까지 `+27.0263592111`, 1200까지 `+42.0263592111`이 남았다. 이번 사이클의
어떤 신규 후보도 이 격차에 근접하면서 시간 안정성 gate를 통과하지 못했다. Public
점수 확인용 제출을 추가하지 않았고 v27 가중치를 사후 조정하지 않았다.

## 이번 사이클 핵심 결과

| 버전 | 신규 가설 | full-2024 | late-2024 | consensus gate | 판정 |
|---|---|---:|---:|---:|---|
| v49 | 48개 OOF temporal convex stack | -1.7446 | -1.9833 | 통과 후 외부 반전 | 기각 |
| v50 | pitcher×count×hand 저랭크 SVD | +2.1194 | +0.1128 | 2-origin 통과 | 개선폭·월 안정성 미달 |
| v51 | 현재 시즌 표본+AR(1) pitcher state | +0.0329 | -0.2273 | 0개 | 기각 |
| v52 | MR/way-off 보조 중심화 로짓 | -4.6850 | -0.3379 | 0개 | 기각 |
| v53 | 저랭크 야구 interaction FM | +1.8216 | +5.5581 | 0개 | 월 변동성 실패 |
| v54 | 두-source FM 평균/부호합의 | +0.8611 | +2.1233 | 0개 | 안정화됐지만 폭 미달 |
| v55 | train-only prototype retrieval | -0.6406 | +0.1103 | 0개 | 선택 단계부터 기각 |

v50과 v53/v54는 일부 평균 양수 신호를 만들었지만 승격하지 않았다.

- v50은 full-2024에서 8개 월 중 3개만 양수였고 최악 월이 `-10.9248`이었다.
- v53은 full-2024 월 승률 50%, late-2024 33%였다.
- v54는 변동을 줄였지만 full-2024 gain `+0.8611`과 active 월 승률 71.4%에 그쳤다.

작은 양수 평균이나 후기 구간만 보고 route·가중치를 다시 고르면 반복 사용한 2024에
과적합되므로 제출 후보로 전환하지 않는다.

## 공개 구현 감사가 미친 영향

공개 코드는 구조적 아이디어만 참고했고 데이터·계수·외부 성공률은 가져오지 않았다.

- [danny010712/LG-Aimers-9th-Hackathon](https://github.com/danny010712/LG-Aimers-9th-Hackathon):
  MR/way-off 보조 로짓을 v52로 prior-OOF 재구성했다. 공개 계수 대신 로컬 과거 OOF
  계수를 사용했지만 계수 부호가 시즌 간 반전해 실패했다. 외부 2025 성공률 기반 global
  shift는 규칙·재현성 위험 때문에 적용하지 않았다.
- [mk-isos/lg-aimers-9-pitch-control](https://github.com/mk-isos/lg-aimers-9-pitch-control):
  저랭크 투수 context와 dynamic pitcher state를 각각 v50/v51로 독립 재구성했다.
  v50은 작게 양수였지만 안정성 기준을 넘지 못했고 v51 효과는 거의 0이었다.
- [IROHA0508/LG_Aimers_Hackathon](https://github.com/IROHA0508/LG_Aimers_Hackathon):
  임베딩 MLP/등장 엔티티 통계 아이디어를 검토했으나 random stratified CV와 배포 시점
  feature 생성이 일치하지 않아 보고된 로컬 점수를 시간 전이 근거로 사용하지 않았다.
  저장소의 strict temporal embedding MLP와 TabM 감사도 2024에서 반전했다.

추가 공개 검색에서 이 세 계열과 기존 베이스라인 외에 검증 가능한 독립 row-level
OOF/test 예측은 확인하지 못했다.

## 검증·재현 상태

- 전체 테스트: `196 passed`
- repository audit: 480개 파일, 비밀정보·금지 경로·5MB 초과 추적 후보 0개
- 모든 신규 평가 신호: row-local, 평가행 간 공유 없음
- 선택: 2022 + late-2023 exact recipe
- 외부 진단: full-2024 + late-2024
- 신규 모델·OOF·NPZ·CSV prediction: `artifacts/`에만 저장, Git 미포함
- 제출 ZIP: Git 미포함
- 인증정보: 저장소에 없음; v27 dry-run에서 `credentials_present=false`

이번 사이클 검증 커밋은 `08b0fc0`부터 `30ab381`까지 15개 연구 단위로 분리했다.
사용자의 기존 dirty worktree와 데이터·모델 artifact는 스테이징하지 않았다.

## 제출 우선순위와 실행

1. **1순위: `submit_v27.zip`** — 이미 제출된 champion이므로 동일 파일을 다시 제출하지
   않는다.
2. **비상 백업: `submit_v25.zip`** — Public `1155.8293405409`; v27 실행 문제가 생길
   때만 사용한다.
3. v50/v53/v54는 연구 신호일 뿐 제출 후보가 아니다.

로컬 ZIP 구조·해시 점검:

```powershell
python -m src.submit_dacon submit_v27.zip --memo "champion v27"
```

정말 재제출해야 할 외부 운영 사유가 생긴 경우에만 Git에 포함되지 않는 환경변수 또는
`.env`에 `DACON_API_TOKEN`, `DACON_TEAM_NAME`을 둔 뒤 다음처럼 실행한다.

```powershell
python -m src.submit_dacon submit_v27.zip --memo "champion v27" --execute
```

## 다음 재개 조건

현재 저장소의 ASOF lookup, state/mode, TrackMan student, calibration, tree/linear/deep
direct model, FM, prototype retrieval은 모두 시간 전이 감사가 끝났다. 같은 2024를 보고
미세조정하는 작업은 중단한다. 기대값이 높은 재개 조건은 다음 둘뿐이다.

1. 별도 팀원이 독립 생성한 동일 row 순서의 season-forward OOF와 2025 test prediction
2. 새로운 독립 라벨 연도 또는 대회가 허용한 새로운 pre-pitch 관측 정보

새 자산을 받으면 provenance·row/target 정렬·학습 cutoff를 먼저 감사하고, v27과 오차
공분산이 충분히 낮을 때만 사전 고정 저가중치 blend를 봉인 검증한다.
