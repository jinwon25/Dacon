# 투구 제구 성공 확률 예측 AI 해커톤

LG Aimers 9기 DACON 해커톤의 팀 공용 연구 저장소입니다. 코드, 검증 절차, 의사결정 보고서만 Git으로 공유하며 DACON 원본 데이터·학습 모델·OOF·제출 ZIP은 각 팀원의 비공개 로컬 artifact로 관리합니다.

## 현재 상태 — 2026-08-16

| 구분 | 상태 |
|---|---|
| 공식 champion | `submit_v21.zip` |
| v21 Public | **1151.5138356157** |
| 확인 당시 순위 | **12위** |
| v21 제출 ID | `1534213` |
| 현재 목표 | **Top 10 / Public 1160 근접** |
| v20 대비 Public gain | **+0.0414068967** |
| 목표 1160까지 | **8.4861643843** |
| Top 10까지 | **6.4456139434** |
| 오늘 남은 제출 | **0회** |
| v21 순방향 최소 gain | **+10.7720** vs v20 |
| v21 패키지 검증 | 12개 게이트 전부 통과; 원본 77개 통과, clean checkout 73개 통과·4개 skip |

v21은 API 성공과 공식 리더보드 반영을 확인한 새 champion입니다. v20보다 `+0.0414` 개선됐지만 로컬 `+10.7720`의 대부분은 2025 평가 분포로 전이되지 않았습니다. 같은 2024 잔차 lookup 계열의 Public 사후 가중치 조정은 하지 않습니다.

핵심 판단 근거는 [Target 1200 연구 보고서](reports/target1200_research_20260816.md), 공식 결과는 [v21 Public 결과](reports/v21_public_result_20260816.md), 규정 검토는 [공식 준수 감사](reports/dacon_official_compliance_audit_20260816.md)에서 확인하세요.

## 처음 시작하기

```powershell
git clone https://github.com/Lg-Aimers-chungang/hackathon.git
cd hackathon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
$env:PYTHONPATH=(Resolve-Path '.').Path
pytest -q
```

각 팀원이 DACON에서 직접 받은 파일을 아래 위치에 둡니다.

```text
data/train.csv
data/trackman_history.csv
data/test.csv
data/sample_submission.csv
```

원본 데이터는 커밋하지 않습니다. 상세 계약은 [data/README.md](data/README.md)와 [data_description.md](data_description.md)를 참고하세요.

## 폴더 안내

```text
hackathon/
├─ src/          학습·스크리닝·패키징·검증 구현
├─ tests/        누수 방지와 추론 회귀 테스트
├─ reports/      공식 제출 결과와 모델 의사결정 근거
├─ docs/         협업·방법론 문서
├─ configs/      재현 가능한 설정
├─ notebooks/    탐색용 워크벤치
├─ data/         로컬 전용 DACON 데이터; README만 Git 추적
├─ artifacts/    로컬 전용 모델·OOF; README만 Git 추적
└─ submissions/  로컬 전용 제출 ZIP 이력; 문서만 Git 추적
```

보고서 탐색 순서는 [reports/README.md](reports/README.md)를 따르세요.

## v21 재현 순서

아래 과정은 v19 OOF, v20 부모 ZIP과 기존 artifact가 로컬에 준비되어 있다는 전제입니다. 필요한 경로와 판단 근거는 [Target 1200 연구 보고서](reports/target1200_research_20260816.md)에 기록돼 있습니다.

```powershell
# 1. 2024 v19 OOF 잔차에서 동결 recency EB artifact 학습
python -m src.train_v21_context_state_eb --project .

# 2. v20을 부모로 v21 ZIP 생성
python -m src.package_v21_context_state_eb `
  --project . `
  --parent submissions/history/submit_v20.zip `
  --output submit_v21.zip

# 3. 실제 ZIP 계보·런타임·배치 독립성 검증
python -m src.validate_v21_context_state_eb `
  --project . `
  --candidate submit_v21.zip `
  --parent submissions/history/submit_v20.zip
```

## v21 설계 요약

v21은 v20 위에 여섯 개의 동결 recency empirical-Bayes lookup만 더합니다.

1. R_CORE 투수 × 타자 손 × 최근 3경기 제구 구간
2. R_CORE 투수 × 타자 손 × 최근 5경기 제구 구간
3. 투수 × 타자 손 × 주자 상황
4. 카운트 × 양손 × 이닝 구간
5. 카운트 × 양손 × 최근 3경기 제구 구간
6. 카운트 × 양손 × reverse rate 구간

추론 시에는 TrackMan 현재 투구 물리값이나 test 전체 통계를 사용하지 않습니다. 모든 correction은 2024 `target - v19 OOF`에서 미리 동결되고 각 test 행에 독립적으로 적용됩니다.

## 검증 원칙

- 시간 순서를 지킨 2023→2024 및 시즌 전반→후반 검증
- v20과 paired Brier 차이 비교
- 월·도메인 분해
- 투수, 타자, 투수×타자 5,000회 군집 부트스트랩
- Public 점수로 가중치를 역튜닝하지 않음
- test 행 간 집계·정렬·전후 참조 금지
- ZIP 계보, SHA-256, 120초 런타임, 4GB 메모리, 배치 불변성 검증

## Git 협업 규칙

- `main`에 직접 push하지 않습니다.
- 개인 브랜치에서 작업하고 PR로 리뷰받습니다.
- 예: `jinwon25/target-1160-research`
- 데이터, 모델, OOF, 제출 ZIP, 인증정보는 커밋하지 않습니다.
- 실험 결과는 재현 명령·부모 모델·시간축·기각 사유와 함께 보고서에 남깁니다.

자세한 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 따르세요.

## 주요 참고 자료

- [DACON 공식 평가](https://dacon.io/competitions/official/236743/overview/evaluation)
- [DACON 공식 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [Generalized distillation](https://arxiv.org/abs/1511.03643)
- [Model-selection overfitting](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Pigeonhole bootstrap](https://doi.org/10.1214/07-AOAS122)
