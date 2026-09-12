# Private Score 재현용 학습 코드 — v345

본 패키지는 v345 Private Score 재현에 필요한 학습 코드, 고정 입력, 추론본,
실행 환경과 검증 자료를 제공한다. 대회가 제공하지 않은 외부 데이터와 평가 정답은 포함하지 않는다.

## 구성

- `00_RUN_REPRODUCTION.ipynb`: Python 스크립트 기반 재현 절차를 순서대로 실행하는 노트북
- `code/notebooks/experiment_workbench.ipynb`: 초기 v17 모듈의 DRY_RUN 회귀 검증용 노트북
- `02_EMAIL_ENVIRONMENT_TEXT.txt`: 메일 본문에 붙여 넣을 개발 환경·라이브러리 버전 문안
- `code/train.py`: 데이터 검사, 개별 모델 학습, lookup 생성, 재빌드 및 추론 검증
- `code/common_features/`: 공통 시즌 표본·평활 피처
- `code/h1/`, `code/strict/`, `code/fallback_xgb/`: 모델별 학습 모듈
- `code/fallback_routes/`: 고정 XGB 혼합 경로 조립
- `code/src/`: 기본 모델·보정·최종 조립 코드
- `code/tests/`: 단위·회귀 테스트
- `code/lineage_inputs/`, `code/reproduction_inputs/`: 고정 학습 체크포인트
- `code/DEVELOPMENT_RECORD.md`: 최종 v345 계보와 모델 선택 원칙
- `verification/`: 검증 결과

공식 데이터 4개를 `code/data/`에 배치한 뒤 `03_REPRODUCE.md`를 따른다.
학습 환경은 `01_ENVIRONMENT.md`, 제출 전 확인 사항은 `04_SUBMISSION_CHECKLIST.md`,
구성요소별 재현 범위와 검증 결과는 `verification/`에 정리돼 있다.

## 검증 범위

동결 v343과 공식 train에서 Beta 보정을 재학습해 원 v345 ZIP을 동일하게 재빌드하는
경로가 있다. H1·등판량 H1·C3·퓨처스 전문가·strict·초기 V2 OOF 및 부모
v124/v124_bridge의 주요 native 자산은 공식 데이터 또는 동봉된 공식-data 파생 입력으로
신규 학습 비교를 완료했다. 구성요소별 검증 조건과 결과는 `verification/`에 기록했다.
원 추론 모델·가중치·확률은 이번 폴더 통합에서 변경하지 않았다.

최종 추론 ZIP SHA-256:
`D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA`
