# Private Score 재현 코드 제출 전 확인

## 제출 구성

- Python 실행 노트북: `00_RUN_REPRODUCTION.ipynb`
- 메일 기입용 환경 문안: `02_EMAIL_ENVIRONMENT_TEXT.txt`
- 개발 환경: `01_ENVIRONMENT.md`
- 학습 코드와 실행 방법: `code/`, `03_REPRODUCE.md`
- 개발·선택 기록: `code/DEVELOPMENT_RECORD.md`
- 추론 코드: 리더보드에 제출한 v345 추론본을 `inference/`에 보존
- 검증 보고서: `verification/`
- 솔루션 PPT와 Phase 3 참가 여부: 메일에 별도 첨부

## 확인된 항목

- 공식 데이터와 동봉 체크포인트의 SHA-256 검사
- 동결 v343과 공식 train의 Beta 재학습을 통한 v345 ZIP 정확 재빌드
- 원 제출 v345와 동봉 추론 `script.py`의 바이트 단위 일치
- H1·workload-H1·C3·Futures·strict 및 주요 부모 native 자산의 신규 학습 비교
- 동결 lookup 재생성 비교
- 단일행·셔플·분할 입력에 대한 최종 추론 행 독립성
- 전체 테스트 및 제출 폴더 코드 범위 검사

구성요소별 결과와 수치는 `verification/README.md` 및
`verification/training_coverage.json`에서 확인한다.

## 검증 범위

정확 재빌드, 구성요소 신규 학습 및 고정 입력 기반 비교는 서로 구분해 기록했다.
환경에 따라 직렬화 결과가 달라질 수 있는 모델은 고정 requirements와 비교 보고서를
함께 제공한다. 최종 추론에는 리더보드 제출본의 모델과 가중치를 그대로 사용한다.
세부 조건은 `verification/training_coverage.json`을 따른다.

## 실행 점검

코드 자료만 점검한다.

```powershell
python code/check_submission_contents.py --package-dir . --code-only
```

PPT와 참가 여부 자료까지 준비한 뒤 전체 구성을 점검할 수 있다.

```powershell
python code/check_submission_contents.py --package-dir . --pptx SOLUTION.pptx --attendance PARTICIPATION.csv
```

`PARTICIPATION.csv`는 로컬 점검용 편의 형식이며 열은
`member,phase3_attendance`, 값은 `attending` 또는 `not_attending`이다.
실제 메일 제출 형식은 주최 측 안내를 우선한다.

확인한 공식 안내:

- [대회 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [학습·추론 과정과 설정 도출 근거 검증 안내](https://dacon.io/en/competitions/official/236743/talkboard/417157)
