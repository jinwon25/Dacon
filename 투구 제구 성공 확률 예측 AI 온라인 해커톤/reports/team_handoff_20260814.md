# 팀 전달 패키지 안내 (2026-08-14)

## 기준 산출물

- 제출 파일: `submit_v10.zip`
- Public: `1015.8076603531`
- 순위: `212`
- 제출 목록 번호: `48461`
- SHA-256: `08DCDB49997AE680CCED2F15B561280BAB063D40993825E621AA96024040B345`

최종 팀 전달용 ZIP을 갱신할 때 `submission/submit_v10.zip`은 실제 평가에 사용한 파일을 변경 없이
복사한 것이다. 같은 제출 산출물이 필요하면 이 내부 ZIP을 그대로 사용한다.

## 로컬 실행

1. `submission/submit_v10.zip`을 별도 폴더에 푼다.
2. 공식 평가 파일을 그 폴더의 `data/test.csv`에 둔다.
3. Python 3.11 환경에서 `pip install -r requirements.txt`를 실행한다.
4. `python script.py`를 실행한다.
5. 결과는 `output/submission.csv`에 생성된다.

입력 파일과 환경이 같으면 같은 모델 파일과 추론 코드가 실행된다. 공식 데이터는
라이선스와 배포 범위를 고려해 전달 ZIP에 넣지 않았으며, 팀원이 데이콘에서 받은
원본을 사용해야 한다.

## 포함 파일

- `submission/submit_v10.zip`: 실제 제출 패키지.
- `source_snapshot.zip`: 해당 Git 커밋의 프로젝트 소스·테스트·문서 스냅샷.
- `TEAM_HANDOFF.md`: 이 안내서.

## 규칙 준수

추론은 각 평가 행과 공식 학습 데이터로 만든 고정 모델·lookup만 사용한다. 평가
데이터의 다른 행을 이용한 집계, rolling, lag, 빈도, 순위, 분포 보정은 없다. 1행,
전체, 셔플, 분할, 타 행 변경, 반복 입력 감사에서 최대 절대 차이는
`0.0`으로 허용오차 `1e-12`를 통과했다.

세부 근거는 `source_snapshot.zip` 안의 다음 문서에 있다.

- `reports/public_progress_20260814.md`
- `reports/package_validation.md`
- `reports/top10_strategy_20260814.md`
- `reports/final_v10_20260814.md`
