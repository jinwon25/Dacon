# v345 학습 및 구성 기록

## 최종 산출물

최종 추론 산출물은 `submit_v345.zip`이며 SHA-256은
`D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA`이다.
학습과 피처 생성에는 대회가 제공한 `train.csv`와 `trackman_history.csv`만 사용했다.

## 모델 선택 원칙

평가 지표가 Brier Skill Score이고 검증 신호가 작아, 하나의 모델로 모든 행을
교체하지 않고 기준 확률에 검증된 행 집합별 보정을 더하는 구조를 사용했다.
각 구성요소는 시간 순서를 지킨 검증과 행 독립성 검사를 거쳐 적용 범위와 혼합
가중치를 고정했다. 평가 배치의 평균·순위·빈도·정답은 피처나 경로 선택에 사용하지 않는다.

## 배포 계보

- v124: 다축 quadratic stack 기준 예측
- v142~v148: H1, C3 및 bridge 보정
- bridge027·v244: 압박 상황 경로와 fallback XGBoost 혼합
- v290·v320: Futures 전용 모델과 저차원 보정
- v334·v343: anchor, 선수 전이 및 workload-H1 보정
- v345: developing·mixed 셀의 Beta-Binomial 보정

최종 경로·상수·혼합 순서는 `../inference/script.py`에 고정돼 있다. 학습 명령은
`../03_REPRODUCE.md`, 환경은 `../01_ENVIRONMENT.md`, 구성요소별 비교 결과는
`../verification/`에 기록했다.

## 검증 범위

원 제출 추론 코드는 변경하지 않았으며, 동결 v343과 공식 train에서 Beta 구성요소를
다시 적합해 동일한 v345 ZIP을 재빌드하는 경로를 확인했다. 구성요소별 신규 학습,
고정 입력 및 비교 조건은 `../verification/training_coverage.json`에 구분해 기록했다.
