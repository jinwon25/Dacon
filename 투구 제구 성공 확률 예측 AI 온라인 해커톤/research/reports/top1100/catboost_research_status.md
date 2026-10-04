# CatBoost 스크린 결과와 미완료 검증

2024 탐색 점수는 초기 고정 재실행보다 좋았으나 다년 학습 외 예측·시드 반복·투수별 재추출 검증이 미완료였습니다. 유망한 연구 신호이며 제출 후보 통과나 해당 모델 계열 전체의 실패를 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: CatBoost research status

CatBoost is retained as a research candidate.

- 2024 fixed outer screen: Brier `0.248273754`.
- Legacy v2 frozen-replay 2024 Brier: `0.248440533`.
- Point estimate delta: `-0.000166533`.
- 2021–2023 CatBoost OOF: not run.
- Honest v2 OOF: blocked by strict nested runtime.
- Recency-weighted delta, worst-fold delta, seed replication and pitcher-cluster bootstrap: not evaluable.

Interpretation: this is a promising 2024 signal, not a failed model family and not a submission candidate. The next permitted experiment is a fixed CatBoost logit-offset residual with at most two preregistered configurations, trained and selected only from prior-origin OOF.
