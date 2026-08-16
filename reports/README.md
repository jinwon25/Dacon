# 보고서 빠른 안내

최신 의사결정은 아래 순서로 확인합니다.

1. [target1160_v20_candidate_20260816.md](target1160_v20_candidate_20260816.md) — v20 구성, 순방향 검증, 부트스트랩, 한계, 재현 명령
2. [v20_validation.md](v20_validation.md) / [JSON](v20_validation.json) — 실제 ZIP 계보·런타임·메모리·배치 독립성
3. [v19_public_result_20260816.md](v19_public_result_20260816.md) — 공식 Public 1144.1518과 champion 결정
4. [target1150_joint_candidate_20260816.md](target1150_joint_candidate_20260816.md) — v19의 state/failure-mode 근거
5. [submissions.csv](submissions.csv) — 공식 제출 이력의 단일 기록원

## 현재 판정

- 공식 champion: `submit_v19.zip`
- 미제출 challenger: `submit_v20.zip`
- v20 순방향 최소 gain: `+14.2517` vs v19
- v20 중심 Public 추정: 약 `1159.75`
- 주의: 2024 월별 증분은 8개 중 5개 양수이므로 v20은 공격형 후보

날짜가 붙은 나머지 보고서는 과거 실험과 기각 가설의 근거입니다. 삭제하지 않되 최신 결론으로 오인하지 않도록 새 연구에서는 위 다섯 문서를 먼저 링크하세요.

제출 ZIP, 모델, OOF는 Git에서 관리하지 않습니다. 실제 파일 위치 규칙은 [../submissions/README.md](../submissions/README.md)와 [../artifacts/README.md](../artifacts/README.md)를 참고하세요.
