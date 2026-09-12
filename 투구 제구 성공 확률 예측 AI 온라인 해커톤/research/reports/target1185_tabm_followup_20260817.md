# 1185 목표 TabM-mini 독립 구조 감사 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

## 결론

새로운 제출 후보는 만들지 않았다. 현재 champion은 계속 `submit_v27.zip`, Public
**1157.9736407889**, 확인 당시 **10위**다.

기존 embedding MLP와 다른 parameter-efficient ensemble 구조를 구현해 검증했지만,
late-2023에서 선택한 F 경로가 full/late-2024에서 모두 반전했다. 모델 용량이나 단일
MLP 최적화 문제라기보다 F의 구조 변화가 선택 자료에서 강한 가짜 개선을 만드는 것이
다시 확인됐다. 동일 deep tabular 구조의 width·k·epoch 튜닝은 재시도하지 않는다.

## 방법

ICLR 2025 TabM 논문과 공식 구현의 핵심 원칙을 현재 CPU/제출 환경에 맞게 작게
구현했다.

- 논문: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html
- 공식 구현: https://github.com/yandex-research/tabm
- `k=8`, shared 2-layer MLP, hidden width 48
- 첫 feature-mixing linear layer 전에 member별 element-wise affine view 생성
- member 8개의 Brier loss를 각각 계산한 뒤 평균해 학습
- 추론에서만 8개 확률 평균
- 기존 PyTorch 외 추가 패키지 없음
- 2023 예측은 2021·2022만, 2024 예측은 2022·2023만 학습
- architecture와 2-season window는 고정하고 late-2023에서 epoch/route/eta만 선택
- v27 위 1~15% convex blend, row-local domain routing만 사용

retrieval 계열 TabR도 검토했지만 147만 source 행 후보 검색, hidden 24.5만 행 추론,
제출 runtime·memory 제약에 비해 위험이 컸다. TabM 공식 연구도 단순 공유 MLP
ensemble이 attention/retrieval보다 성능-효율 trade-off가 좋다고 보고하므로 TabM을
먼저 선택했다.

## 시간축 결과

| 단계 | 결과 |
|---|---:|
| selection 후보 | 70개 |
| late-2023 사전 gate 통과 | 49개 |
| 선택 레시피 | epoch 1, F, eta 15% |
| late-2023 gain | **+68.6187** |
| late-2023 최악 월 gain | +84.2858 |
| late-2023 F gain | +793.5432 |
| full-2024 gain | **-8.2710** |
| full-2024 F gain | -70.2804 |
| late-2024 gain | **-3.4328** |
| late-2024 F gain | -27.6392 |

full-2024에서는 8개 월 중 2개만 양수였고 최악 월은 `-22.0006`이었다. late-2024도
8월 `-15.3182`, 9월 `+8.6962`, 10월 `+53.2467`로 월 안정성이 없었다. 2024를
본 뒤 R route나 더 작은 eta로 다시 선택하지 않았고, Public 제출도 하지 않았다.

## 해석

기존 단순 embedding MLP는 late-2023 `+154.17`에서 full-2024 `-67.42`로
반전했다. TabM-mini는 반전 폭을 줄였지만 부호를 안정화하지 못했다. 공유 가중치
ensemble의 regularization만으로는 F의 target/수집 체계 변화에 대응할 수 없다.

이번 결과로 다음 항목을 재시도 금지 목록에 추가한다.

1. 최근 2시즌 direct MLP를 v27에 단순 혼합
2. TabM-mini의 `k`, width, epoch만 조정하는 탐색
3. late-2023 F gain을 기준으로 deep model route를 다시 선택
4. 2024 결과를 본 뒤 동일 prediction의 R-only 또는 더 작은 eta를 채택

독립 모델 다양성의 다음 유효 경로는 팀원이 별도로 학습한 row-level OOF를 받는
것이다. 로컬 데이터만으로 계속할 경우에는 기존 official ASOF와 다른 새로운 관측
정보가 필요하며, 현재 feature set에서 같은 supervised 구조를 반복하는 것은 우선순위가
낮다.

## 재현

```powershell
python -m src.archive.v45_tabm_mini_screen --project .
python -m pytest -q
```

생성된 model prediction은
`artifacts/v45_tabm_mini_20260817_01/`에만 저장되며 Git에 포함하지 않는다.
