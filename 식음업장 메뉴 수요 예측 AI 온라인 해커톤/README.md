# 식음업장 메뉴 수요 예측 AI 온라인 해커톤

데이콘 [**식음업장 메뉴 수요 예측 AI 온라인 해커톤**](https://dacon.io/competitions/official/236559/overview/description) 솔루션.
리조트 내 9개 영업장 · 193개 메뉴의 과거 판매 데이터로 **다음 7일 메뉴별 매출수량**을 예측한다.

> 상태: **연구 단계 (재정비 중)**. 기존 실험 코드를 레포 컨벤션에 맞춰 정리하고, 로컬 검증 → 모델 개선을 진행 중.

---

## 문제 정의

- **타겟**: 영업장·메뉴별 다음 7일 일별 매출수량 (`매출수량`)
- **데이터**
  - `data/train/train.csv`: 102,676 행 (long) — `영업일자, 영업장명_메뉴명, 매출수량`. 2023-01-01 ~ 2024-06-15 (532일), 193개 메뉴 전부 풀히스토리.
  - `data/test/TEST_00..09.csv`: **독립 평가 샘플 10개**, 각 샘플은 메뉴별 28일 관측치만 포함 → 직후 7일 예측.
  - `data/sample_submission.csv`: 70행(10샘플 × 7일) × 193 메뉴열 (wide).
- **지표**: **가중 SMAPE** (영업장별 가중치, 작을수록 좋음) — 정확한 가중치 정의는 `src/metric.py` 상단 주석 참조(확인 필요).
- **데이터 특성**: 매출수량의 **53%가 0**, 평균 10.65, 최대 1372 (강한 우편향 롱테일). 영업장별 행 수 편차 큼(담하·미라시아 최대).

## 누수(Data Leakage) 규칙 (공식 Q&A)

- 각 TEST 샘플은 **자신의 28일만** 사용. 다른 샘플(TEST_00↔01) 정보 공유 **금지**.
- 28일 입력 윈도우를 과거로 **확장 금지**, 추론 시점 이후 정보 **금지**.
- **허용**: recursive 예측(예측값을 다음 날 입력으로), 메뉴별 다른 모델, 입력 28일 내 통계 피처, 달력/공휴일 등 사전 지식.
- test 결과를 학습/pseudo-label 에 사용 **금지**.

---

## 폴더 구조

```text
.
├── README.md
├── requirements.txt
├── data/                      # gitignore (대회 원본)
│   ├── train/train.csv
│   ├── test/TEST_00..09.csv
│   └── sample_submission.csv
├── src/
│   ├── baseline_lstm.py       # 대회 제공 베이스라인 (메뉴별 개별 LSTM)
│   └── metric.py              # 가중 SMAPE + 로컬 검증 분할(make_holdout)
├── notebooks/
│   └── trial_final.ipynb      # time-feature + meta-learner 실험
├── docs/
│   └── WORK_LOG.md            # 작업 로그
└── submissions/               # gitignore (생성 제출물)
    └── baseline_submission.csv
```

## 실행

```bash
pip install -r requirements.txt
python src/baseline_lstm.py     # -> submissions/baseline_submission.csv
python src/metric.py            # 지표 self-test
```

## 접근 메모

- 베이스라인은 메뉴별 독립 LSTM(28→7, MinMax 스케일). 0이 많고 시리즈가 짧아 메뉴별 단일모델은 과적합/과소적합 위험.
- 개선 방향(검토 중): 글로벌 모델(메뉴 임베딩) · 요일/공휴일 피처 · 0 다수 대응(롱테일/intermittent) · 단순 통계 베이스라인(요일별 평균) 대비 검증 · 앙상블.
- **로컬 검증 우선**: test 와 동일 구조(28→7) 홀드아웃으로 가중 SMAPE 를 재현해 LB 변환률을 확보한 뒤 모델을 키운다.
