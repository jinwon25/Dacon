# 식음업장 메뉴 수요 예측 AI 온라인 해커톤

데이콘 [**식음업장 메뉴 수요 예측 AI 온라인 해커톤**](https://dacon.io/competitions/official/236559/overview/description) 솔루션.
리조트 내 9개 영업장 · 193개 메뉴의 과거 판매 데이터로 **다음 7일 메뉴별 매출수량**을 예측한다.

> 상태: **연구 중**. 현재 최고 제출 **LB Private 0.5481** (가중 SMAPE, 낮을수록 좋음) — 업장별 nz-mean 블렌드. 공식 LSTM 베이스라인(Private 0.6939) 대비 **-0.1458**. 모델 개선 진행 중.

## 현재 결과 (LB 진행)

| 후보 | 로컬 가중 SMAPE | LB Private | 비고 |
|---|---:|---:|---|
| 공식 LSTM 베이스라인 (메뉴별 개별 LSTM) | — | 0.693935 | 대회 제공 |
| 전역 블렌드 (0.4·요일평균 + 0.6·28일평균) | 0.6791 | 0.6695 | 무학습, 베이스라인 추월 |
| **업장별 nz-mean 블렌드 (tuned)** | 0.5850* | **0.5481** | ★ 현재 최고 (Public 0.5485) |
| + wd_nz 성분 (4-blend) | 0.5715* | (재제출 대기) | nested OOF 추가 개선 |

`*` = nested OOF(정직) 가중 SMAPE. 단독 글로벌 LightGBM은 0.841로 요일평균(0.74)에도 못 미쳐 폐기.

평가 지표: **가중 SMAPE** — 영업장별 가중치(담하·미라시아 고가중, 가중치 값 **비공개**), **실제값 0 항목은 평가 제외**, 낮을수록 좋음.

---

## 문제 정의

- **타겟**: 영업장·메뉴별 다음 7일 일별 매출수량 (`매출수량`)
- **데이터**
  - `data/train/train.csv`: 102,676 행 (long) — `영업일자, 영업장명_메뉴명, 매출수량`. 2023-01-01 ~ 2024-06-15 (532일), 193개 메뉴 전부 풀히스토리.
  - `data/test/TEST_00..09.csv`: **독립 평가 샘플 10개**, 각 샘플은 메뉴별 28일 관측치만 포함 → 직후 7일 예측.
  - `data/sample_submission.csv`: 70행(10샘플 × 7일) × 193 메뉴열 (wide).
- **지표**: 가중 SMAPE (담하·미라시아 고가중·비공개, **0 실제값 평가 제외**). 정의는 `src/metric.py` 참조.
- **데이터 특성**: 매출수량의 **53%가 0**, 평균 10.65, 최대 1372 (강한 우편향 롱테일). 영업장별 행 수 편차 큼(담하·미라시아 최대).

## 누수(Data Leakage) 규칙 (공식 Q&A)

- 각 TEST 샘플은 **자신의 28일만** 사용. 다른 샘플(TEST_00↔01) 정보 공유 **금지**.
- 28일 입력 윈도우를 과거로 **확장 금지**, 추론 시점 이후 정보 **금지**.
- **허용**: recursive 예측(예측값을 다음 날 입력으로), 메뉴별 다른 모델, 입력 28일 내 통계 피처, 달력/공휴일 등 사전 지식.
- test 결과를 학습/pseudo-label 에 사용 **금지**.

---

## 솔루션 — "0 제외 SMAPE" 특성을 노린 무학습 블렌드

### 핵심 통찰

평가에서 **실제값 0 항목이 제외**되므로, 예측 목표는 "전체 평균"이 아니라 **비-0 날의 수준(level)** 이다.
28일 윈도우의 0 제외 평균(`nz_mean`)을 블렌드 성분으로 넣자 가중 SMAPE가 급감했다 (전역 블렌드 0.6695 → 업장튜닝 0.5481).

```
성분: weekday_mean(요일평균) · mean28(28일평균) · nz_mean(0 제외 평균) · wd_nz(요일별 0 제외 평균)
      업장별 simplex 가중 + 전역 스케일(0.7~0.9)을 4-fold 홀드아웃에서 탐색
예:   연회장   = nz 1.0           (영업장 SMAPE 0.897 → 0.355)
      미라시아 = 0.25·wd + 0.75·nz (0.695 → 0.578, 고가중)
      담하     = 0.25·wd + 0.75·nz (0.691 → 0.613, 고가중)
```

### 핵심 설계 결정

1. **무학습 블렌드 > 단일 모델.** 메뉴별 LSTM·단일 글로벌 GBDT(0.841) 모두 요일평균(0.74)도 못 이긴다 — 0이 53%·짧은 시리즈·상대지표 조합에서 학습 모델이 과적합. 통계 베이스라인 블렌드가 더 견고.
2. **nz_mean이 결정적 레버.** 0 제외 평가 특성을 직접 겨냥 → 전역 대비 **-0.094**(정직 로컬), 모든 폴드 일관.
3. **업장별 가중 튜닝.** 고가중 담하·미라시아를 함께 끌어내리는 것이 점수의 핵심. wd_nz(요일별 비-0 평균) 추가로 주간 계절성까지 결합 (nested OOF 0.585 → 0.572).
4. **로컬 CV ≈ Private LB.** 4-fold 홀드아웃 가중 SMAPE가 LB Private와 거의 일치 (전역 0.679 ≈ 0.6695, 업장튜닝 0.585 ≈ 0.5481) → 제출 없이 로컬로 개선 측정 가능. 가중치 비공개를 ×2 proxy로 대체해도 변환이 깨끗.

### 시도했지만 효과 없음

| 기법 | 결과 |
|---|---|
| 메뉴별 개별 LSTM (대회 베이스라인) | 요일평균(0.74) 수준도 못 짜냄 |
| 단일 글로벌 LightGBM (direct multi-horizon) | OOF 0.841 — 요일평균보다 나쁨 |
| mean7 성분 | 블렌드 가중 0 (28일평균이 더 안정) |
| SMAPE-aware 스케일 c | 효과 미미 (0.679 → 0.677) |

---

## 폴더 구조

```text
.
├── README.md                    # 프로젝트 요약 (이 파일)
├── requirements.txt             # Python 의존성
├── .gitignore
│
├── src/                         # 베이스라인 / 블렌드 / 지표 코드
│   ├── metric.py                # 가중 SMAPE + 로컬 검증 분할(make_holdout)
│   ├── baselines.py             # 무학습 베이스라인(요일평균/mean7/mean28 등) 로컬검증
│   ├── experiments.py           # 블렌드 alpha/스케일 탐색
│   ├── features.py              # 글로벌 GBDT용 피처(28일 윈도우 + 달력)
│   ├── train_lgbm.py            # 글로벌 LightGBM (direct multi-horizon, 폐기)
│   ├── tune_per_store.py        # ★ 업장별 nz-mean 블렌드 가중/스케일 튜닝
│   ├── predict_blend.py         # 전역 블렌드 제출 생성 → blend_submission.csv
│   ├── predict_tuned.py         # ★ 업장별 튜닝 블렌드 제출 생성 → tuned_submission.csv (현재 최고)
│   └── baseline_lstm.py         # 대회 제공 베이스라인(메뉴별 LSTM)
│
├── notebooks/
│   └── trial_final.ipynb        # time-feature + meta-learner 실험
│
├── docs/
│   └── WORK_LOG.md              # 작업 로그 (EDA·로컬검증·블렌드 연구 시간순)
│
├── data/                        # gitignore (대회 원본)
│   ├── train/train.csv
│   ├── test/TEST_00..09.csv
│   └── sample_submission.csv
│
└── submissions/                 # gitignore (생성 제출물)
```

커밋 기준:
- 포함: 코드(`src/`), README, `docs/`, `requirements.txt`, 노트북
- 제외: `data/`(대회 원본), `submissions/`(재생성 가능한 생성 제출물), 캐시·로그

---

## 재현 방법

Python 3.11 권장. **무학습 블렌드라 CPU만으로 충분**(LSTM 베이스라인만 PyTorch 사용).

```bash
pip install -r requirements.txt

python src/metric.py            # 지표 self-test
python src/baselines.py         # 무학습 베이스라인 로컬 4-fold 검증
python src/tune_per_store.py    # 업장별 nz-mean 블렌드 가중/스케일 탐색
python src/predict_tuned.py     # -> submissions/tuned_submission.csv (현재 최고, LB Private 0.5481)
```

대회 데이터는 `data/`(`data/train/`, `data/test/`, `data/sample_submission.csv`)에 배치한다.

## 다음 단계 (`docs/WORK_LOG.md` 상세)

1. 갱신된 4-blend(wd_nz 포함, nested OOF 0.5715) 재제출 → LB 변환율 확인.
2. LGBM을 **residual 보정**으로(target = y − 블렌드)해 단독 GBDT 실패를 우회, 메뉴 단위 편차 흡수.
3. 고가중 담하·미라시아(여전히 ~0.56~0.60) 집중 — 가장 큰 남은 레버.

## 환경

- Python 3.11. CPU만으로 충분 (무학습 블렌드).
- 라이브러리: pandas, numpy, scipy, scikit-learn, lightgbm, torch(베이스라인용) — 전부 오픈소스(BSD/MIT/Apache).
