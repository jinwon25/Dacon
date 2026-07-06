# 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026

데이콘 [**제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026**](https://dacon.io/competitions/official/236727/overview/description) 참가 솔루션.
LDAPS·GFS 기상예보와 학습 기간 터빈 SCADA를 활용해 3개 KPX 그룹의 다음 날 시간별 발전량을 예측한다.

> 상태: **연구 중**. 첫 Public 제출 Score **0.6352790723** — 공간 바람장·물리 피처 기반 LightGBM. 공식 RandomForest 방식의 로컬 기준선 대비 시간순 검증 Score **+0.0398**.

## 현재 결과

2022~2023년 학습, 2024년 전체 홀드아웃 검증 결과다. 실제 발전량이 설비용량의 10% 이상인 공식 평가 대상만 채점했다.

| 모델 | 그룹 1 | 그룹 2 | 그룹 3 | 평균 Score |
|---|---:|---:|---:|---:|
| 공식 RandomForest 방식 재현 | 0.6124 | 0.6255 | 0.5890 | 0.6090 |
| **공간·물리 피처 LightGBM v1** | **0.6669** | **0.6693** | **0.6103** | **0.6488** |

첫 Public 제출(`lgbm_v1.csv`, submission ID `1473991`):

| 제출 시각 (KST) | Score | 1-NMAE | FICR |
|---|---:|---:|---:|
| 2026-07-06 22:58:18 | **0.6352790723** | 0.8741552448 | 0.3964028998 |

로컬 대비 차이는 Score `-0.01355`, 1-NMAE `-0.00295`, FICR `-0.02415`다. 절대오차 일반화는 양호하지만 정산 경계 적중률의 분포 이동이 더 크다.

> 로컬 FICR은 공식 첨부 코드 확보 전까지 `오차율 ≤6%: 4`, `≤8%: 3`, 그 외 `0`으로 계산한다. 모델 간 비교에는 동일하게 적용하지만 공식 점수와 절대값은 다를 수 있다.

---

## 문제 정의

- **타겟**: 2025년 8,760시간 × KPX 그룹 3개의 발전량(kWh)
- **설비용량**: 그룹 1·2 각 21,600kWh, 그룹 3 21,000kWh
- **입력**:
  - LDAPS: 시간당 16개 격자의 고해상도 기상예보
  - GFS: 시간당 9개 격자의 전지구 기상예보
  - SCADA: 학습 기간 17개 터빈의 10분 단위 발전량·풍속·풍향
- **예보 구조**: 매일 09시 초기화, 13시 사용 가능한 예보로 다음 날 01시부터 24시간 예측(리드타임 12~35시간)
- **평가**: `0.5 × (1-NMAE) + 0.5 × FICR`; 실제 발전량이 설비용량의 10% 이상인 시간만 평가

그룹 1·2 라벨은 2022~2024년, 그룹 3은 2023~2024년만 제공된다. 그룹 1·2에도 약 100시간의 라벨 결측이 있어 그룹별 유효 행으로 학습한다.

## 누수 방지 원칙

- 예측 기준시점 이후 생성·관측·확정된 정보는 사용하지 않는다.
- 테스트 기간의 실제 발전량과 SCADA는 제공되지 않으므로 자기회귀 입력으로 가정하지 않는다.
- NWP는 `data_available_kst_dtm` 기준으로 사용 가능 여부를 판단한다.
- 검증은 랜덤 분할이 아니라 연도 순서가 보존된 2024년 홀드아웃을 사용한다.
- SCADA는 학습 기간의 NWP 편향 보정·파워커브 학습에만 사용한다.

---

## 솔루션 — 공간 바람장 + 물리 피처 LightGBM

공식 베이스라인은 모든 기상 격자를 평균해 지형·풍향 정보를 잃는다. v1은 풍속·풍향 성분을 격자별로 보존하고 나머지 기상장은 공간 통계로 압축한다.

```text
LDAPS 16 grids ─┐
                ├─ grid-level u/v + wind-speed magnitude ─┐
GFS 9 grids ────┘                                         │
                                                          ├─ group-wise LightGBM
temperature / pressure / humidity / clouds ─ spatial stats│
calendar + 12~35h forecast lead ──────────────────────────┘
```

### 핵심 설계 결정

1. **공간 평균 대신 격자별 바람장 유지**: 터빈 주변 지형과 풍향에 따른 국지 차이를 모델이 선택할 수 있다.
2. **물리 파생 피처**: 높이별 U/V에서 풍속 크기를 만들고 예측값을 `[0, 설비용량]`으로 제한한다.
3. **평가 분포 대응**: 전체 학습과 `실제 ≥ 설비용량 10%` 중심 학습을 2024년에서 비교해 그룹별로 선택한다.
4. **그룹별 보정**: 시간순 검증에서 scale·offset을 제한된 격자로 탐색한다.
5. **실험 회전율 관리**: 핵심 바람장은 격자별, 온도·기압·습도 등은 mean/std/min/max로 압축해 702개 피처로 구성한다.

### SCADA 발견 및 다음 단계

이상치(`±51,770,425`)를 제거하고 10분 발전량을 시간 합계로 변환하면 SCADA와 KPX 라벨의 상관계수는 그룹별 **0.979~0.988**이다. 다음 실험은 다음 두 경로의 앙상블이다.

1. 직접 경로: LightGBM + CatBoost + XGBoost의 시간순 OOF 블렌드
2. 물리 경로: `NWP → 현장 풍속/출력 보정 → 단조 파워커브 → 발전량`

세부 문헌·모델 선택 근거는 [`docs/modeling_strategy.md`](./docs/modeling_strategy.md)에 정리했다.

---

## 폴더 구조

```text
.
├── README.md                 # 프로젝트 개요와 현재 결과
├── requirements.txt         # 검증한 Python 의존성 버전
├── .gitignore               # 원본 데이터·모델·생성 제출 제외
├── train.py                 # 시간순 검증, 모델 선택, 전체 재학습
├── inference.py             # 저장 모델로 제출 파일 생성
├── src/
│   ├── features.py          # NWP 공간·물리·달력 피처
│   └── metrics.py           # 1-NMAE/FICR 로컬 지표
├── docs/
│   └── modeling_strategy.md # 문헌 기반 실험 로드맵
├── data/                    # gitignore: 대회 원본 데이터
├── artifacts/               # gitignore: 학습 모델·리포트
└── submissions/             # gitignore: 생성 제출 파일·LB 기록
```

커밋 기준:

- **포함**: 학습·추론·피처·지표 코드, README, 문헌 전략, requirements, 공개 공식 베이스라인 노트북
- **제외**: 대회 원본 데이터, `info.xlsx`, 모델 체크포인트, 제출 CSV, 캐시·로그

---

## 재현 방법

Python 3.11.2, CPU 환경에서 검증했다.

```bash
pip install -r requirements.txt

# 시간순 검증 + 그룹별 최종 모델 학습
python train.py --data-dir data --artifact-dir artifacts

# 제출 파일 생성
python inference.py \
  --data-dir data \
  --artifact-dir artifacts \
  --output submissions/lgbm_v1.csv
```

대회 데이터는 설명서 구조 그대로 `data/train/`, `data/test/`, `data/sample_submission.csv`에 둔다. `train.py`와 `inference.py`는 최종 산출물 요구사항에 맞게 분리되어 있으며 모든 seed를 고정한다.

### 검증 환경

| 항목 | 버전 |
|---|---|
| Python | 3.11.2 |
| LightGBM | 4.6.0 |
| NumPy | 2.2.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.1 |

## 대회 규칙 준수

- 외부 데이터 및 원격 추론 API 미사용
- 모든 모델은 제공 train 데이터로 로컬 학습
- 테스트 실제값·SCADA·사후 관측값 미사용
- 원본 대회 데이터는 저장소에 포함하지 않음
- 예보 생성·공개 시점을 기준으로 피처 가용성 판단
