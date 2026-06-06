# 스마트 창고 출고 지연 예측 AI 경진대회

데이콘에서 주최한 [**스마트 창고 출고 지연 예측 AI 경진대회**](https://dacon.io/competitions/official/236696/overview/description) 솔루션.

## 최종 결과

> 최종 순위 **32위 (상위 10%)**. 시작 베이스라인 대비 LB MAE를 **10.7079 -> 9.86576**으로 낮췄고, 절대 개선 폭은 **-0.84214 MAE**입니다.

| | OOF MAE | LB MAE | 비고 |
|---|---:|---:|---|
| 시작 베이스라인 (3-모델 LightGBM) | 8.688 | 10.7079 | 첫 제출 |
| **최종 mega-blend (19-모델)** | **8.4129** | **9.86576** | **32위 / 상위 10%** |

평가 지표: 출고 지연 시간 예측 MAE (낮을수록 좋음). LB 개선 폭 **-0.84**.

---

## 문제 정의

- **타겟**: 다음 30분간 출고 지연 평균 시간 (`avg_delay_minutes_next_30m`, 단위: 분)
- **데이터**:
  - `train.csv`: 250,000 행 × 93 컬럼 (시나리오별 25 슬롯의 운영 메트릭)
  - `test.csv`: 50,000 행 (라벨 없음)
  - `layout_info.csv`: 300 layout의 정적 메타 (면적, 로봇 수, pack station 수 등)
- **구조**: 각 시나리오 = 25개 시간 슬롯의 시계열. 동일 시나리오의 25 슬롯은 추론 시 동시 관측 가능 (공식 Q&A).
- **분포 이동 (핵심 어려움)**:
  - test의 layout_id 100개 중 50개만 train과 공유 (나머지 50개는 disjoint)
  - adversarial validation AUC = 1.0 (분류기가 train/test를 완벽 분리)
  - 결과: OOF와 LB 사이 ~1.43의 일관된 gap

---

## 솔루션 아키텍처

### 최종 19-모델 Blend (OOF 8.4129)

```
                    ┌────────── GBDT 7-모델 (~26% 가중치) ──────────┐
                    │  - LightGBM log-L1 (3 seeds, +new features) │
                    │  - LightGBM Tweedie (vp=1.5, 1.7)           │
                    │  - CatBoost log-MAE                          │
                    │  - XGBoost reg:absoluteerror log             │
                    └──────────────────────────────────────────────┘
                                          │
        ┌──────── Sequence Models 12-모델 (~74% 가중치) ────────┐
        │  ┌─ SeqCNN (small/big, 3 seeds) ─────────── 26%     │
        │  ├─ BiGRU (3 seeds) ──────────────────────── 28%    │
        │  ├─ Mixup SeqCNN (3 seeds, manifold mixup) ─ 13%    │
        │  └─ Mixup BiGRU (2 seeds, swap noise) ────── 17%    │
        └──────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  Global SLSQP Weighted Average → submissions/submission.csv
```

### 핵심 설계 결정

1. **log1p 타겟**: 우측으로 치우친 분포 (mean 19, p99 121, max 716)에서 일관된 OOF 개선 (-0.06).
2. **Sequence 모델 도입 (돌파구)**: GBDT만 쓰면 천장 ~OOF 8.44. 25 slot scenario를 1D CNN/GRU로 처리하니 OOF 8.49로 -0.05, LB 10.04→9.92 (-0.12). 본 대회의 결정적 점프.
3. **Mixup + SwapNoise + SmoothL1**: Verma 2019 Manifold Mixup + Jahrer 2017 SwapNoise + SmoothL1 (β=1.0). NN의 마지막 짜내기. Mixup SeqCNN은 단독 OOF 8.58 (vs vanilla 8.70).
4. **Global SLSQP 블렌드**: per-bin/isotonic 후처리는 LB 악화 (분포 이동 하 OOF 과적합). 단순 SLSQP(비음수 합=1) 가중치가 최선.

### 시도했지만 효과 없음

| 기법 | 결과 |
|---|---|
| per-bin SLSQP (3 bins) | OOF -0.015, LB +0.013 (악화) |
| Isotonic calibration | OOF +0.4 즉시 악화 |
| Adversarial sample weighting | AUC=1.0 → 가중치 모두 ~1e-7 (rank로 변환해도 무효) |
| Layout cluster encoding (KMeans K=20) | OOF +0.02 |
| Drop layout features | 단독 OOF +0.15, 블렌드 가중 0 |
| Quantile matching post-process | OOF +0.03 |
| 2-stage tail recovery (residual + classifier) | 꼬리 -10%이지만 전체 가중 0 |
| AutoGluon (10분/fold × 5) | OOF 8.81, 블렌드 가중 0 |
| HistGradientBoosting | OOF 8.72, 블렌드 가중 0 |
| BiLSTM, Transformer | 블렌드 가중 0 (BiGRU와 중복) |
| Stacking with Ridge meta | OOF 8.90 (악화) |

---

## 폴더 구조

```text
.
├── README.md                         # 프로젝트 요약
├── requirements.txt                  # Python 의존성
├── .gitignore
│
├── src/                              # 학습/진단/블렌딩 코드
│   ├── train_*.py
│   ├── feature_cache.py
│   ├── adversarial_validation.py
│   └── blend_safe.py
│
├── notebooks/                        # 베이스라인/탐색 노트북
├── docs/                             # 상세 작업 로그
├── submissions/                      # 최종 제출 파일
│
├── data/                             # 대회 원본 데이터, gitignore
│   └── cache/                        # 자동 생성 캐시, gitignore
│
└── models/                           # 학습 산출물, gitignore
```

커밋 기준:
- 포함: 코드, README, `docs/`, requirements, 최종 `submissions/submission.csv`
- 제외: `data/*.csv`, `data/cache/`, `models/`, `outputs/`, 로그, notebook checkpoint

주요 스크립트 역할:
- `src/train_solution.py`, `src/train_catboost.py`, `src/train_xgb.py`: GBDT 계열 학습
- `src/train_seq_cnn*.py`, `src/train_seq_gru*.py`: 시계열 딥러닝 모델 학습
- `src/blend_safe.py`: OOF 기반 최종 19-model blend 생성

`data/`의 대회 원본 CSV와 `models/`의 모델 산출물은 재생성 가능한 파일이거나 라이선스 이슈가 있는 데이터이므로 Git에 포함하지 않습니다.

---

## 재현 방법

### 1. 환경 설치

Python 3.11 권장. CPU만으로 학습 가능.

```bash
pip install -r requirements.txt
```

### 2. 데이터 배치

대회에서 받은 파일을 `data/`에 둡니다:

```
data/
├── train.csv
├── test.csv
├── layout_info.csv
└── sample_submission.csv
```

### 3. 피처 캐시 빌드 (1회)

```bash
python src/feature_cache.py
```

피처 엔지니어링 결과를 `data/cache/{train,test}_features.parquet`로 저장 → 후속 학습 시간 절약.

### 4. 모델 학습 (순차, CPU 시간 1-2시간씩)

```bash
# GBDT 7-모델
python src/train_solution.py --use-cache --drop-layout-id --log-target --output-dir models/lgb_log_l1_v2_seed42
python src/train_solution.py --use-cache --drop-layout-id --log-target --seed 2026 --output-dir models/lgb_log_l1_seed2026
python src/train_solution.py --use-cache --drop-layout-id --log-target --seed 7 --output-dir models/lgb_log_l1_seed7
python src/train_solution.py --use-cache --drop-layout-id --objective tweedie --tweedie-variance-power 1.5 --output-dir models/lgb_tweedie_seed42
python src/train_solution.py --use-cache --drop-layout-id --objective tweedie --tweedie-variance-power 1.7 --output-dir models/lgb_tweedie17_seed42
python src/train_catboost.py --output-dir models/cat_log_seed42 --log-target
python src/train_xgb.py --output-dir models/xgb_log_l1_seed42

# Sequence 12-모델 (PyTorch CPU, 각 ~10분)
for seed in 42 7 2026; do
    python src/train_seq_cnn.py --seed $seed --output-dir models/seqcnn_seed$seed
    python src/train_seq_gru.py --seed $seed --output-dir models/seqgru_seed$seed
    python src/train_seq_cnn_mixup.py --seed $seed --output-dir models/seqcnn_mixup_seed$seed
done
python src/train_seq_cnn.py --hidden 128 --n-blocks 4 --output-dir models/seqcnn_big_seed42
python src/train_seq_gru_mixup.py --seed 42 --output-dir models/seqgru_mixup_seed42
python src/train_seq_gru_mixup.py --seed 7 --output-dir models/seqgru_mixup_seed7
```

### 5. 최종 블렌드

```bash
python src/blend_safe.py \
    --model-dirs \
        models/lgb_log_l1_v2_seed42 models/lgb_log_l1_seed2026 models/lgb_log_l1_seed7 \
        models/lgb_tweedie_seed42 models/lgb_tweedie17_seed42 \
        models/cat_log_seed42 models/xgb_log_l1_seed42 \
        models/seqcnn_seed42 models/seqcnn_seed7 models/seqcnn_seed2026 \
        models/seqcnn_big_seed42 \
        models/seqcnn_mixup_seed42 models/seqcnn_mixup_seed7 models/seqcnn_mixup_seed2026 \
        models/seqgru_seed42 models/seqgru_seed7 models/seqgru_seed2026 \
        models/seqgru_mixup_seed42 models/seqgru_mixup_seed7 \
    --output-dir models/blend_final

cp models/blend_final/submission.csv submissions/submission.csv
```

---

## 주요 발견

1. **시계열 구조의 데이터에서 sequence NN은 GBDT를 보완할 수 있다**
   - 25-slot scenario의 시간 패턴을 GBDT는 lag 피처로만 보지만, 1D CNN/GRU는 시퀀스 전체를 본다
   - 단독 성능이 GBDT보다 약간 떨어져도 (8.7 vs 8.6) 블렌드에서 큰 가중치 (개별 NN ~0.2 weight)

2. **분포 이동 하 OOF/LB 괴리**
   - adversarial AUC=1.0이면 OOF에 강하게 fitting되는 후처리(per-bin, isotonic, stacking)는 LB 악화
   - 단순 평균 또는 global SLSQP만 사용하는 것이 안전
   - LB-OOF gap ~1.43은 일관됨 → OOF 개선 폭은 LB로 거의 그대로 전이

3. **Mixup의 효과 (regression on tabular sequences)**
   - Manifold Mixup (Verma 2019) + C-Mixup (Yao 2022)으로 SeqCNN OOF 8.70 → 8.58 (-0.12)
   - SwapNoise (Jahrer 2017 Porto Seguro 우승) auxiliary task 보탬
   - SmoothL1 loss는 heavy-tail 회귀에서 pure L1보다 안정

4. **카테고리 누설 진단**
   - 단순 layout_id뿐 아니라 시나리오 단위 집계(`*_seq_mean`, `*_seq_std`, `robot_total_observed`)가 layout 식별자로 작용 → adversarial AUC 측정으로 확인

---

## 작업 흐름 (시간순)

자세한 내용은 [WORK_LOG.md](./docs/WORK_LOG.md) 참고.

| 단계 | 결과 |
|---|---|
| 1. 베이스라인 LightGBM 3-모델 (L1, L2, log L1) | LB 10.22 |
| 2. 6-모델 GBDT 블렌드 (Tweedie 추가) | LB 10.04 |
| 3. CatBoost + XGBoost 추가 | LB 10.04 |
| 4. **Sequence CNN 도입 (돌파구)** | LB 9.92 (51위 진입) |
| 5. BiGRU + 시드 ensemble | LB 9.88 |
| 6. Tweedie variance_power=1.7 추가 | LB 9.87 |
| 7. 새 피처 (within-scenario rank, anomaly, rolling 5) | OOF -0.015 |
| 8. **Mixup BiGRU + SwapNoise + SmoothL1** | LB 9.866 |
| 9. **최종 19-모델 mega-blend (Mixup CNN×3, Mixup BiGRU×2 추가)** | **LB 9.86576** |

---

## 대회 규칙 준수

- 사전학습 모델: 허용된 라이선스 (MIT/Apache 2.0/BSD)만 사용. ✓
- 원격 API 모델 (OpenAI, Gemini 등): 미사용. ✓
- `test.csv`로 학습 금지 → 학습 fit에 test 행 미포함, pseudo-label 미사용. ✓
- 같은 시나리오 내 25 슬롯 동시 관측은 공식 Q&A에서 허용 → scenario 내부 보간/요약은 train/test 각각 자기 시나리오 내부에서만 계산. ✓
- 외부 데이터 미사용. ✓
- 시드 고정 재현 가능. ✓

---

## 사용 라이브러리

| 라이브러리 | 버전 | 라이선스 |
|---|---|---|
| LightGBM | 4.6.0 | MIT |
| CatBoost | 1.2.8 | Apache 2.0 |
| XGBoost | 3.0.1 | Apache 2.0 |
| PyTorch | 2.7.1+cpu | BSD |
| scikit-learn | 1.6.1 | BSD-3 |
| pandas | 2.2.3 | BSD-3 |
| numpy | 2.2.6 | BSD-3 |
| scipy | 1.15.3 | BSD-3 |
| pyarrow | 20.0.0 | Apache 2.0 |

모두 오픈소스 + 상업적 이용 허용 라이선스.

## 환경

- Python: 3.11.2
- OS: Windows 10 (Linux/Mac 호환)
- CPU 전용 (GPU 미사용)
- 전체 학습 시간: ~5-7시간 (순차 실행 기준)
