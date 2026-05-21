# 작업 로그 — 스마트 창고 출고 지연 예측 AI 경진대회

이 문서는 baseline → 최종 LB 9.86576까지의 시도와 발견을 시간순으로 기록합니다. 각 단계의 OOF/LB 변화와 그 이유를 함께 적어, 비슷한 분포 이동 회귀 문제에 적용할 수 있는 일반 원칙을 추출하는 것이 목표입니다.

## 최종 결과 요약

```
시작 LB: 10.7079 (베이스라인 3-모델)
최종 LB: 9.86576 (19-모델 mega-blend)
개선:    -0.84 (51위권 진입)
```

## 데이터 구조 이해

처음 데이터를 받고 가장 먼저 한 일은 train/test의 분포 차이 진단:

```python
import pandas as pd
tr = pd.read_csv('data/train.csv')
te = pd.read_csv('data/test.csv')
overlap = set(tr['layout_id']) & set(te['layout_id'])
# 결과: train 250 layouts, test 100 layouts, overlap 50 (50% disjoint)
```

→ test 행의 40%는 train에서 한 번도 본 적 없는 layout. 이게 본 대회의 핵심 어려움.

`avg_delay_minutes_next_30m` (target):
- mean 18.96, std 27.35
- p50 9.03, p90 45.24, p99 120.85, max 715.86
- 강한 우측 치우침 → log1p 변환 효과 클 것

## 시간순 작업 흐름

### Phase 1: 베이스라인 (LB 10.71 → 10.22)

처음 시도한 LightGBM L1 모델로 OOF MAE 8.73, LB 10.71. 이후:
- L2 추가 → 블렌드 OOF 8.69, LB 10.22 (3-모델 비-log L1+L2 블렌드)
- log1p 타겟 + Tweedie (vp=1.5) 추가 → 6-모델 블렌드, OOF 8.60, **LB 10.04**

핵심 발견:
- **`--drop-layout-id` 필수**: layout_id를 카테고리로 직접 입력하면 OOF 악화. test layout이 다수 disjoint이라 카테고리 임베딩 학습이 일반화 안 됨.
- **log1p 타겟이 일관되게 OOF 개선**: 우측 치우침 분포에서 중간/하단 fit 안정화.
- **Tweedie의 다양성**: 단독 OOF 8.70로 약간 떨어지나, log L1과 분포 가정이 달라 블렌드에서 33% 가중치 받음.

### Phase 2: GBDT 다양성 (LB 10.04 정체)

CatBoost, XGBoost, HistGradientBoosting 추가:
- CatBoost log-MAE: OOF 8.68
- XGBoost reg:absoluteerror log: OOF 8.66
- HistGradientBoosting absolute_error: OOF 8.72

8-모델 블렌드 OOF 8.59. **LB 10.04** 정체.

이 시점에서 **adversarial validation** 진단:
```python
# train + test 결합해 이진 분류 (train=0, test=1)
# 결과: AUC = 1.0 (완벽 분리)
```

→ 모든 GBDT가 train layout-specific 패턴에 동일하게 과적합. 알고리즘 다양성만으로는 천장 도달.

#### 시도했지만 효과 없는 기법들

| 기법 | OOF | LB | 이유 |
|---|---:|---:|---|
| per-bin (3 bins) SLSQP | 8.574 | 10.0547 (악화) | OOF에 fitting → LB 일반화 안 됨 |
| Isotonic calibration | 8.976 | - | 즉시 OOF 악화 |
| Adversarial weighting (p/(1-p)) | - | - | AUC=1.0 → 가중치 모두 ~1e-7 |
| Quantile matching post-process | 8.62 | - | underprediction은 L1 최적이라 수정하면 손해 |
| Layout cluster encoding (KMeans K=20) | 8.71 | - | cluster_mean_y target encoding 효과 없음 |
| Drop layout features | 8.80 | - | OOF 단독 악화, 블렌드 가중 0 |
| 2-stage tail recovery (residual + classifier) | 9.10 | - | 꼬리 -10%이지만 전체 가중 0 |
| Stacking with Ridge meta | 8.90 | - | 메타가 OOF에 과적합 |

### Phase 3: Sequence 모델 도입 (LB 10.04 → 9.92, 51위 진입) ★

GBDT 천장을 깬 결정적 단계.

**관찰**: 각 시나리오는 25개 시간 슬롯의 시계열인데 GBDT는 lag1/2 + rolling 3만 본다. 25 슬롯 전체의 시퀀스 패턴을 NN으로 학습하면 GBDT가 못 보는 신호 포착 가능.

**구현**: 1D CNN with residual + dilated convolution over 25 slots.

```python
# 핵심 구조 (train_seq_cnn.py)
- Per-slot encoder: linear(332→64)
- 3 ResBlock1D with dilation [1, 2, 4]
- Per-slot head: linear → 1 (predict per slot)
- L1 loss on log1p target
- Scenario-grouped batches (B scenarios × 25 slots × F features)
```

결과: 단독 OOF **8.70** (GBDT base 8.65와 거의 동급). 블렌드에서 0.40 가중치!

8-모델 + SeqCNN 블렌드: **OOF 8.49, LB 9.9235 (51위 진입)**.

이게 본 대회의 결정적 점프 (LB -0.12).

### Phase 4: Sequence 다양화 (LB 9.92 → 9.87)

Sequence 모델의 다양성 확장:

| 모델 | 단독 OOF | 블렌드 가중치 |
|---|---:|---:|
| SeqCNN seed 42 | 8.7016 | 0.10 |
| SeqCNN seed 7 | 8.7267 | 0.05 |
| SeqCNN seed 2026 | 8.6775 | 0.18 |
| SeqCNN big (hidden 128, 4 blocks) | 8.7056 | 0.10 |
| BiGRU seed 42 | 8.6268 | 0.20 |
| BiGRU seed 7 | 8.6142 | 0.16 |
| BiGRU seed 2026 | 8.6349 | 0.10 |
| BiLSTM seed 42 | 8.6869 | 0 (가중) |
| Transformer seed 42 | 8.7939 | 0 (가중) |

**시드 ensemble vs 아키텍처 다양성**: BiGRU 시드 3개 합계 0.46 가중치 (시드 다양성이 매우 효과적). 반면 BiLSTM/Transformer는 BiGRU와 신호 중복 → 가중 0.

새 피처 추가:
- **within-scenario rank** (분포 자유, layout disjoint에 강건)
- **anomaly indicator**: |x - median| / MAD (robust z-score)
- **rolling 5 window mean/std**

피처 수 332 → 424. LGB v2 (재학습) OOF 8.65 → 8.64. BiGRU v2 OOF 8.63 → **8.58 (-0.04)** — 가장 큰 NN 단독 개선.

14-모델 블렌드 OOF 8.42, **LB 9.87**.

### Phase 5: SOTA 권고 적용 (LB 9.87 → 9.866)

OOF/LB 정체 진단 후 SOTA 문헌 권고를 실행:
1. **Manifold Mixup** (Verma et al. ICML 2019) — input + manifold mixup, lam ~ Beta(0.4, 0.4)
2. **SwapNoise** (Jahrer 2017 Porto Seguro winner) — 15% 피처 swap + reconstruction loss aux
3. **SmoothL1 loss** (β=1.0) — heavy-tail에서 pure L1보다 안정 (Barron 2019 권고)
4. **C-Mixup** (Yao et al. 2022) — regression-specific mixup

```python
# train_seq_cnn_mixup.py 핵심
def forward(self, x, mixup_lam, mixup_idx, manifold=False):
    if not manifold:
        x = mixup_lam * x + (1 - mixup_lam) * x[mixup_idx]  # input mixup
    h = self.encoder(x)
    if manifold:
        h = mixup_lam * h + (1 - mixup_lam) * h[mixup_idx]  # manifold mixup
    h = self.cnn_blocks(h)
    return self.head(h), self.recon_head(h)  # main + aux
```

| 모델 | 단독 OOF | 변화 |
|---|---:|---:|
| Mixup SeqCNN seed 42 | 8.5774 | vs vanilla 8.7016 (**-0.12**) |
| Mixup SeqCNN seed 7 | 8.6044 | |
| Mixup SeqCNN seed 2026 | 8.5678 | |
| Mixup BiGRU seed 42 | 8.5854 | vs vanilla 8.6268 (-0.04) |
| Mixup BiGRU seed 7 | 8.5909 | |

블렌드 OOF 8.42 → **8.41**, LB 9.87 → **9.866**.

### Phase 6: 최종 19-모델 mega-blend (LB 9.86576)

마지막 시도로 Mixup CNN/BiGRU 시드 모두 통합:

| Family | 가중치 합계 |
|---|---:|
| LightGBM (3 seeds + 2 Tweedie) | 0.27 |
| CatBoost + XGBoost | ~0 (signal 중복) |
| Vanilla SeqCNN (3 seeds + big) | 0.27 |
| BiGRU (3 seeds) | 0.27 |
| Mixup SeqCNN (3 seeds) | 0.13 |
| Mixup BiGRU (2 seeds) | 0.17 |

최종 OOF **8.4129**, LB **9.86576**.

### 시도했지만 의미 없었던 것 (Phase 6 이후)

- **AutoGluon presets='good_quality'** (10분/fold × 5): OOF 8.81, 블렌드 가중 0
- **Stacking with Ridge meta**: OOF 8.90 (악화)
- **Knowledge distillation 시도**: 시간 부족으로 미완

## 핵심 학습

### 1. 분포 이동 하의 평가
- adversarial AUC > 0.9이면 OOF/LB 괴리 큼
- per-bin/isotonic/stacking은 OOF에 fitting되어 LB 악화 가능
- **단순 SLSQP (비음수, 합=1) 가중치만 안전**

### 2. Sequence 데이터에서의 NN 가치
- 시계열 구조가 있는 회귀 문제에서 GBDT lag 피처는 부족
- 1D CNN/GRU가 GBDT 천장 깰 수 있음 (본 대회 -0.12 LB)
- 시드 ensemble이 NN에서 매우 효과적

### 3. Heavy-tail 회귀 학습 트릭
- log1p 타겟 변환 (거의 항상 효과)
- Mixup + SwapNoise (Verma 2019, Jahrer 2017)
- SmoothL1 > L1 (Barron 2019)
- 분위수 매칭 후처리는 underprediction이 L1 최적이라 손해

### 4. 누설 피처 진단
- adversarial validation의 feature importance로 확인
- 단순 ID뿐 아니라 시나리오 단위 집계가 layout 식별자가 될 수 있음
- 본 대회: `robot_total_observed`, `*_seq_mean/std/trend`가 큰 누설

## LB 제출 이력

| sub | 시각(KST) | OOF | LB | 비고 |
|---|---|---:|---:|---|
| 1 | 2026-05-02 22:05 | n/a | 10.7079 | 초기 |
| 2 | 2026-05-02 22:22 | n/a | 10.5024 | |
| 3 | 2026-05-02 22:52 | 8.6880 | 10.2240 | 3-모델 (non-log L1+L2) |
| 4 | 2026-05-02 23:57 | n/a | 10.0594 | |
| 5 | 2026-05-03 00:18 | 8.6006 | 10.0416 | 6-모델 LightGBM only |
| 6 | 2026-05-03 16:27 | 8.5741 | 10.0547 | 9-모델 + per-bin (악화) |
| 7 | 2026-05-03 16:55 | 8.5905 | 10.0397 | 6-모델 + CAT + XGB global SLSQP |
| 8 | 2026-05-03 ~21 | 8.4930 | **9.9235** | 8-모델 + Sequence CNN (51위 진입) |
| 9 | 2026-05-04 00:17 | 8.4477 | 9.8838 | + BiGRU 시드 |
| 10 | 2026-05-04 00:49 | 8.4426 | 9.8841 | + Mixup BiGRU |
| 11 | 2026-05-04 ~01 | 8.4235 | 9.8729 | + Tweedie 1.7 |
| 12 | 2026-05-04 ~02 | 8.4171 | **9.866** | + Mixup BiGRU 17-모델 |
| 13 | 2026-05-04 ~10 | 8.4129 | **9.86576** | 최종 19-모델 mega-blend |

## 대회 규칙 준수

- 사전학습 모델 사용 라이선스: MIT/Apache 2.0/BSD만 사용 ✓
- 원격 API 미사용 ✓
- test.csv 학습 금지 → 모든 train만으로 fit, pseudo-label 미사용 ✓
- 시드 고정 재현 가능 ✓
- AI 도구/에이전트 코드 주석 금지 ✓
- 제출 형식: ID + target, 50000행, NaN/neg 없음 ✓

## 환경 정보

- Python: 3.11.2
- LightGBM 4.6.0, CatBoost 1.2.8, XGBoost 3.0.1, sklearn 1.6.1
- pandas 2.2.3, numpy 2.2.6, scipy 1.15.3
- PyTorch 2.7.1+cpu (GPU 미사용)
- OS: Windows 10
- 전체 학습 시간 (CPU): ~5-7시간 (순차)
