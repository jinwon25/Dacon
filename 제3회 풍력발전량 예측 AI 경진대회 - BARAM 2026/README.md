# 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026

DACON [제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026](https://dacon.io/competitions/official/236727/overview/description) 참가용 실험 저장소입니다.

목표는 LDAPS/GFS 기상예보와 학습 기간 SCADA를 활용해 2025년 시간 단위 KPX 3개 그룹 발전량을 예측하는 것입니다. 대회 평가는 `0.5 * (1-NMAE) + 0.5 * FICR`이며, 실제 발전량이 설비용량의 10% 이상인 시간대만 평가됩니다.

## 현재 상태

첫 public 제출은 `submissions/lgbm_v1.csv`였고, public score는 `0.6352790723`입니다. 현재 최고 public score는 `blend_over115_scada_stack5.csv`의 `0.6402652274`입니다.

이번 업데이트에서는 기존 LightGBM 직접 예측 파이프라인을 유지하면서, 그룹 3에 CatBoost 후보를 추가했습니다. 2024년 holdout 검증 기준으로 그룹 3이 크게 개선되어 평균 검증 점수가 상승했습니다.

| 모델 | 그룹 1 | 그룹 2 | 그룹 3 | 평균 |
|---|---:|---:|---:|---:|
| LightGBM v1 | 0.6669 | 0.6693 | 0.6103 | 0.6488 |
| Hybrid: LGBM + CatBoost G3 | 0.6669 | 0.6693 | 0.6251 | 0.6538 |

선택된 hybrid 구성:

| 타깃 | 모델 | 학습 row 선택 | 보정 |
|---|---|---|---|
| `kpx_group_1` | LightGBM | 전체 유효 label | `scale=1.03`, `offset=600` |
| `kpx_group_2` | LightGBM | 설비용량 10% 이상 label | `scale=1.03`, `offset=-400` |
| `kpx_group_3` | CatBoost | 설비용량 10% 이상 label | `scale=1.04`, `offset=-400` |

생성된 제출 후보:

- `submissions/hybrid_lgbm_cat_g3_full_cal.csv`
- `submissions/hybrid_lgbm_cat_g3_half_cal.csv`
- `submissions/hybrid_lgbm_cat_g3_raw.csv`

public LB에서는 `full_cal`이 `half_cal`보다 높았습니다.

| 제출 ID | 파일 | Score | 1-NMAE | FICR |
|---:|---|---:|---:|---:|
| 1473991 | `lgbm_v1.csv` | 0.6352790723 | 0.8741552448 | 0.3964028998 |
| 1476563 | `hybrid_lgbm_cat_g3_half_cal.csv` | 0.6358853777 | 0.8740510337 | 0.3977197217 |
| 1476564 | `hybrid_lgbm_cat_g3_full_cal.csv` | 0.6366019333 | 0.8736225508 | 0.3995813158 |
| 1476569 | `hybrid_lgbm_cat_g3_cal125.csv` | 0.6367222443 | 0.8730264358 | 0.4004180529 |
| 1476571 | `hybrid_lgbm_cat_g3_cal150.csv` | 0.6355476403 | 0.8722118983 | 0.3988833824 |
| 1477436 | `blend_v1_mix50_cal125.csv` | 0.6390450795 | 0.8745955866 | 0.4034945724 |
| 1477441 | `blend_v1.csv` | 0.6394982166 | 0.8752066646 | 0.4037897687 |
| 1477445 | `blend_v1_over110_cal125.csv` | 0.6398927384 | 0.8752047108 | 0.4045807659 |
| 1478708 | `blend_v1_over115_cal125.csv` | 0.6398981745 | 0.8751878268 | 0.4046085222 |
| 1478711 | `blend_v1_over120_cal125.csv` | 0.6396174227 | 0.8751591011 | 0.4040757444 |
| 1480959 | `blend_over115_scada_stack5.csv` | 0.6402652274 | 0.8753153766 | 0.4052150782 |
| 1480964 | `blend_over115_scada_stack10.csv` | 0.6397287088 | 0.8754003341 | 0.4040570835 |

`blend_over115_scada_stack5`는 SCADA proxy stack을 5% 섞은 후보이며, 기존 최고 대비 1-NMAE와 FICR이 모두 개선됐습니다. `stack10`은 1-NMAE는 더 올랐지만 FICR이 하락해 과혼합으로 판단합니다.

## 모델링 방향

현재 실전 우선순위는 다음과 같습니다.

1. **누수 없는 NWP 직접 예측**: 테스트 기간에는 과거 발전량/SCADA가 없으므로, 예측기준시점 이전에 공개된 LDAPS/GFS만 사용합니다.
2. **GBDT 계열 우선**: 표본 수가 약 2.6만 시간이고, 미래 기상 covariate가 강한 정형 회귀 문제라 LightGBM/CatBoost가 deep sequence 모델보다 재현성과 검증 안정성이 좋습니다.
3. **그룹별 모델 선택**: 그룹 1/2는 LightGBM이 강하고, 그룹 3은 CatBoost가 더 안정적입니다.
4. **물리 피처는 실험 후보로 유지**: 터빈 위치 기반 거리 가중 NWP와 117m hub-height proxy를 구현했지만, 전체 투입 시 검증 성능이 악화되어 기본 학습에서는 제외합니다. `--feature-set own_idw_nohub` 등으로 재실험 가능합니다.
5. **FICR 경계 대응**: MAE만 낮추는 것보다 6%, 8% 오차 경계 안으로 들어오는 샘플 수가 중요하므로 검증 기반 scale/offset 보정을 별도로 적용합니다.

## 폴더 구조

```text
.
├── README.md
├── requirements.txt
├── train.py
├── inference.py
├── src/
│   ├── features.py
│   └── metrics.py
├── docs/
│   └── modeling_strategy.md
├── data/          # 원본 데이터, git 제외
├── artifacts*/    # 학습 모델, git 제외
└── submissions/   # 제출 CSV, git 제외
```

## 실행 방법

```bash
pip install -r requirements.txt

python train.py \
  --data-dir data \
  --artifact-dir artifacts_hybrid \
  --feature-set base \
  --catboost-targets kpx_group_3

python inference.py \
  --data-dir data \
  --artifact-dir artifacts_hybrid \
  --output submissions/hybrid_lgbm_cat_g3_full_cal.csv \
  --calibration-strength 1.0
```

보정 강도별 제출 파일:

```bash
python inference.py --data-dir data --artifact-dir artifacts_hybrid --output submissions/hybrid_lgbm_cat_g3_half_cal.csv --calibration-strength 0.5
python inference.py --data-dir data --artifact-dir artifacts_hybrid --output submissions/hybrid_lgbm_cat_g3_raw.csv --calibration-strength 0.0
```

## 재현 환경

검증 환경:

| 항목 | 버전 |
|---|---|
| Python | 3.11.2 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.8 |
| NumPy | 2.2.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.1 |

## 대회 규칙 준수

- 예측기준시점 이후 생성/공개/확정된 기상예보, 실측, 사후 보정 자료를 사용하지 않습니다.
- 테스트 기간 실제 발전량과 테스트 기간 SCADA를 사용하지 않습니다.
- 외부 데이터는 현재 사용하지 않습니다.
- train/inference 코드를 분리해 2차 검증 요구사항에 맞춥니다.
