# KMA 전문가의 연도 전이 검증

이전 연도에서 학습한 별도 전문가를 다음 연도 기준 예측과 비교했습니다. 해당 경로는 기각됐으며, 이를 이전 연도의 기존 기준 모델 OOF 복원으로 해석하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: KMA 2023→2024 year-forward sparse expert blend — 2026-07-26

## Decision

The branch is rejected and no submission CSV was created. The active candidate
remains:

`artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv`

This experiment does not claim to reconstruct a 2023 incumbent OOF. It trains a
new independent KMA expert on 2023 and evaluates transfer into the existing 2024
exact-OOF surface.

## Source collection and causality

The existing KMA collector downloaded the complete 2023 UMRG/N512 archive:

- 366 forecast issues
- 8,760 hourly targets
- 1,464 source objects
- two forecast cycles
- surface plus 850/700 hPa context
- conservative 12-hour publication delay
- 732 availability audit rows and zero timing violations
- feature SHA-256:
  `9f96d82686aeee8a92e6ad2d9bfdb8ec011bf52adde38d16eee69e56c1c56736`

The retained provenance is:

`artifacts_final/external_weather/kma_um_regional_context_2023/manifest.json`

## Validation contract

1. Train the prior expert only on available 2023 labels and causal KMA/GFS
   forecasts.
2. Reuse the already public-successful KMA/GFS power-curve architecture,
   movement bound, gate definitions, and `median_group123_ratio` proxy.
3. Use the 2024 Q1-trained active expert as the development reference.
4. Select a sparse prior/recent expert blend only on 2024 Q2.
5. Limit changed Q2 rows to at most 25%.
6. Refit the active expert on H1 and evaluate H2 exactly once.
7. Require positive score, 1-NMAE, and FICR, non-negative monthly scores,
   positive issue-block bootstrap q05, and a minimum locked score gain of
   `0.001`.

## Direct history pooling

Sixteen combinations were tested from four label proxies and four historical
weighting modes. None improved score, 1-NMAE, and FICR together against the
recent Q1 curve on Q2.

The least harmful pooled policy used `median_group123_ratio` with 2023 sample
weight `0.25`, but still lost:

| Component | Q2 delta |
|---|---:|
| Score | `-0.00076797` |
| 1-NMAE | `-0.00045336` |
| FICR | `-0.00108258` |

The direct pooling branch was closed before H2.

## Sparse mixture-of-experts

The Q2-selected policy moved 20% toward the frozen 2023 expert only where:

- expert direction: either
- disagreement coverage: top 5%
- active prediction ratio: 10%–60% of capacity
- absolute expert disagreement: at least `459.15 kWh`

Q2 changed 95 rows (4.35%) and improved all components:

| Component | Q2 delta |
|---|---:|
| Score | `+0.00034336` |
| 1-NMAE | `+0.00009111` |
| FICR | `+0.00059561` |

On locked H2 the same rule changed 106 rows (2.40%) and reversed:

| Component | H2 delta |
|---|---:|
| Score | `-0.00018437` |
| 1-NMAE | `-0.00002288` |
| FICR | `-0.00034586` |

July, September, October, and November had negative score deltas. The 2,000
issue-block bootstrap produced:

- score q05: `-0.00058264`
- 1-NMAE q05: `-0.00011139`
- FICR q05: `-0.00109112`
- all-component-positive fraction: `10.6%`

The selected policy passed the Q2 and movement gates but failed every locked
performance and stability gate.

## Interpretation

The API/data bottleneck is resolved. The remaining bottleneck is temporal
non-stationarity: adding an older year weakens the recent power curve, and even
a sparse Q2-positive regime rule does not transfer to H2. Further tuning of
historical weights, disagreement thresholds, or coverage would reuse the locked
failure and is therefore closed.

Reproducible outputs:

- `experiments/kma_year_forward_blend.py`
- `artifacts_final/diagnostics/kma_year_forward_blend_20260726.json`
