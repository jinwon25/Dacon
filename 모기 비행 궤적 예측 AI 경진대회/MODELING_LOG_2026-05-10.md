# Modeling Log - 2026-05-10

## Current Public LB

Best public LB remains:

```text
0.6834
outputs/00_submit/submission_best_public_0p6834_boundary_gate.csv
```

Recent OOF-best boundary public checks did not beat it:

| submission | file | public LB |
|---|---|---:|
| submission_22 | seed20260606, cap0.006, apply0.75, gate | 0.6820 |
| submission_23 | seed20260606, cap0.004, apply1.0, gate | 0.6818 |

## Reliable Internal Criterion

Use 5-fold OOF as private-LB proxy. Do not tune on public LB.

Current best completed 5-fold boundary OOF:

```text
outputs/02_boundary_oof/
```

| rank | config | gate OOF |
|---:|---|---:|
| 1 | cap0.004, apply1.0, seed20260606 | 0.6619 |
| 2 | cap0.004, apply0.75, seed20260606 | 0.6618 |
| 3 | cap0.005, apply0.75, seed20260606 | 0.6615 |

Boundary tuning is now saturated. It improves selector OOF only from about `0.6572` to `0.6619`.

## Experiments Run

### LightGBM Candidate Ranker

Script:

```text
src/stack_candidate_ranker.py
```

Results:

| target | OOF gate |
|---|---:|
| utility | 0.6423 |
| hit | 0.6534 |
| neg_err | 0.6335 |

Decision: failed. Do not use as standalone selector.

### Row-Level Residual Corrector

Script:

```text
src/residual_row_corrector.py
```

Best result over projected selector base:

```text
base selector projected: 0.6572
row residual corrected: 0.6577
```

Decision: too small; below boundary OOF.

### Boundary Prediction Blend

Best blend did not beat direct best boundary:

```text
best single boundary gate: 0.6619
best pair blend:           0.6619
```

Decision: no useful ensemble gain among current boundary variants.

### Extra Candidate Probe

Adding `perp=-0.40` candidates increases candidate oracle:

```text
base oracle:     0.7188
expanded oracle: 0.7255
```

But prior-only scoring with expanded candidates lowered OOF:

```text
old selector soft:       0.6561
expanded prior+residual: 0.6547
```

Decision: extra candidates need selector retraining; they cannot be appended with prior only.

### Selector Retraining

Completed:

```text
outputs/06_selector_experiments/attn_gru_hier_adapter_seed20260615/
```

Result:

```text
projected OOF: 0.6525
```

Decision: failed. Hier-family gate + latent physics adapter was too restrictive.

Timed out before report:

```text
outputs/06_selector_experiments/attn_gru_extra_perpm040_seed20260616/
outputs/06_selector_experiments/attn_gru_pretrain_heavy_lightfine_seed20260617/
```

No completed report was produced, so do not use these as evidence.

## Next High-Leverage Direction

To target public `0.69`, the next real improvement must come from selector/candidate training, not boundary caps.

Recommended next run on GPU or with longer timeout:

1. Expand candidates with only the most useful extra family:

   ```text
   d1=1.98, perp=-0.40, par in {0.65, 0.75, 1.15}
   ```

2. Train default attn-GRU, not hier/adapter.

3. Keep fine-tuning weak or use pretrain-best checkpoint if fine-tuning degrades fold hit.

4. Evaluate by 5-fold OOF first; submit only if OOF beats `0.6619` by a meaningful margin.

Minimum bar for another public submission:

```text
5-fold OOF gate >= 0.6640
```

Anything below that is unlikely to justify using another public submission.

## 2026-05-10 Late: perp=-0.40 Candidate Expansion Setup

후보 3개 추가 (turn 족, 총 27 → 30):

```text
frenet_par065_perp_neg040  (1.98, 0.65, -0.40)
frenet_par075_perp_neg040  (1.98, 0.75, -0.40)
frenet_par115_perp_neg040  (1.98, 1.15, -0.40)
```

`src/pipeline.py` 만 수정. py_compile 통과, 모듈 로드 시 `inline selector loaded: 30 candidates` 확인.

영향: `outputs/selector_full/` 의 score bank 는 27-cand 기준이라 새 코드와 불호환. 새 selector 는 `outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/` 에 학습 후 OOF 기준 통과 시 정식 경로로 승격.

실행 가이드:

```text
NEXT_RUN_2026-05-10_perpm040.md
```

Acceptance:

- Phase 1 (selector): oracle ≥ 0.7220 AND selector gate OOF ≥ 0.660
- Phase 2 (boundary): OOF gate ≥ 0.6640 → 공개 제출 가능

## 2026-05-11 Phase 1 Result (CPU, seed 20260618)

학습 환경: 로컬 CPU (16 cores / 12 torch threads), elapsed 3589.5s ≈ 60 min.

| metric | value | vs old (27-cand, seed 20260506) | gate | result |
|---|---:|---:|---:|:---|
| candidate_oracle | 0.7250 | +0.0062 (vs 0.7188) | ≥ 0.7220 | PASS |
| attn_gru soft OOF | 0.6576 | −0.0112 (vs 0.6688) | — | — |
| attn_gru argmax_soft_gate OOF | 0.6576 | — | ≥ 0.660 | **FAIL** |
| ensemble projected OOF | 0.6582 | −0.0104 (vs 0.6686) | — | — |

추가 관찰:

- gate 하이퍼파라미터 탐색이 `margin_threshold=Infinity, argmax_rate=0.0`로 collapse → gate가 soft와 동일하게 수렴.
- 새 후보 `frenet_par115_perp_neg040`은 학습된 selector top1으로 선택됨 (fold0 val 432/2020). par065/075는 prior_top4에 거의 등장 안 함.
- full-fit `pretrain=19 epochs, finetune=12 epochs`.

진단: oracle 상한은 확장됐지만 selector가 새 후보를 통해 신뢰 가능한 결정을 내리지 못함. seed 또는 후보 분포 영향 가능. NEXT_RUN 문서 결정 규칙대로 seed 변경 2차 시도 진행.

Phase 2 (boundary OOF sweep) 미실행 — selector gate 0.6576에서 boundary 적용 시 0.65 미만 추정, 0.6640 acceptance 미달.

## 2026-05-11 Phase 1 Retry (CPU, seed 20260619)

elapsed 4658s ≈ 78 min.

| metric | seed 20260618 | seed 20260619 | old (27-cand, seed 20260506) |
|---|---:|---:|---:|
| oracle | 0.7250 | 0.7250 | 0.7188 |
| soft | 0.6576 | **0.6569** | 0.6688 |
| projected | 0.6582 | **0.6569** | 0.6686 |
| argmax_soft_gate | 0.6576 | **0.6569** | 0.6688 |

두 seed 간 차이 0.0007 — 시드 노이즈가 아닌 후보 확장 자체의 구조적 영향. 30-cand 변형 모두 acceptance 미달 → 확장 축소 결정.

## 2026-05-11 Candidate Subset Reduction (28-cand, par115 only)

`par065`/`par075`는 어떤 fold prior_top4에도 등장 안 함, `par115`는 fold1/4/5의 prior_top4에 등장 → 후자가 신호 후보, 전자는 노이즈.

후보 축소: `frenet_par065_perp_neg040`, `frenet_par075_perp_neg040` 제거, `frenet_par115_perp_neg040` 유지. 총 28 candidates.

elapsed 4536.6s ≈ 76 min, seed 20260618.

```text
outputs/06_selector_experiments/attn_gru_par115only_seed20260618/
```

결과:

| metric | 28-cand par115only | gate | result |
|---|---:|---:|:---|
| oracle | 0.7242 | ≥ 0.7220 | PASS |
| soft | 0.6564 | — | — |
| projected | 0.6568 | — | — |
| argmax_soft_gate | **0.6564** | ≥ 0.660 | **FAIL** |

## 2026-05-11 Final Summary of perp=-0.40 Expansion

| variant | oracle | soft | projected | gate |
|---|---:|---:|---:|---:|
| 27-cand baseline (seed 20260506) | 0.7188 | **0.6688** | 0.6686 | **0.6688** |
| 30-cand seed20260618 | 0.7250 | 0.6576 | 0.6582 | 0.6576 |
| 30-cand seed20260619 | 0.7250 | 0.6569 | 0.6569 | 0.6569 |
| 28-cand par115only seed20260618 | 0.7242 | 0.6564 | 0.6568 | 0.6564 |

진단: perp=-0.40 후보 추가는 oracle을 +0.005~0.006 올리지만 selector 출력을 −0.012로 더 크게 떨어뜨림. 3개 구성 모두에서 동일 패턴 관찰 → 시드/구성 노이즈가 아닌 구조적 영향. 기존 turn 족(par110/par120 with perp=-0.20) 영역과 겹쳐 selector confidence 분산만 키운다는 해석.

결정: 후보 확장 방향 닫음. 코드 27-cand로 롤백 (2026-05-11). 다음 시도는 다른 방향에서 선택.

남은 옵션 (메모):

- 다중 모델 selector 앙상블 (현재는 attn_gru 단독): TCN/BiGRU 추가 학습 후 앙상블
- selector hyperparameter sweep (pairwise/distill/prior strength)
- perp=-0.40 외 다른 차원 후보 확장 (jerk, 더 큰 time_scale 등)
- 잔차/보정 측 강화 (residual_row_corrector 재실험, boundary cap/apply 미탐색 조합)
- 공개 LB는 0.6834 유지

학습 결과 디렉토리는 보존:

```text
outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/
outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260619/
outputs/06_selector_experiments/attn_gru_par115only_seed20260618/
```

