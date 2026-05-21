# Next Run - 2026-05-10 (perp=-0.40 candidate expansion)

## What Changed

`src/pipeline.py` `CANDIDATES` 리스트에 turn 족 후보 3개 추가 (총 27 → 30):

```text
CandidateSpec("frenet_par065_perp_neg040", 1.98, 0.65, -0.40)
CandidateSpec("frenet_par075_perp_neg040", 1.98, 0.75, -0.40)
CandidateSpec("frenet_par115_perp_neg040", 1.98, 1.15, -0.40)
```

검증:

```text
candidates: 30
new families: ['turn', 'turn', 'turn']
py_compile: OK
```

## Compatibility Note

기존 `outputs/selector_full/oof_selector_scores.npz` 와 `test_selector_scores.npz` 는 27-candidate 기준이라 30-candidate 코드와 호환 불가. `pipeline.py` 의 candidate-name 가드가 mismatch를 감지해 에러를 낸다.

대응: 새 selector를 다른 경로에 학습. OOF가 0.6640 이상이면 그때 `selector_full/` 자리를 차지하도록 승격.

```text
old (frozen, 27 cands): outputs/selector_full/
new (30 cands):         outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/
```

## Phase 1 — Selector Retraining (Colab T4)

`src/pipeline.py` 의 `run_selector_full` 와 동일한 production 설정에서 out-dir 와 seed 만 교체:

```python
# Colab 셀
import sys, importlib
sys.path.insert(0, "/content/drive/MyDrive/<repo>/src")  # 실제 경로로
import pipeline; importlib.reload(pipeline)
import torch
print("cuda:", torch.cuda.is_available(), "candidates:", len(pipeline.CANDIDATES))

selector_out = pipeline.WORK_DIR / "06_selector_experiments" / "attn_gru_extra_perpm040_v2_seed20260618"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline.call_main(pipeline.SELECTOR_MAIN, [
    "--root", pipeline.DATA_ROOT,
    "--out-dir", selector_out,
    "--models", "attn_gru",
    "--folds", 5,
    "--pre-epochs", 14, "--fine-epochs", 10, "--freeze-fine-epochs", 3,
    "--epoch-plus", 10, "--min-epochs", 5, "--patience", 8,
    "--hidden", 96, "--batch", 1024,
    "--lr", 0.001, "--fine-lr-scale", 0.12,
    "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
    "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
    "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
    "--reverse-pretrain", "--norm-real-only",
    "--device", device, "--seed", 20260618, "--log-every", 1,
])
pipeline.write_selector_score_variants(selector_out)
pipeline.print_selector_summary(selector_out)
```

비활성 (실패한 hier-adapter 시도와 분리):

- `--hier-family-gate` 사용 안 함
- `--latent-physics-adapter` 사용 안 함
- `--latent-env-experts` 기본 1
- `--fine-ensemble` 사용 안 함

만약 fine-tuning 후 OOF hit 가 pretrain best 보다 낮으면 `--fine-lr-scale 0.08` 로 다시 한 번 (약한 fine-tune). 두 번째 시도는 seed 20260619 로 분리.

## Phase 1 Decision Rule

리포트 확인 위치:

```text
outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/tcn_gru_selector_report.json
```

확인 키:

- `model_oof.attn_gru.argmax_soft_gate.metrics.hit`  (selector 자체 gate OOF)
- `oof_tcn_gru_ensemble_argmax_soft_gate.metrics.hit` (ensemble gate OOF, 단일 모델이므로 같은 값)
- `candidate_oracle_metrics.hit` (oracle 상한)

기준:

- oracle ≥ 0.7220 (이전 0.7188 대비 의미 있는 확장)
- selector gate OOF ≥ 0.660 (이전 selector ensemble 대비 의미 있는 향상)

oracle 만 오르고 selector gate 가 오히려 떨어지면 selector 가 새 후보를 활용하지 못한 것 → seed 변경/2차 시도.

## Phase 2 — Boundary OOF Sweep (CPU 가능)

Phase 1 결과가 기준을 통과한 경우에만 진행.

```bash
cd <repo>
python src/boundary_oof_sweep.py \
  --selector-out outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618 \
  --out-dir outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/boundary_oof \
  --config 0.004:1.0:20260606 \
  --config 0.004:0.75:20260606 \
  --config 0.005:0.75:20260606 \
  --config 0.006:0.75:20260606 \
  --device cpu
```

OOF 요약 위치:

```text
outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/boundary_oof/boundary_oof_sweep_summary.json
outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618/boundary_oof/README.md
```

## Phase 2 Acceptance Gate

새 boundary OOF gate 최고값이:

- `>= 0.6640` → 공개 제출 후보. 해당 폴드별 boundary 결과로 test 예측 생성 (`--make-test`) 후 제출.
- `0.6619 ~ 0.6639` → 보류. boundary cap/apply 변형을 더 시도 후 재평가.
- `< 0.6619` → 후보 확장 실패. selector 재학습 자체를 다시 시도하거나 다른 방향 탐색.

## Test Prediction Generation (Phase 2 통과 후)

`boundary_oof_sweep.py` 는 fold 평가만 수행하고 test 예측은 만들지 않으므로 별도 호출 필요. 베스트 config 를 골라 `make-test` 모드로 단발 호출:

```python
# Colab 셀
import pipeline as p
selector_out = p.WORK_DIR / "06_selector_experiments" / "attn_gru_extra_perpm040_v2_seed20260618"
out = p.WORK_DIR / "03_submission_candidates" / "perpm040_v2_BEST"
p.call_main(p.BOUNDARY_MAIN, [
    "--root", p.DATA_ROOT,
    "--out-dir", out,
    "--fold", 0, "--folds", 5,
    "--score-bank", selector_out / "oof_selector_scores.npz",
    "--make-test",
    "--test-score-bank", selector_out / "test_selector_scores.npz",
    "--epochs", 1, "--fine-epochs", 1, "--min-epochs", 1, "--patience", 1,
    "--hidden", 64, "--batch", 8192,
    "--lr", 0.001, "--fine-lr-scale", 0.18,
    "--cap", BEST_CAP, "--apply-scale", BEST_APPLY,
    "--device", "cpu", "--seed", 20260606, "--save-val-pred",
])
# submission_boundary_tiny_gate.csv 가 out 안에 생성됨
```

## Promotion (선택, OOF 통과 후)

새 selector 가 production 으로 승격되면:

```bash
mv outputs/selector_full outputs/90_archive/selector_full_27cand_legacy
mv outputs/06_selector_experiments/attn_gru_extra_perpm040_v2_seed20260618 outputs/selector_full
```

이 시점에 boundary 스크립트와 `run_selector_full` 의 기본 경로가 다시 일치한다.

## Logs to Update After Run

- `MODELING_LOG_2026-05-10.md`: 결과 표에 perp=-0.40 expansion 행 추가
- `WORK_LOG_2026-05-10.md`: phase 별 결정 기록
- `outputs/README.md`: selector 승격 시 인덱스 갱신
