# 초기 검증의 완료·미완료 항목

엄격한 OOF 실행·기존 모델 처리·연결 증거 등 아직 해결되지 않은 항목과 통과한 항목을 구분합니다. 개별 감사의 PASS가 전체 모델 검증 통과를 뜻하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Research debt audit

- **FAIL** `exact V2 row-level component OOF`: strict fixed-fit nested OOF was not completed; frozen cache is diagnostic only.
- **FAIL** `outer early stopping removed and executed`: implementation exists, but full four-fold execution exceeded the local runtime window.
- **PASS** `2024 virgin holdout wording`: reports call 2024 a locked confirmation and document repeated reuse.
- **PASS** `V4/V5 clean ablation interpretation`: branch precedence is documented; the public results are not treated as clean causal ablations.
- **FAIL** `RF categorical ID treatment`: the new structural pilot does not validate the legacy RF branch; RF remains a V2 component only.
- **PASS (screen only)** `fair CatBoost comparison`: categorical-safe CatBoost used a fixed 350 iterations and no 2024 eval_set; its 2024 screen Brier was 0.248273754, so it is not promoted.
- **PASS** `legacy target encoder isolation`: permutation/current-label/future-label audit; excluded from Top-1100 recipes.
- **PASS** `Trackman confidence is identity accuracy`: stability is distinguished from identity; low confidence remains on fallback.
- **PASS (aggregate only)** `hand mapping target-free two-way comparison`: season hand-share audit consistently selects 1=Left, 2=Right; player-level mapping and physical profile promotion remain blocked by placebo failure.
- **PASS** `plate coordinate availability`: schema has no plate_x/plate_z; no plate-location model is claimed.
