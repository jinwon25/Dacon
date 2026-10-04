# 초기 v2 패키지·계보 감사

초기 기준 모델의 해시·구성과 배치 독립성, 재현 계보를 점검한 기록입니다. 최종 v345 결과나 현재 실행 패키지로 읽지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Champion v2 audit

- `submit_v2.zip` SHA-256: **verified**
- SHA-256: `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7` (64 hex characters)
- ZIP size / extracted size: `8,880,825` / `10,933,170` bytes
- Members hashed: **12**
- Baseline commit: `9b0cb8401336916f510669021005e48b4db2c92a`; present check: `present`
- Current branch/HEAD: `codex/baram-2026` / `889b1b83ccd89b84295ac6e593984f4f84aebb95`
- Public record: submission 39023, `submit_v2.zip`, score **763.2665303697**.

## Replayed recipe

The package first blends engineered LightGBM (35%) and official RandomForest (65%) with the frozen train-only calibration. For `game_type=R`, the half-life-1 recency RF replaces the RF component. The resulting candidate is blended with the rolling-damped Trackman LightGBM at 5%. The package contains an `if r_only_rf_model ... elif recency_rf_model` branch; consequently v4/v5 cannot identify the causal contribution of the v2 R-recency or Trackman terms.

This manifest is immutable for this cycle; the original ZIP is not overwritten or reconstructed.
