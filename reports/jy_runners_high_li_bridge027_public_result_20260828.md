# JY Runners/High-LI Bridge 후속 Public 결과

기록 시각: `2026-08-28 03:45 KST`

## 결과 요약

| 항목 | 기준 main submit | jy_runners_gate | jy_runners_high_li_bridge027 |
|---|---:|---:|---:|
| Public | 1172.0772380321 | 1172.1199939043 | **1172.1373858439** |
| 기준 대비 | - | +0.0427558722 | **+0.0601478118** |
| 직전 대비 | - | +0.0427558722 | **+0.0173919396** |

- 기준 main submit: `submissions/releases/v167/submit_v167.zip`
- 최종 전달 파일: `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip`
- SHA-256: `4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E`
- 제출 시각: `2026-08-28 03:38:21 KST`
- 확인 Public: `1172.1373858439`
- 1180까지 남은 gap: `7.8626141561`

## 채택한 조합

최종 조합은 현재 main submit 값을 먼저 재현한 뒤, 아래 조건에서만 더 공격적인 조합을 덮어쓴다.

```text
active = R_CORE and (num_runners_on > 0 or li >= 1.5)
```

active 행에서는 다음 조합을 사용한다.

```text
effective_bridge = parent + 1.2 * (bridge025 - parent)
output = 0.84 * effective_bridge + 0.16 * H1 + 0.50 * C3_recent025
```

비활성 행은 기준 main submit과 동일하게 보존한다. 이 구조는 테스트 전체 통계나 test 행 간 집계를 쓰지 않고, 각 행의 game state만 사용한다.

## 순차 실험 판단

처음에는 여러 후보를 독립 제출 후보처럼 나누지 않고, 실제로 오른 `jy_runners_gate`를 기준으로 순차 확장했다.

1. `runners_on` 게이트
   - Public `1172.1199939043`
   - 기준 대비 `+0.0427558722`
   - multi-axis audit에서 가장 먼저 채택한 row-local gate.

2. `high_li` 추가
   - OOF 순차 감사에서 locked 2024 gain이 `0.483469 -> 0.536002`로 증가.
   - worst month도 `-0.053429 -> -0.005010`으로 완화.
   - 채택.

3. `pressure_count` 추가
   - coverage는 넓어지지만 locked 2024 gain이 `high_li` 단독보다 낮았다.
   - 제출 5회 제한 기준으로 탈락.

4. bridge 강도
   - `runners_or_high_li_bridge027`이 locked 2024 gain `0.638287`로 grid 내 최고.
   - 최종 채택.

## 구조 감사

최종 후보 감사 파일: `artifacts/jy_next_variants_20260828_01/audit_jy_runners_high_li_bridge027.json`

요약:

| 검사 | 결과 |
|---|---:|
| 구조 감사 | 통과 |
| 보호 행 최대 차이 | `0.0` |
| 후보 active fraction | `0.625` |
| 이전 runners 후보 active fraction | `0.375` |
| 새로 열린 fraction | `0.25` |
| 후보 실행 시간, 3600행 probe | `9.723s` |

검증 probe에서는 R_CORE, 주자 있음, high-LI, 비정규 경기, team-13 anchor 케이스를 섞어 넣었다. 최종 후보는 active가 아닌 행에서 기준 main submit과 완전히 동일했다.

## 재현 방법

후보 생성:

```powershell
python scripts\build_jy_next_variants.py
```

순차 OOF 감사:

```powershell
python scripts\audit_jy_sequential_gates.py
```

최종 패키지 구조 감사:

```powershell
python scripts\audit_jy_next_final.py
```

필요 입력:

- `submissions/releases/v167/submit_v167.zip`
- `artifacts/jy_combo_bridge025_h1w016_20260828_01/submit_jy_combo_bridge025_h1w016.zip`
- `artifacts/oof_champion_1161/v84_full_2022.npz`
- `artifacts/oof_champion_1161/v84_late_2023.npz`
- `artifacts/oof_champion_1161/v84_full_2024.npz`
- `artifacts/oof_champion_1170/v148_full_2024.npz`
- DACON 원본 `data/train.csv`, `data/test.csv`

## 다음 실험 메모

- 이번 결과는 두 번 연속 Public이 올랐지만 폭은 작다. 같은 family에서 더 강하게 여는 실험은 제출 효율이 낮을 수 있다.
- `pressure_count`는 이번 순차 기준에서는 보류한다.
- TrackMan confidence 계열은 2024에서 좋아 보였지만 과거 축 근거가 비어 있어 바로 제출하기에는 위험했다.
- 다음 큰 점프를 노리려면 단순 gate 확장보다, independent model signal 또는 TrackMan-derived pitcher profile을 multi-axis OOF로 먼저 보강하는 편이 낫다.
