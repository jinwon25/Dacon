# 1170 챔피언 OOF evidence

## ⚠️ 공개 금지 — 공식 팀원 전용

이 디렉터리의 NPZ에는 **공식 `train.csv`의 정답 레이블(`target`)과 선수 식별자
(`pitcher_id`, `batter_id`)가 포함**된다.

- 공식 DACON 팀원에게만 공유한다.
- 저장소를 공개로 전환하거나 공식 팀원이 아닌 계정을 초대하면 안 된다.
- 외부 재배포, 공개 채널 업로드, 제3자 공유를 금지한다.

## 무엇인가

현재 챔피언 `submit_v148.zip`(Public **1170.3014697177**) 계보의 full-2024 시점별
예측 evidence다. 팀원이 자기 모델과 **잔차 상관·오차 공분산**을 계산하고 블렌드
가능성을 검토할 수 있도록 `artifacts/oof_champion_1161/`과 **동일한 계약 스키마**로
만들었다.

## 어떻게 만들었나

v147 blend 감사는 v142/v138/v124 축을 저장했지만 선택 가중치가 `0.05`였다. 실제 배포
챔피언 v148은 `blend_weight_toward_v138 = 0.15`이므로, 저장된 부모 축에서 다음 식으로
재구성했다(재학습이 아니라 정확한 재구성이다).

```
parent = clip(v142_full_2024 + 0.15 × (v138_full_2024 − v142_full_2024), 0.001, 0.999)
```

### 검증 (`manifest.json`에 수치 보존)

빌더 `src/champion/v148_oof_contract_bundle.py`는 아래를 통과하지 못하면 **파일을
쓰지 않고 예외를 던진다.**

| 검증 | 결과 |
|---|---|
| `active_mask` == 계약 `domain3 == "R_CORE"` (178,729행 원소 단위) | 통과 |
| `late_2024` == `full_2024[game_month >= 8]` (76,896행) | 통과 |
| `raw_index` 순증가, `season` 전부 2024 | 통과 |
| 재구성 공식이 v147 공표 `gain_vs_v124` 재현 | `7.198934146857027` 일치 |
| 재구성 공식이 v147 공표 `late_gain_vs_v124` 재현 | `12.964273655657848` 일치 |

추가 확증: 재구성한 weight-0.15 축의 `late_gain_vs_v124 = 13.776562841982809`은 v152
감사가 공표한 weight-0.15 값과 정확히 일치한다. full-2024 값(`7.662252`)이 v152의
`8.643070`과 다른 것은 v152가 5월 행을 v124로 되돌렸기 때문이며, 5월은 late 축(8월
이후)에 포함되지 않으므로 late가 일치하고 full만 달라지는 것이 정확히 예상되는 결과다.

## NPZ 스키마 — `v148_full_2024.npz` (253,507행)

| 배열 | 의미 |
|---|---|
| `raw_index` | 공식 `train.csv`의 0-based 원본 행 인덱스 |
| `target` | 해당 행의 `control_success` |
| `parent` | **v148 챔피언 예측 확률** |
| `exact_mask` | exact component parity 여부 — exact 비교는 이 행만 사용 |
| `season`, `game_month` | int16 |
| `domain3` | `R_CORE` / `R_ANCHOR` / `F` |
| `pitcher_id`, `batter_id` | int64 |
| `v142_full_2024` | 계보 비교용: 브릿지 이전(weight 0) |
| `v138_full_2024` | 계보 비교용: 브릿지 목표(weight 1) |
| `v124_full_2024` | 계보 비교용: v148 감사의 baseline |

late 축은 별도 파일로 두지 않았다. `game_month >= 8`로 직접 슬라이스하면 v147/v152가
사용한 late 축과 정확히 일치한다.

## 중요한 한계 — 반드시 읽을 것

1. **2024는 독립 holdout이 아니다.** 반복 개발에 사용된 `development_contaminated` 축이다.
   `docs/PROJECT_STATUS.md`가 명시하듯 이를 독립 holdout으로 부르면 안 된다.
2. **모든 과거 시즌을 재학습한 단일 cross-fitted OOF가 아니다.** 고정된 부모 축에서
   배포 가중치로 재구성한 evidence다.
3. **독립 Public 성능 추정이나 블렌드 weight의 단독 근거로 쓰면 안 된다.**
   2026-08-23 v154 사례에서, 로컬 다년도 gain이 `+11.4~+17.6`이고 모든 게이트를
   통과했음에도 실제 Public은 `1168.4038526829`로 **부호가 반전**했다.
4. 이 저장소 실측 이력상 부호 반전을 가장 잘 예측한 신호는 축 개수나 gain 크기가 아니라
   **cluster bootstrap 하한이 0을 넘는지**였다. 자세한 근거는
   `reports/target1180_transfer_meta_audit_20260823.md`.

## 사용해도 되는 용도

- 잔차 상관·오차 공분산 진단
- 계보 재현 확인
- 팀원 모델과의 직교성 평가

각 배열의 `exact_mask`, `domain3`, `season`, `game_month`를 반드시 보존한 채로 사용한다.

## 포함하지 않은 것

`artifacts/external_hoo_lg_aimers09/`는 공개 저장소
[hoo743-ui/LG_Aimers09](https://github.com/hoo743-ui/LG_Aimers09)를 clone한 **제3자
작업물**이며, 아이디어 참고용으로만 사용했다. 이 번들에 포함하지 않는다.
