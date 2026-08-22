# 1161 챔피언 OOF evidence

이 디렉터리는 Public `1161.2020600422` v84 계보의 시점별 예측 evidence를 공식
DACON 팀원끼리 공유하기 위한 private Git LFS 번들이다. NPZ에는 target과 선수 ID가
포함되므로 저장소를 공개로 전환하거나 공식 팀원이 아닌 계정을 초대하면 안 된다.

## 중요한 한계

이 파일들은 최종 2025 배포 ZIP을 모든 과거 시즌에서 완전히 다시 학습한 단일
cross-fitted OOF가 아니다. v94가 복원 가능한 시점별 v84 analogue를 fidelity label과
함께 저장한 평가 번들이다.

- `v84_full_2022.npz`: `R_CORE`와 `F`의 82.70%만 exact component parity다.
  `exact_mask=true`인 행만 exact 비교에 사용한다.
- `v84_late_2023.npz`: 전 domain v84 analogue지만 평가 시즌과 training max season이
  같아 clean holdout이 아니다.
- `v84_full_2024.npz`: 전 domain v84 analogue지만 2024는 반복 개발에 사용된
  `development_contaminated` 축이다.

따라서 이 번들을 독립 Public 성능 추정이나 블렌드 weight의 단독 근거로 사용하면 안
된다. 잔차 상관·오차 공분산·계보 재현 진단에는 사용할 수 있으며, 각 파일의
`exact_mask`, `role`, `fidelity`를 반드시 보존한다.

## NPZ 스키마

| 배열 | 의미 |
|---|---|
| `raw_index` | 공식 train.csv의 0-based 원본 행 인덱스 |
| `target` | 해당 행의 `control_success` |
| `parent` | 해당 시점의 v84 계보 예측 확률 |
| `exact_mask` | 이 행이 명시된 fidelity 범위에서 exact인지 여부 |
| `season`, `game_month`, `domain3` | 시간·domain 감사 필드 |
| `pitcher_id`, `batter_id` | 선수 단위 block/crossed bootstrap용 ID |
| `common_parent` | 동일 행의 공통 wave0 비교 부모 |

`row_id` 문자열은 중복 저장하지 않고 `raw_index`로 공식 train과 정렬한다. 정렬된
`row_id`의 SHA-256은 `manifest.json`에 있다.

## 검증

```powershell
python -m src.audit_champion_private_artifacts `
  --model-zip artifacts/standalone_champion_1161/standalone_champion_1161.zip `
  --model-manifest artifacts/standalone_champion_1161/standalone_manifest.json `
  --oof-dir artifacts/oof_champion_1161 `
  --train-csv data/train.csv
```

원본 데이터가 없는 환경에서는 `--train-csv`를 생략해 파일 해시·스키마·확률·내부
배열 해시만 확인할 수 있다.
