# v104 Public 결과와 새 1162 챔피언

## 결과

- 제출 ID: `60626`
- 제출 파일: `submit_v104_probe.zip`
- 제출 시각: `2026-08-22 23:06:48 KST`
- Public: **1162.6302840289**
- 직전 v84 대비: **+1.4282239867**
- 1170까지: **7.3697159711**
- 공식 실행 시간: `48초`
- 확인 당시 순위/누적 제출: `13위 / 30회`

제출 전 로컬 promotion gate는 통과하지 못했지만 사용자가 명시적으로 승인한 1회 Public probe에서 양의 전이가 확인됐다. 기존 1161 챔피언 ZIP은 보존하고, 제출한 바이트와 같은 파일을 `artifacts/standalone_champion_1162/standalone_champion_1162.zip`으로 승격했다.

## 개선 원인 해석

v104는 v84의 F-route 개선을 그대로 보존하면서 R_CORE에만 두 종류의 독립 보정을 추가했다.

1. full-2023과 late-2024에서 각각 학습한 rank-16 pairwise FM correction의 평균을 logit 공간에서 `0.10` 적용했다.
2. 조건부 LightGBM과 동일 구조의 baseline LightGBM 예측 차이를 확률 공간에서 `0.10` 적용했다.
3. count, pitcher history, platoon 중 사전 고정한 안전 조건을 두 개 이상 만족하는 행만 채택했다.

따라서 이번 상승은 전체 확률의 단순 보정이 아니라, 기존 R_CORE 오차 중 source 간 방향이 비교적 일치한 세부 상황만 수정한 효과로 해석한다. full-2024 기대 gain `+3.2162` 대비 실제 Public gain은 약 `44.4%` 전이됐으며, 1170까지 남았던 격차의 약 `16.2%`를 줄였다.

## 위험과 다음 원칙

- full-2024 worst month `-6.6440`, crossed bootstrap p05 `-2.476`, White Reality Check `p=.117` 위험은 사라진 것이 아니다.
- 이번 Public 결과로 같은 FM/conditional 세기, 안전 범주, majority threshold를 다시 고르면 리더보드 과적합이 된다. 동일 계열 재튜닝은 중단한다.
- 다음 연구는 새 1162 챔피언을 exact parent로 재기준화하고, Public 결과와 독립적으로 정의된 모델·피처·보정 축만 평가한다.
- 팀원 독립 exact OOF와 2025 예측은 계속 없는 것으로 가정한다.

## 재현

```powershell
python -m pytest tests/test_v109_v104_public_probe.py tests/test_v104_source_stability_mask.py -q

python -m src.submit_dacon `
  artifacts/standalone_champion_1162/standalone_champion_1162.zip `
  --project-dir . `
  --credentials-file "<DACON 자격증명 .env 경로>"
```

고정 SHA-256은 `0C3B6A9D88D31D9AC32FB43FFD642FA07BFCC3367699FA71841E35188FBDA0BA`이다. 실제 재제출은 하지 말고 위 명령의 기본 dry-run으로 구조와 자격증명만 확인한다.
