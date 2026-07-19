# BARAM 2026 점수 병목 기록 — 2026-07-18

## 현재 공개 기준선

공개 최고 기록은 submission 27, ID `1494670`의
`blend_best_crossg3_traj_meta_finesweep.csv`다.

| 총점 | 1-nMAE | FiCR |
|---:|---:|---:|
| 0.6417471627 | 0.8754733572 | 0.4080209682 |

목표 0.65까지는 총점 `+0.0082528373`이 필요하다.

## 2026-07-18 공개 전이 실패

| ID | 파일 | 총점 | 1-nMAE | FiCR | 최고점 대비 총점 |
|---:|---|---:|---:|---:|---:|
| 1495095 | `submission_scada_g12_controlled_g3_monthmask_meta.csv` | 0.6389115671 | 0.8719628694 | 0.4058602647 | -0.0028355956 |
| 1495096 | `blend_best_crossg3_season_affine1165m300_skip1_6_12.csv` | 0.6329965062 | 0.8691989122 | 0.3967941001 | -0.0087506565 |

두 후보 모두 총점뿐 아니라 1-nMAE와 FiCR가 함께 하락했으므로 폐기한다.

submission 30은 기준선에서 그룹 1·2가 동일하고 그룹 3만 약 74.79%의 행을
양(+) 방향으로 바꿨다. 따라서 전체 총점 하락의 원인을 그룹 3 계절 affine
보정으로 직접 귀속할 수 있다. 그룹 3 기여분으로 환산한 변화는 총점
`-0.0262519695`, 1-nMAE `-0.0188233350`, FiCR `-0.0336806043`이다.

issue-cycle 감사에서는 train/test가 각각 367/365개 cycle이고 lead hour가 모두
12–35시간으로 같아, 이 실패를 issue timing 누수로 설명할 증거는 발견되지 않았다.
반면 2024→2025 기준 예측 평균 비율은 그룹 1 `0.3378→0.3898`, 그룹 2
`0.3557→0.3996`, 그룹 3 `0.3023→0.3476`으로 이동했다. GFS 100m 풍속도 평균
`+0.493 m/s`, gust는 `+0.590 m/s` 높았다. 현재 근거상 주원인은 단일 2024 연도에서
여러 affine를 고른 선택 편향, 2025 기상·계절 분포 이동, 75–100% 행에 대한 외삽이다.

## 폐기된 가설

- 2024 exact OOF 또는 locked H2에서 얻은 양(+) affine 보정량이 2025 공개 구간에도 전이된다는 가설
- 그룹 3의 넓은 행 범위에 scale/offset을 적용하면 FiCR 병목이 해소된다는 가설
- 월 마스크만으로 affine 보정의 전이 위험을 충분히 통제할 수 있다는 가설
- 로컬 macro 0.65 초과를 공개 0.65 가능성으로 직접 투영하는 판단 방식

이 결과는 역방향 affine를 곧바로 시도하라는 근거가 아니다. 공개 점수를
연속적인 보정 최적화의 목적함수로 사용하면 leaderboard 과적합이 된다.

## 공개 점수와 독립적인 구조 실험

두 개의 비-affine 분기를 exact rolling finesweep OOF 기준선에서 추가 검증했지만
제출 승격 기준을 통과하지 못했다.

- issue-cycle trajectory residual은 locked 그룹 3 점수를 `+0.0009954` 높였지만,
  최악 월이 `-0.0000063`이었고 2025 test-analogue 25% 구간에서는
  `-0.000005`로 전이 근거가 없었다.
- 조건부 residual distribution의 locked 그룹 3 점수는 `-0.0012619`,
  FiCR는 `-0.0025441`, 최악 월은 `-0.0046995`, issue-cycle bootstrap 양수 비율은
  `0.102`였다. 동일 action grid의 mean-residual control도 FiCR와 bootstrap gate를
  통과하지 못했다.

따라서 두 분기 모두 서비스 DB에서 `rejected`로 닫았고 후보 CSV를 만들지 않았다.
이는 현재 병목이 단순한 affine scale이나 평균 residual이 아니라, 2025 분포 이동에서
FiCR 경계의 방향까지 안정적으로 설명하는 새 신호가 필요하다는 근거다.

## 제출 후보 운영

`submissions/`에는 현재 다음 두 파일만 활성 상태로 둔다.

1. `blend_best_crossg3_traj_meta_finesweep.csv` — 공개 최고 기준선
2. `results.csv` — 공개 결과 원장

공개 실패 파일은 `submissions/archive/public_rejected_2026-07-18/`, 같은 방향의
미제출 affine 후보는 `submissions/archive/rejected_affine_family_2026-07-18/`에
복구 가능하게 보관한다.

기각된 spatiotemporal 분기의 재생성 가능한 대형 학습/테스트 텐서 두 개는 제거해
`113,691,379 bytes`를 정리했다. 최종 예측, 검증 보고서, provenance, 공통 feature
cache와 exact lineage는 보존했으며 상세 내역은
`artifacts_final/cleanup_manifest_20260718.json`에 기록했다.

## 수정된 승격 기준

새 후보는 다음 조건을 모두 만족해야 한다.

- affine 또는 전역 사후보정이 아니라 새 정보나 구조적 오차를 설명하는 모델일 것
- 연도·issue-cycle 순방향 blocked 검증에서 일관된 양의 변화가 있을 것
- 그룹별 1-nMAE와 FiCR 중 어느 하나도 의미 있게 악화되지 않을 것
- 월별·day-bootstrap 하위 5% 구간이 허용 범위 안에 있을 것
- 공개 실패 family를 재사용하면 변경 행 범위와 방향을 크게 축소하고 독립 근거를 제시할 것
- 제출 파일은 승격 이후에만 `submissions/`에 둘 것

다음 탐색은 SCADA/NWP의 issue-safe 구조적 residual, regime gating, 예측 다양성이
있는 OOF 앙상블을 우선한다. 공개 점수는 가설의 사후 판정에만 사용한다.
