# 1158.0746 기준 1170 후속 연구 — 2026-08-22

> 후속 v61 exact gate OOF, v62 직교 후보, 공개 자료·최신 문헌 감사 결과는
> [`target1170_research_update_20260822.md`](target1170_research_update_20260822.md)를
> 최신 기준으로 사용한다.

## 결론

현재 유지할 모델은 Public **1158.0745556751**의 TrackMan-ASOF gate 버전이다.
이번에 감사·구현한 22개 후속 후보 중 패키징 게이트를 통과한 후보는 없다. 점수
상승을 과장하지 않고 현재 champion을 보존한다.

전달용 기준은 `artifacts/standalone_champion_1158/standalone_champion_1158.zip`이다.
이 ZIP은 `v22 → v25 → v26 → 0819` 부모 ZIP을 순서대로 요구하지 않으며,
`script.py`, `requirements.txt`, `model/`만으로 실행된다.

| 항목 | 결과 |
|---|---:|
| 현재 Public | **1158.0745556751** |
| 목표 1170까지 | **11.9254443249** |
| 단일 ZIP SHA-256 | `16DAFD86C00DDE5F93D88FF6BE606F6832A7B5A3F8B546CEA299928B78036526` |
| ZIP 크기 | 32,609,803 bytes |
| 정적 금지 연산 | 0개 |
| singleton/full 최대 차이 | `5.55e-17` |
| shuffled/full 최대 차이 | `5.55e-17` |
| partitioned/full 최대 차이 | `0` |

## 기준선 계보 감사

팀 `main`의 실제 최종 payload에서 `model/v25_postbreak_anchor_spec.json`의
`blend_eta`는 **0.15**다. 반면 v29~v57 후속 실험은 대부분
`v27_parent`의 eta **0.10**을 기준으로 후보를 평가했다. 따라서 당시의 양수 local
gain은 1158 모델 위의 증분으로 해석할 수 없다.

최종 0819 gate는 v26 예측 뒤 R_ANCHOR에서만 다음 shrink를 적용한다.

```text
gate = min(0.03, 0.03 × TrackMan pitcher confidence × count pressure)
prediction = prediction + gate × (ASOF prior - prediction)
```

TrackMan profile과 ASOF prior는 공식 train으로 미리 동결되어 있다. 최종 gate의
연도별 OOF profile은 저장소에 남지 않아 이번 local 감사에서는 eta=0.15 v26
부모를 공통 기준으로 사용하고, 배포 시에는 검증된 gate를 그대로 유지했다.

## 새 감사와 실험

### v58 — 기존 v31~v57의 eta=0.15 재기준화

기존 후보가 만든 확률 이동량 `candidate - eta0.10_parent`를 그대로 보존해
eta=0.15 부모 위에 더했다. 2024 정답으로 재튜닝하지 않았다. 20개 중 통과 0개다.

| 후보 | 2024 전체 gain | 양수 월 비율 | 최악 월 | 2024 후반 복제 | 판정 |
|---|---:|---:|---:|---:|---|
| v50 low-rank pitcher context | +2.3072 | 37.5% | -10.7140 | +0.2284 | 탈락 |
| v53 factorization offset | +1.8216 | 50.0% | -10.8594 | 없음 | 탈락 |
| v57 independent blend | +1.2597 | 50.0% | -9.1762 | +0.4875 | 탈락 |
| v56 shared-horizon FM | +1.0181 | 75.0% | -4.7126 | 없음 | 탈락 |
| v54 horizon factorization | +0.9861 | 71.4% | -0.8051 | 없음 | 탈락 |

가장 큰 평균 gain도 월별·도메인 안정성이 부족했다. v50/v57을 그대로 패키징하면
평균 개선 하나를 위해 큰 월별 하방을 감수하게 된다.

### v59 — spline-logistic 모델 주변화

R_ANCHOR 직접모형의 규제 `C={0.01, 0.1, 1}` 예측을 확률·logit 공간에서
주변화했다. 모델 간 불일치가 큰 행에는 더 작은 혼합률을 적용하되, 임계값은
선택 train에서 동결했다.

| 축 | gain | 양수 월 비율 | 최악 월 | R_ANCHOR gain |
|---|---:|---:|---:|---:|
| 2023 후반 선택 | +15.0351 | 100% | +11.0747 | +86.4652 |
| 2024 전체 감사 | +0.2898 | 57.1% | -2.9087 | +1.6452 |
| 2024 후반 복제 | **-0.3524** | 50.0% | -0.9882 | **-2.0155** |

선택축 이득이 외부 시간축으로 전이되지 않아 탈락했다. 불확실성 주변화 자체는
합리적이지만 현재 세 모델이 같은 데이터와 특성을 공유해 오차 다양성이 작다.

### v60 — train 중심 확률 분산 보정

공개 구현에서 보고된 fixed-center spread scaling을 현재 부모에 이식했다. 중심은
평가 데이터 평균이 아니라 각 source train의 target 평균으로만 고정했고,
linear/logit, alpha 0.95~1.15, ALL/R_CORE/R_ANCHOR/F를 비교했다. 선택 결과는
F에 대한 linear alpha `0.95`였다.

| 축 | gain | 양수 월 비율 | 최악 월 | F gain |
|---|---:|---:|---:|---:|
| 2023 후반 선택 | +67.1910 | 100% | +72.7363 | +777.0321 |
| 2024 전체 감사 | **-0.3574** | 62.5% | **-7.3589** | **-3.0373** |
| 2024 후반 복제 | +1.1807 | 66.7% | -0.9054 | +9.5068 |

2024 전체에서 반전되어 탈락했다. 특히 F의 연도별 regime 이동 때문에 한 시즌의
분산 보정 방향을 다음 시즌에 고정하는 전략은 위험하다.

## 공식 행 독립성 준수

[DACON 공식 독립 예측 공지](https://dacon.io/competitions/official/236743/talkboard/417123)의
기준을 다음처럼 코드 게이트로 고정했다.

- 한 행의 예측은 그 행의 입력, 공식 train, train으로 미리 만든 artifact만 사용한다.
- 다른 test 행의 평균·분포·빈도·순위·선수/팀/월/경기 집계를 사용하지 않는다.
- test 행 간 rolling, lag, shift, 누적 통계와 파일상 이전 행을 사용하지 않는다.
- 전체·한 행씩·셔플·분할 실행에서 같은 행의 확률이 `1e-12` 이내로 같아야 한다.
- 평가 파일 크기나 분포로 alpha, center, threshold를 다시 정하지 않는다.

단일 패키지 정적 감사는 `groupby`, `rolling`, `expanding`, `ewm`, `shift`,
`diff`, `rank`, `quantile`, `value_counts`, 누적 연산 등 고위험 호출을 거부한다.
현재 payload는 정적·동적 검사를 모두 통과했다. v58~v60도 test 집계 없이 한 행의
기존 확률과 train 고정 상수만 사용하는 row-local 변환이다.

## 1170을 위한 우선순위

현재 점수에서 필요한 `+11.9254`는 같은 spline 혼합률이나 scalar calibration의
미세 조정만으로 얻기 어렵다. 다음 순서가 비용 대비 합리적이다.

1. **최종 1158 전체 OOF 복원**: fold별 TrackMan profile과 ASOF bank를 train
   fold만으로 다시 만들고 0819 gate까지 포함한 exact temporal OOF를 생성한다.
   지금은 gate 전 부모로만 후속 후보를 비교하므로 작은 개선의 신뢰도가 제한된다.
2. **팀원 모델의 OOF 기반 직교 앙상블**: 세 팀원의 row-level OOF와 추론 확률을
   같은 ID 순서로 모아 비음수 constrained blend를 학습한다. 평균 gain뿐 아니라
   `연도×월×R/F×R_ANCHOR` 최악 집단을 함께 제한한다. Public 점수만으로 가중치를
   역산하지 않는다.
3. **현재 상태를 직접 예측하는 독립 모델**: 최근 시즌 전용 CatBoost/LightGBM과
   투수·타자 partial pooling 모델을 별도로 학습한다. 기존 spline과 다른 특성·손실·
   표본 창을 사용해 잔차 상관을 낮추는 것이 목적이다.
4. **환경 강건 최적화**: 시즌, 월, R/F, R_ANCHOR를 environment로 둔 Group DRO
   또는 worst-group regularization을 적용한다. F의 2022→2023 급락 때문에 전체
   평균 ERM만으로는 선택축 과적합이 반복된다.
5. **교차적합 calibration**: beta/logit/isotonic 보정은 각 temporal fold의 train
   예측으로만 적합하고 다음 시즌에 고정한다. 2024 label이나 평가 분포로 보정
   중심을 정하지 않는다.

### 승격 게이트

새 후보는 아래를 모두 통과할 때만 standalone 후보 ZIP을 만든다.

- 2024 전체 local gain `>= +5`
- 월별 양수 비율 `>= 75%`, 최악 월 `> -5`
- R_CORE/R_ANCHOR/F 최소 domain gain `>= 0`
- 독립 2024 후반 복제 gain `> 0`이고 모든 월 양수
- parent 대비 평균 오차 상관이 충분히 낮거나, 높은 상관을 상쇄할 명확한 증거
- 정적·singleton·shuffle·partition 행 독립성 검사 통과

게이트 통과 전에는 현재 champion을 덮어쓰거나 DACON 제출 파일을 만들지 않는다.

## 근거 자료

- [DACON 평가 데이터 독립 예측 원칙 공지](https://dacon.io/competitions/official/236743/talkboard/417123)
- [Gneiting & Raftery, Strictly Proper Scoring Rules](https://doi.org/10.1198/016214506000001437): Brier 계열 점수에서 확률의 정직한 calibration과 sharpness를 함께 봐야 하는 근거
- [Ovadia et al., Can You Trust Your Model's Uncertainty Under Dataset Shift?](https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html): 분포 이동에서 post-hoc calibration의 한계와 모델 주변화 검토 근거
- [Sagawa et al., Distributionally Robust Neural Networks for Group Shifts](https://openreview.net/forum?id=ryxGuJrFvS): season/domain worst-group 목적의 근거
- [Kull et al., Beta Calibration](https://proceedings.mlr.press/v54/kull17a.html): 확률 보정 후보군의 1차 문헌
- [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html): 독립 calibration fold와 단순 scaling 비교 근거
- [hoo743-ui/LG_Aimers09](https://github.com/hoo743-ui/LG_Aimers09): train 고정 중심 spread scaling과 CatBoost 경로를 확인한 공개 구현. 점수·모델·분포가 다르므로 아이디어만 재검증했고 직접 채택하지 않았다.

## 재현 명령

```powershell
python -m pytest -q `
  tests/test_package_standalone_champion.py `
  tests/test_v58_eta15_rebase_audit.py `
  tests/test_v59_anchor_model_marginalization.py `
  tests/test_v60_frozen_spread_scaling.py

python -m src.package_standalone_champion `
  --payload-dir artifacts/standalone_champion_1158/payload `
  --output artifacts/standalone_champion_1158/standalone_champion_1158.zip `
  --smoke-test-csv data/test.csv
```

v58~v60의 대형 cache는 최종 판단을 이 보고서에 고정한 뒤 삭제할 수 있다. 코드와
테스트가 동일 결과를 재생성하며, 현재 릴리스 실행에는 해당 cache가 필요하지 않다.
