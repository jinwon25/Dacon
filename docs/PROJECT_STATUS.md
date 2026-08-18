# 프로젝트 현황

마지막 갱신: `2026-08-17 05:54 KST`

## 현재 champion

> **2026-08-18 정정**: 이 절은 이전에 `submit_v27.zip`을 champion으로 기록했으나 공식 DACON 제출 이력 재대조 결과 오류였다(제출 ID `1535195`는 DACON의 실제 5자리 ID 형식과 다름). 실제 champion은 `submit_v26.zip`이다. 근거: [`../reports/target1170_followup_20260817.md`](../reports/target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

| 항목 | 값 |
|---|---:|
| champion | `submit_v26.zip` |
| Public | **1157.9736407889** |
| 제출 ID | `51773` |
| 확인 당시 순위 | 미확인 (정정 이전 값은 v27 기준) |
| v25 대비 | **+2.1443002480** |
| 1170까지 | **12.0263592111** |
| 1200까지 | **42.0263592111** |
| SHA-256 | `8BE26E156A91C1FA5E9989EEE5043DB5992A960DF35A53D8A9BFEF7E6825D096` |

v26은 v25와 동일한 2024-only spline-logistic 직접확률 모델을 R_ANCHOR에 15%(스크립트 기본값) 혼합한다. Public에서 v25 대비 `+2.1443`이 추가 전이됐고 R_CORE/F는 부모와 동일하다. 10% probe(v27)와 손잡이 routing(v28)은 champion(v26)에 못 미쳤으므로 같은 계열 미세조정은 중단한다.

## champion 계보

```text
v13 exact-ASOF
  → v17 R_CORE pressure EB
    → v19 multi-year state + latent failure-mode routing
      → v20 pooled residual + training-only TrackMan PFD
        → v21 recent-control state + count-context recency EB
          → v22 low-variance domain calibration + row-local ASOF prior
            → v25 post-break R_ANCHOR direct-probability overlay
              → v26 frozen signal weight 15% (champion)
```

## 바로 읽을 문서

1. `reports/target1170_followup_20260817.md`
2. `reports/v28_validation.md`
3. `reports/v24_semantic_eda_20260817.md`
4. `reports/v22_public_result_20260817.md`
5. `reports/dacon_official_compliance_audit_20260816.md`
6. `reports/submissions.csv`

## 구현·재현 파일

```text
src/package_v26_anchor_weight_probe.py
src/v26_exact_diversity_screen.py
src/v28_domain_specialist_screen.py
src/v29_anchor_route_screen.py
src/package_v28_anchor_hand_route.py
src/validate_v28_anchor_hand_route.py
```

최종 모델·OOF·데이터·제출 ZIP은 Git에 올리지 않는다. 로컬 champion ZIP은 프로젝트 루트에 두고 이전 ZIP은 `submissions/history/`로 이동한다.

## 검증 상태

- 전체 테스트: `119 passed`
- v26 245,789행 추론: `56.240초`
- v26 peak RSS: `1,410.7 MB`
- 배치 불변성 최대 오차: `1.11e-16`
- 패키지·규정 게이트: `13/13 passed`
- 공식 Public: `1157.9736407889`

## 다음 연구 원칙

1. v26 R_ANCHOR 혼합 가중치·손잡이·카운트 routing의 리더보드 연속 미세 조정은 금지한다.
2. 2024 단일 source 잔차 lookup보다 여러 연도에서 동일한 방향을 보이는 생성 구조를 우선한다.
3. 현재 투구 구종·위치·실패유형처럼 추론 시 알 수 없는 oracle 정보는 사용하지 않는다.
4. test 전체 집계·빈도·순서·그룹을 사용하지 않는다.
5. 새 후보는 시간축 최소 gain, 월·도메인, 군집 bootstrap, 패키지 런타임을 모두 통과해야 한다.
6. 팀 저장소는 개인 브랜치만 사용하고 main에 직접 push하지 않는다.

2026-08-17 v23 구조 감사에서 잔차·다년 직접모형·latent state/mode·임베딩 신경망·그룹 강건 학습·post-break spline의 여섯 계열을 평가했다. 2023 선택 gain이 최대 `+505.7564`였던 후보도 동결한 2024에서 `-2.8009`로 반전했으며 모든 후보가 기각됐다. 2022 latent failure-mode OOF는 생성했지만, 다음 모델 선택 전에 2022·2023·2024 전체의 완전한 v22 analogue OOF를 동일 recipe로 완성해야 한다.

2026-08-17 v24 의미 감사에서 공식 `R=1군 정규시즌`, `F=퓨처스리그/2군`을 확인했다. ASOF 누적값은 전 train 행에서 직전 투구 수·성공 수와 정확히 일치했고, TrackMan 익명 ID 연결은 이미 121만 7,598행을 덮는다. 새 LI/점수 상황형, 레벨별 선수 이력형과 Public 역산 단계 가중치는 시간축 게이트를 통과하지 못했다. v22를 유지하며 `submit_v24.zip`은 생성하지 않았다.

2026-08-17 v25 연구에서 실패유형 privileged profile과 세 종류 pitch-type student를 시간축 외부감사로 기각했다. 이후 post-break R_ANCHOR 직접확률 모델은 2023 선택 `+31.4866`, 2024 외부연도 `+1.7500`, 2024 전반→후반 `+0.8211`을 기록했고 Public `+2.7857`로 전이됐다. 상세 근거는 `reports/v25_public_result_20260817.md`에 있다.
