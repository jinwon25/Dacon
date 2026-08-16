# 프로젝트 현황

마지막 갱신: `2026-08-16 17:10 KST`

## 현재 champion

| 항목 | 값 |
|---|---:|
| champion | `submit_v21.zip` |
| Public | **1151.5138356157** |
| 제출 ID | `1534213` |
| 확인 당시 순위 | **12위** |
| v20 대비 | **+0.0414068967** |
| Top 10 경계 | `1157.9594495591` |
| Top 10까지 | **6.4456139434** |
| 1160까지 | **8.4861643843** |
| SHA-256 | `125330E9A2D99532025B0FD1ED845390C7C12D7A3A9383B26FD430F7B795D049` |
| 오늘 남은 제출 | **0회** |

v21은 v20의 완전한 자식이며, 여섯 개의 동결 recency empirical-Bayes lookup만 추가한다. Public 방향은 양수였지만 로컬 `+10.7720` 중 `+0.0414`만 전이됐다. 이 계열의 추가 미세 조정은 중단한다.

## champion 계보

```text
v13 exact-ASOF
  → v17 R_CORE pressure EB
    → v19 multi-year state + latent failure-mode routing
      → v20 pooled residual + training-only TrackMan PFD
        → v21 recent-control state + count-context recency EB
```

## 바로 읽을 문서

1. `reports/v21_public_result_20260816.md`
2. `reports/target1200_research_20260816.md`
3. `reports/dacon_official_compliance_audit_20260816.md`
4. `reports/v21_validation.md`
5. `reports/submissions.csv`

## 구현·재현 파일

```text
src/train_v21_context_state_eb.py
src/package_v21_context_state_eb.py
src/validate_v21_context_state_eb.py
src/v10_overlay_script.py
tests/test_v21_context_state_eb.py
```

최종 모델·OOF·데이터·제출 ZIP은 Git에 올리지 않는다. 로컬 champion ZIP은 프로젝트 루트에 두고 이전 ZIP은 `submissions/history/`로 이동한다.

## 검증 상태

- 전체 테스트: `77 passed`
- 245,789행 추론: `51.120초`
- peak RSS: `1,402.2 MiB`
- 배치 불변성 최대 오차: `1.11e-16`
- 패키지·규정 게이트: `12/12 passed`
- 공식 Public: `1151.5138356157`

## 다음 연구 원칙

1. v21 lookup 가중치 미세 조정은 금지한다.
2. 2024 단일 source 잔차 lookup보다 여러 연도에서 동일한 방향을 보이는 생성 구조를 우선한다.
3. 현재 투구 구종·위치·실패유형처럼 추론 시 알 수 없는 oracle 정보는 사용하지 않는다.
4. test 전체 집계·빈도·순서·그룹을 사용하지 않는다.
5. 새 후보는 시간축 최소 gain, 월·도메인, 군집 bootstrap, 패키지 런타임을 모두 통과해야 한다.
6. 팀 저장소는 개인 브랜치만 사용하고 main에 직접 push하지 않는다.
