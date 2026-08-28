# v180 signed-stack Public 결과

기록 시각: `2026-08-28 KST`

## 결과

| 항목 | 값 |
|---|---:|
| 제출 ID | `71754` |
| 제출 파일명 | `submit_v180.zip` |
| 제출 시각 | `2026-08-28 14:59:16 KST` |
| Public | `1172.0987352738` |
| 기존 챔피언 | `1172.1373858439` |
| 기존 챔피언 대비 | `-0.0386505701` |
| v167 대비 | `+0.0214972417` |
| 1180까지 gap | `7.9012647262` |

개별 제출 점수와 제출 ID는 사용자가 DACON 제출 화면에서 확인해 제공했다. 공식 best-only
리더보드는 더 높은 기존 챔피언을 계속 표시한다.

## 제출 패키지

- 빌드 원본: `artifacts/v180_signed_stack_package_20260828_01/submit_v180_signed_stack.zip`
- Downloads 제출본: 파일명만 `submit_v180.zip`으로 변경
- SHA-256: `91CA020EFA776BA12BF50630FCE533A06EE6E6984FD89C26E5C20BB5FFB5AAC4`
- 크기: `86,084,812 bytes`
- 로컬 245,789행 환산 실행 시간: `177.248초`
- singleton/shuffle/partition 최대 오차: `1.67e-16`

## 판정

v178은 full-2022 `+0.748373`, late-2023 `+0.617614`, locked-2024 `+0.193590`
BSS gain과 세 종류 bootstrap 양수 p05, 제한된 Reality Check `p=0.002499`를 통과했다.
그러나 실제 Public에서는 직접 부모보다 `0.0386505701` 낮았다.

따라서 v180은 **기각하고 승격하지 않는다**. 기존
`jy_runners_high_li_bridge027` Public `1172.1373858439`를 계속 공식 챔피언으로 유지한다.
이번 한 점의 Public 결과로 signed-stack 계수나 scale을 재조정하지 않는다.
