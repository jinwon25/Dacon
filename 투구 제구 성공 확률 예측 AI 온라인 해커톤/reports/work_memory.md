# 작업 메모리

## 제출 ZIP 이름 규칙

- 최종 산출물 ZIP 파일명은 **40자 이내**.
- 기본 형식은 `submit_vN.zip`이며 N으로 버전을 확인한다.
- 기존 ZIP을 덮어쓰지 않고, 제출 전 SHA-256과 파일명을 기록한다.

## 현재 공개 기준선

- 현재 공개 챔피언: `submit_v2.zip`, Public 763.2665303697.
- `submit_v3.zip`은 Trackman 10% probe로 761.9846367188을 기록했으므로 폐기한다.
- 다음 통제 후보: `submit_v4.zip` (R-only RF 25%, Trackman 0%), `submit_v5.zip` (R-only RF 25% + Trackman 5%).

## 의사결정 원칙

- 한 제출에서는 변경 축을 하나만 분리하거나, 사전에 로컬 검증된 조합만 사용한다.
- R-only RF 25%는 2021~2024 season-forward 검증에서 네 해 모두 개선했다.
- Trackman은 공개 검증상 5%를 상한으로 두고 10% 이상으로 확대하지 않는다.
- 공개 결과가 로컬 검증과 다르면 공개 결과를 우선하고, 해당 가설을 폐기하거나 weight를 낮춘다.
