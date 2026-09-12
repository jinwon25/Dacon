# v320 F 포트폴리오 후보 — 1180 목표 후속 결과

## 결론

공식 champion은 Public `1176.757071668`의 v290으로 유지한다. v320은 v290의 R 경로를
비트 수준으로 보존하면서 F 경로에 서로 다른 두 신호를 결합한 **탐색 제출 후보**다.
full-2024 잠금 평가에서 v290 대비 `+3.808797` BSS를 기록했다. v290의 v244 대비 로컬
이득 `+4.514976`과 단순 합산한 로컬 누적 이득은 `+8.323773`이다. 이 수치를 Public 점수로
환산하지 않으며, 1180 도달 여부는 실제 제출만 판정할 수 있다.

## 고정 레시피

- 적용 범위: `game_type == "F"`만 변경, R 및 기타 경로 완전 보호
- 최근 F 직접 전문가: v290의 10%를 20%로 확대
- 저랭크 신호: 과거 v50에서 동결된 `pitcher × (count, batter_hand)` rank-2 잔차,
  smoothing 300, probability delta weight 0.5
- 20% 직접 전문가 용량은 v290 이전의 고정 로컬 곡선에서 선택됐다.
- 저랭크 레시피는 v290 이전 두 origin에서 선택된 과거 레시피를 그대로 재현했다.
- full-2024 잠금셋은 레시피·용량 선택에 사용하지 않았다.

## 시간 전이 성능

| 평가축 | 직접 전문가 추가 | 저랭크 | 포트폴리오 |
|---|---:|---:|---:|
| late-2023 source | +135.222 | +0.502 | **+135.651** |
| full-2024 locked | +1.610 | +2.114 | **+3.809** |

full-2024 포트폴리오는 8개월 중 5개월 양수였고 최악 월은 `-61.818`이다. 전체 행 RMS
이동은 `0.00172569`, F 활성 행 평균 절대 이동은 `0.00396362`다. 직접 전문가와 저랭크
신호는 서로 완전히 독립적이지 않지만 결합 locked 이득이 각 단독 이득보다 크다.

## 강건성

- pitcher cluster bootstrap: p05 `+8.081`, 양수 확률 `0.987`
- chronological block bootstrap: p05 `+11.948`, 양수 확률 `0.9945`
- crossed pitcher×batter bootstrap: p05 `-3.694`, 양수 확률 `0.929`
- 제한된 최종 4개 후보군 White Reality Check: `p=0.005997`

점 추정과 두 강건성 축은 통과했지만 crossed bootstrap p05는 음수다. 따라서 승격 확정본이
아니라 **Public 확인이 필요한 유의미한 후보**로 분류한다.

## 패키지 감사

- ZIP: `artifacts/v320_futures_portfolio_package_20260830_01/submit_v320_futures_portfolio.zip`
- SHA-256: `95ABFE9DDBA696A425FB930A0A5ACE1007D1AF5EEBF608437297479111B64D44`
- 크기/멤버: `91,469,844 bytes`, `188 files`
- parent v290 SHA 일치: `09DE96351304E80BC23B4A9B61739B901E6FD1A4C8BD18E131A352312BA0A4B6`
- CRC 및 루트 구조 통과, 금지된 행간 연산 0
- R 경로 parity 최대 오차 `0`
- F 공식 최대 오차 `5.55e-17`
- singleton `5.55e-17`, shuffle `0`, partition `0`
- 245,789행 runtime `187.69초`, finite/range 통과

감사 원본은 같은 artifact 폴더의 `formula_independence_audit.json`,
`standalone_audit_summary.json`, `summary.json`에 고정했다.

## 위험 및 의사결정

v296 실패로 full-2024 로컬 이득을 Public 점수에 1:1 대응하거나 신뢰구간으로 환산하는
평가 방식은 폐기했다. v320의 전체 RMS 이동도 과거 1180 목표에 대응시킨 경험적 규모
`0.005752`보다 작다. 반면 v320은 R을 건드리지 않고, v290 이전 근거가 있는 두 F 신호를
결합하며, source/locked/독립성/실행 감사를 통과했다. 그러므로 현재 남은 결정적 검증은
하나의 exploratory 공식 제출이다. 결과가 나오기 전에는 이 계열의 weight를 Public에 맞춰
재튜닝하지 않는다.

후속으로 Beta–Binomial current-season pooling, 표본 성숙도 gate, 300-tree ExtraTrees와의
strict-forward 결합까지 독립 검증했다. Beta 계열은 locked에서 반전했고 ExtraTrees 결합은
source 단계에서 R/F 모두 0%가 선택됐다. 따라서 이들을 v320에 사후 추가하지 않는다. 상세:
`reports/v321_v323_public_method_independent_screen_20260830.md`.
