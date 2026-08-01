# KMA 다년 G3 단독 risk probe 선정

작성일: 2026-08-02

## 선정 파일

`submissions/probe_kma_umrg_full2022_g3_only_20260802.csv`

- SHA-256:
  `8d523d5cabe3b2d7c57dbe485b1dccee08c9280a8a3f3cb8203ebfdc7023fc24`
- 행 수: 8,760
- 크기: 789,389 bytes
- DACON sample schema 감사: 통과
- 동일 해시의 기존 제출: 없음

## 구성

현 공개 최고 제출 `1502437`의 생산 파일을 incumbent로 사용했다.

- G1: incumbent와 완전히 동일
- G2: incumbent와 완전히 동일
- G3: 2022~2024 pooled KMA UMRG 학습의 2025 예측만 교체
- alpha: `0.8`
- G3 blend weight: `0.05`

전체 pooled staging 후보는 G1과 G3를 함께 바꾸지만, 현 활성 OOF에서 그
조합은 음수였다. 따라서 최종 probe는 통계 게이트를 통과한 G3만 복사하고
G1 변경을 폐기했다.

## 근거와 위험

2024 frozen active OOF 대비 G3 단독 macro delta는 다음과 같다.

- score: `+0.0012430465`
- 1-NMAE: `+0.0002842032`
- FiCR: `+0.0022018899`
- Q1/Q2/H2와 12개월 score 모두 양수
- 10,000회 IID·월층화 40/60 보완 표본 q05 통과
- 2,000회 H2 issue-block bootstrap q05 통과

그러나 UMKR, JMA GSM, JMA MSM 등 로컬 양성 G3 외부기상 교체가 공개
리더보드에서 반복 역전됐다. 이 후보도 정규 promotion은
`rejected_historical_public_failure_guard`이며, 사용자가 한 번의 탐색 제출
위험을 명시적으로 수용한 probe다.

2025 G3 이동은 다음과 같다.

- 변경 행: 8,760
- signed mean: `+74.6656 kWh`
- absolute mean: `129.7537 kWh`
- p95 absolute movement: `345.7271 kWh`
- maximum absolute movement: `724.5374 kWh`
- maximum capacity ratio: `3.4502%`

## 제출 상태

CSV 생성과 로컬 감사는 완료했다. 현재 프로세스 환경에
`DACON_API_TOKEN`과 `DACON_TEAM_NAME`이 없어 외부 업로드는 실행하지
않았다. 토큰을 채팅이나 Git에 기록하지 말고 로컬 환경변수로 설정한 뒤
자동 제출을 재개하거나, 위 CSV를 DACON 웹에서 한 번 직접 업로드한다.
