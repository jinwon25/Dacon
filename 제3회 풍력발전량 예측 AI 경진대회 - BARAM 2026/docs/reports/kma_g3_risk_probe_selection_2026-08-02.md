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

## 공개 제출 결과

사용자가 DACON 웹에서 한 번 직접 제출했으며, 결과는 실패였다.

- submission ID: `1508186`
- 제출 시각: `2026-08-02 01:04:19`
- score: `0.6454933612`
- 1-NMAE: `0.8755129049`
- FiCR: `0.4154738175`

incumbent `1502437` 대비 macro delta는 다음과 같다.

- score: `-0.0006317302`
- 1-NMAE: `-0.0002713428`
- FiCR: `-0.0009921177`

G1과 G2가 완전히 동일하므로 관측된 차이는 G3에만 귀속된다. 3배로 환산한
G3 단독 효과는 score `-0.0018951906`, 1-NMAE `-0.0008140284`, FiCR
`-0.0029763531`이다. 로컬 다기간 양성과 강한 재표본 검증에도 불구하고
공개 점수가 역전됐으므로, 외부기상·궤적 기반 G3 전체 교체 계열은 종료한다.
