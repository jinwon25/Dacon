# Dacon | 데이터 분석·예측 대회 기록

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) [![코드 라이선스: MIT](https://img.shields.io/badge/License-MIT-3DA639)](LICENSE) [![검증](https://github.com/jinwon25/Dacon/actions/workflows/portfolio-ci.yml/badge.svg?branch=main)](https://github.com/jinwon25/Dacon/actions/workflows/portfolio-ci.yml)

데이터의 변화와 평가 지표를 이해하고, 가설을 검증해 예측을 개선한 프로젝트 모음입니다.
프로젝트마다 문제 정의, 실험 결과, 실패한 접근, 실행 방법을 기록했습니다.
작성자: [@jinwon25](https://github.com/jinwon25)

## 먼저 볼 프로젝트

| 프로젝트 | 확인할 역량 | 결과와 읽을 내용 |
|---|---|---|
| [모기 비행 궤적 예측](모기%20비행%20궤적%20예측%20AI%20경진대회/README.md) | 오차 분석, 가설 검증, 모델 간 상호 보완성 | Private **0.703151**, **2위 / 543팀**. 검증 성능과 제출 성능이 달랐던 이유 |
| [투구 제구 성공 확률 예측](투구%20제구%20성공%20확률%20예측%20AI%20온라인%20해커톤/README.md) | 시즌 변화 분석, 시간 검증, 확률 보정, 실험 관리 | **5인 팀** 리더보드 **20위 / 1,090팀**, Public **1182.9497**. 본인은 EDA부터 최종 재현 패키지까지 전반 담당 |
| [스마트 창고 출고 지연 예측](스마트%20창고%20출고%20지연%20예측%20AI%20경진대회/README.md) | 회귀 오차 해석, 시계열 특징, OOF 앙상블 | MAE **9.86576**, **32위 / 607팀**. 서로 다른 모델의 결합과 분포 변화 |

대회 점수는 해당 평가 데이터에서의 결과입니다. 실제 운영에서의 비용 절감이나 서비스 효과를 측정한 결과로 해석하지 않습니다.

## 전체 프로젝트

| 프로젝트 | 상태·결과 | 주요 접근 |
|---|---|---|
| [모기 비행 궤적 예측](모기%20비행%20궤적%20예측%20AI%20경진대회/README.md) | 종료 · Private R-Hit@1cm **0.703151** ↑ | Kalman 잔차, Neural ODE, Frenet, 회전물리 앙상블 |
| [투구 제구 성공 확률 예측](투구%20제구%20성공%20확률%20예측%20AI%20온라인%20해커톤/README.md) | 종료 · Public BSS **1182.9497** ↑ | 도메인 분리, 시간축 검증, 국소 보정 |
| [스마트 창고 출고 지연 예측](스마트%20창고%20출고%20지연%20예측%20AI%20경진대회/README.md) | 종료 · LB MAE **9.86576** ↓ | GBDT·CNN·GRU 19개 모델, SLSQP 가중치 |
| [풍력발전량 예측 · BARAM 2026](제3회%20풍력발전량%20예측%20AI%20경진대회%20-%20BARAM%202026/README.md) | 종료 · 291위 / 985팀 · Public **0.6474704399** ↑ | 기상장·물리 특징, 그룹별 검증, 실패 분석 |
| [식음업장 메뉴 수요 예측](식음업장%20메뉴%20수요%20예측%20AI%20온라인%20해커톤/README.md) | 종료 · 352위 / 817팀 · Private 가중 SMAPE **0.5481** ↓ | 지표 특성 분석, 업장별 단순 통계 블렌드 |
| [K리그 최종 패스 좌표 예측](K리그-서울시립대%20공개%20AI%20경진대회/README.md) | 종료 대회 학습 · 로컬 CV 거리 **13.95** ↓ | **DuckDB SQL EDA**, GBDT·GRU, 관찰→가설→검증 |
| [내가 축구 감독이라면 · RE:TACTIC](내가%20축구%20감독이라면/README.md) | 전술 웹서비스 프로토타입 | 실제 관측·사용자 입력·휴리스틱 구분, React·TypeScript |

↑ 높을수록 좋음, ↓ 낮을수록 좋음. 대회 순위·참가 팀 수는 프로젝트 기록 기준이며, K리그 CV는 공식 리더보드 점수가 아닙니다. 투구의 리더보드 순위와 오프라인 본선 진출·수상은 별개입니다.

## 저장소를 읽는 방법

1. 프로젝트 README에서 **문제 → 접근 → 결과 → 한계**를 확인합니다.
2. `src/`, `sql/`, `notebooks/`에서 분석과 구현을 확인합니다.
3. `docs/`, `research/reports/`에서 세부 실험과 선택 근거를 확인합니다. 과거 로그의 “현재”는 작성 당시를 뜻합니다.

**SQL·EDA**는 K리그, **시즌 변화와 검증 설계**는 투구, **오차 분석과 지표 해석**은 모기·식음업장 프로젝트에서 확인할 수 있습니다.

## 공개 범위와 실행

공개하는 것은 코드, 출력이 제거된 노트북, 설명 문서와 집계된 실험 지표입니다.
대회 원본 데이터, 행별 예측·OOF, 선수 연결표, 학습 모델, 제출 ZIP, 비공개 공유 링크는 **커밋 이력에서도 제외**했습니다.
데이터는 각 대회의 공식 데이터 페이지에서 이용 조건에 따라 직접 받아야 합니다.

```bash
git clone https://github.com/jinwon25/Dacon.git
cd Dacon
python .github/scripts/audit_public_repository.py --history
python -m unittest discover -s .github/tests -v
```

위 검사는 Python 표준 라이브러리만 사용합니다. 모델 학습은 각 프로젝트의 별도 의존성과 데이터가 필요합니다.
특히 투구의 원 제출과 동일한 재조립에는 공개하지 않은 체크포인트가 필요하며, 공개 clone만으로 전체 점수 재현을 보장하지 않습니다.

공개 판단·규정·검증 결과는 [공개 검토 기록](.github/PUBLICATION.md)에 모았습니다. `.github/scripts/`·`.github/tests/`는 공개 파일 검사를 위한 보조 도구이고, 분석 코드는 각 프로젝트 안에 있습니다.

## 함께 볼 분석 프로젝트

대회 솔루션 외에 의사결정과 해석 중심 사례는 [이커머스 고객 세분화](https://github.com/jinwon25/ecommerce-customer-segmentation), [KBO 투수 피로도](https://github.com/jinwon25/KBO-pitcher-fatigue), [취업 준비 텍스트 마이닝](https://github.com/jinwon25/employment-readiness-text-mining)에서 확인할 수 있습니다.
