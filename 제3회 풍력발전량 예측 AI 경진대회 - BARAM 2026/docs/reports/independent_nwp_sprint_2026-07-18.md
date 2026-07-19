# 독립 NWP 전문가 및 강건 라우팅 스프린트 (2026-07-18)

## 결론

제공된 LDAPS/GFS 안에서 전문가를 재조합하거나 소스별 모델을 추가하는 방법은
오류 다양성 상한은 컸지만, 정답 없이 개선 블록을 식별하는 순방향 게이트에서
0.65에 기여할 수준의 효과를 만들지 못했다. 신규 제출은 생성하지 않았다.

다음 유효 분기는 기존 두 NWP와 실제로 독립적인 KMA UM 운영 예보다. 수집기,
인과성 검사, 원본 체크섬 및 join 코드는 준비되어 있으나 현재 환경에는 사용자
발급 `KMA_API_KEY`가 없다.

## 1. 기존 구조 전문가 강건 라우팅

배포 가능한 two-seed 시공간 평균 전문가만 사용하고, 2024년 2~6월을 월별
expanding-window로 검증한 뒤 H2를 한 번 열었다.

| 구간 | 총점 | 1-nMAE | FiCR |
|---|---:|---:|---:|
| 개발 월 통합 | +0.002584 | +0.001065 | +0.004103 |
| 잠금 H2 | -0.000397 | +0.000305 | -0.001099 |

개발 구간의 모든 월을 지켜도 H2에서 FiCR가 반전됐다. 기존 전문가를 기상
유사도로 고르는 계열은 중단한다.

- 코드: `experiments/robust_weather_router.py`
- 결과: `artifacts_final/diagnostics/robust_weather_router_20260718.json`

## 2. LDAPS/GFS 소스 분리 전문가

2023년 1~9월로 학습하고 2023년 4분기에서 iteration을 선택한 뒤, 2023년
전체로 재학습해 2024년을 예측했다. 2024년은 소스 모델 선택에 사용하지 않았다.

### LightGBM H2 단독 성능

| 전문가 | 총점 변화 | 1-nMAE 변화 | FiCR 변화 |
|---|---:|---:|---:|
| LDAPS all | -0.054734 | -0.029211 | -0.080257 |
| LDAPS eligible | -0.039140 | -0.018425 | -0.059855 |
| GFS all | -0.045237 | -0.024060 | -0.066413 |
| GFS eligible | -0.035961 | -0.015794 | -0.056127 |

6시간 단위 정답 oracle은 `+0.068072`, FiCR `+0.117093`이었고 LDAPS와
GFS 오류 상관은 약 `0.72~0.78`이었다. 독립성은 있으나 단독 정확도가 너무
낮았다.

### CatBoost eligible H2 단독 성능

| 전문가 | 총점 변화 | 1-nMAE 변화 | FiCR 변화 |
|---|---:|---:|---:|
| LDAPS | -0.028534 | -0.013381 | -0.043686 |
| GFS | -0.015081 | -0.007358 | -0.022805 |

CatBoost가 격차를 줄였지만 완전 교체는 불가능했다. CatBoost 두 전문가의 6시간
oracle 상한은 총점 `+0.052584`, FiCR `+0.091068`이었다.

- 코드: `experiments/nwp_source_ablation.py`
- 코드: `experiments/nwp_source_catboost.py`
- 결과: `artifacts_final/nwp_source_ablation/report.json`
- 결과: `artifacts_final/nwp_source_catboost/report.json`

## 3. 여섯 소스 전문가 희소 라우팅

LightGBM 4개와 CatBoost 2개를 함께 사용해 최대 2~5% 블록에서 원 모델의
5~15%만 이동하도록 제한했다. 구 게이트에서는 Q2 정책 2개가 비열화 없이
남았지만, 잠금 H2 결과는 18개 행에서 총점 `+0.0000037`, FiCR `0`에 불과했다.

이는 통계적으로나 0.65 목표 기여도 면에서 무의미하다. 서비스 승격 조건에
잠금 총점 `+0.002`, FiCR `+0.003`의 최소 효과 크기를 추가했고 이 실험은
최종 거절했다.

- 코드: `experiments/nwp_source_router.py`
- 결과: `artifacts_final/diagnostics/nwp_source_router_20260718.json`

## 다음 실행 조건

1. 사용자 발급 KMA APIHub 키를 프로세스 환경변수 `KMA_API_KEY`로만 제공한다.
2. `fetch_kma_um_global.py`로 2024년 단일 issue 파일럿을 먼저 수행한다.
3. 응답 파싱·공개시각·리드 보간·체크섬 검증 후 2024년 Q1/Q2/H2만 수집한다.
4. KMA 신호가 잠금 총점, 1-nMAE, FiCR와 모든 월 게이트를 통과해야만 2025년
   데이터를 수집하고 `submissions/` 후보를 생성한다.

키 값은 명령행, 보고서, 로그, 소스 코드에 기록하지 않는다.
