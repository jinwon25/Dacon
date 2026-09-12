# KMA UM 10.4 컨텍스트 스프린트 (2026-07-19)

## 결론

단일 지점 10 m 전구 UM 바람을 기존 GFS와 비교하는 것만으로는 0.65 돌파 신호가
아니다. 2024년 7월부터 9월 8일까지 확보된 예비 잠금 구간에서 선택 구간은
통과했지만, 잠금 점수는 `-0.00028103`, 1-NMAE는 `-0.00013592`, FiCR은
`-0.00042614` 하락했다. 이 분기는 제출하지 않는다.

다음 검증 단위는 10.4 발표시간 기준 시계열 API가 한 번에 제공할 수 있는 정보를
활용한 다음 조합이다.

1. 예측에 사용 가능한 최신 UM 사이클과 그 직전 6시간 사이클의 차이
2. 10 m, 850 hPa, 700 hPa U/V와 연직 시어·풍향 일치도
3. 전구 UMGL 30 km가 통과한 뒤에만 지역 UMRG 12 km 단일 이슈 파일럿

설비 좌표는 대략 위도 `37.27~37.29`, 경도 `128.95~128.98` 범위라 전구 N128
격자에서는 사실상 한 점이다. 이 단계에서 주변 위경도를 무작정 늘리는 것보다
시간·연직 컨텍스트와 더 높은 해상도의 독립 모델을 먼저 확인하는 편이 정보 대비
요청 비용이 작다.

## 방법론 근거

기상청 공식 수치모델 문서는 10.4 발표시간 기준 시계열 요청에 `ef=시작,끝,간격`,
`data=P`, `level=850,700`을 사용할 수 있다고 설명한다. UM 변수표의 풍장 코드는
U `0,2,2`와 V `0,2,3`이므로 API 정수 코드는 각각 `2002`, `2003`이다. N128의
2024년 과거 예보는 UM 생산 종료 후에도 과거 시점 조회가 가능하다는 공지 범위에
들어간다.

- [기상청 수치모델 API와 10.4 시계열 조회](https://apihub.kma.go.kr/apiList.do?seqApi=9&seqApiSub=278)
- [기상청 UM GRIB 변수표](https://apihub.kma.go.kr/getAttachFile.do?fileName=um_var_inf.pdf)
- [UM 생산 종료와 과거 자료 조회 공지](https://apihub.kma.go.kr/notice.do?seqNotice=52)

예보 사이클 간 변화는 진짜 앙상블 분산과 같지는 않으므로 `spread`로 과장하지
않고 `run_change`로 취급한다. 다만 NWP 앙상블의 분산과 다중 모델 차이가 풍력
오차·램프 불확실성에 정보를 준다는 선행 결과가 있어, 직전 사이클과 상층 구조는
현재 단일 결정론 예보에 없는 불확실성 대리변수로 검증할 가치가 있다.

- [Optimising ensemble information in numerical weather forecasts of wind power generation](https://doi.org/10.1088/1748-9326/ab5e54)
- [Forecasting ramps of wind power production with NWP ensembles](https://doi.org/10.1002/we.526)
- [Predicting power ramps from joint distributions of future wind speeds](https://doi.org/10.5194/wes-7-2195-2022)

문헌은 방향을 정하는 근거일 뿐 승격 근거가 아니다. 승격 여부는 아래 2024년 시간
분리 검증만으로 결정한다.

## 구현

`experiments/fetch_kma_um_context.py`는 이슈마다 다음 네 개 객체를 기본 요청한다.

- 최신 사이클 단일면 10 m U/V
- 최신 사이클 850/700 hPa U/V
- 직전 사이클 단일면 10 m U/V
- 직전 사이클 850/700 hPa U/V

각 객체는 원문 바이트, 인증키가 제거된 URL, 조회시각, 초기화·공개·예측 기준시각,
SHA-256을 보존한다. 12시간 공개 지연을 적용해 기준시각보다 최소 4시간 이전에
공개된 사이클만 사용하고, 3시간 예보 사이를 시간 보간하되 범위 밖 외삽은 거부한다.
응답 초기화시각, 유효시각, 변수, 고도 중 하나라도 불완전하면 파일을 쓰지 않는다.

`.env.local`의 `KMA_API_KEY=...`를 직접 읽을 수 있다. 프로세스 환경변수가 있으면
그 값을 우선하며, 키는 명령행·로그·원문 출처 URL·매니페스트에 기록하지 않는다.

## 실행 순서

권한·응답 형식을 확인하는 전구 네 요청 파일럿:

```powershell
python -m experiments.fetch_kma_um_context --max-issues 1 --output-dir artifacts_final/external_weather/kma_um_context_pilot
```

파일럿이 `violations: 0`으로 끝난 경우에만 2024년 전구 전체 수집과 검증:

```powershell
python -m experiments.fetch_kma_um_context
python -m experiments.kma_um_context_screen
```

전구 파일럿 이후 지역모델 파라미터를 확인하는 네 요청 파일럿:

```powershell
python -m experiments.fetch_kma_um_context --group UMRG --nwp N512 --max-issues 1 --output-dir artifacts_final/external_weather/kma_um_regional_pilot
```

지역 API가 해당 과거 조합을 반환하지 않으면 모델 코드 추측으로 전체 요청을 반복하지
않고 중단한다. UMKR 1.5 km는 보수적 공개 지연에서 하루 마지막 목표들이 48시간
예측 범위를 벗어날 수 있으므로, 범위 밖 외삽을 도입하지 않는 한 우선순위에서 제외한다.

## 고정 승격 계약

학습은 2024 Q1, 정책 선택은 Q2, 잠금 검증은 H2 한 번만 사용한다. 다음 조건을
모두 만족해야만 `submission_eligible=true`가 된다.

- 선택 구간에서 점수·1-NMAE·FiCR 모두 비음수가 되는 정책 존재
- 잠금 H2의 세 지표 모두 양수
- 동일한 달력/기준예측 통제군 대비 세 지표 증분 모두 양수
- 세 개 시드 각각의 잠금 세 지표가 모두 양수
- 잠금 H2 모든 월의 FiCR이 비음수
- 잠금 점수 개선 `>= 0.001`
- 통제군 대비 잠금 점수 증분 `>= 0.0005`
- 전체 2024년 자료이며 예비 구간 플래그가 아님

현재 공개 최고 `0.6417471627`에서 0.65까지는 `+0.0082528373`이 필요하다. 작은
양수 하나를 근거로 2025 자료와 제출을 계속 늘리지 않도록 최소 개선폭도 고정했다.

## 현재 검증 상태

- 기존 단일 10 m 전구 분기: 예비 잠금 음수, 제출 거부
- 두 사이클·상층 컨텍스트 수집기: 구현 완료
- CP949/UTF-8 응답, Pa/hPa 고도, 두 사이클, 보간, 파생 특성, 키 로딩 테스트: 통과
- 전체 저장소 테스트: `170 passed`
- 실제 10.4 전구 네 요청 파일럿: 통과
  - 원문/체크섬 4개 일치, 최신·직전 사이클 및 10/850/700 hPa 완전
  - 인과성 위반 0, 최소 공개 여유 240분
  - 파생 컨텍스트 특성 41개, 산출물 내 인증키 잔존 0건
- 실제 10.4 지역 12 km 네 요청 파일럿: 통과
  - 원문/체크섬 4개 일치, 인과성 위반 0, 최소 공개 여유 240분
  - 전구와 공통 파생 특성 41개가 모두 달라 UMRG 요청이 실제 반영됨
  - 산출물 내 인증키 잔존 0건
- 지역 12 km 2024년 전체 수집: 통과
  - 367개 이슈, 8,784시간, 원문 1,468개 체크섬 일치
  - 중복·결측·인과성 위반·인증키 잔존 모두 0
- 직접 잔차 모델 잠금 H2: 거부
  - 점수 `-0.00042926`, 1-NMAE `-0.00024016`, FiCR `-0.00061837`
  - 세 시드 모두 세 지표 음수
- fine meta-gate 위험 라우터: Q2에서 거부
  - KMA 없는 동일 통제 라우터는 유효했지만 KMA 포함 라우터는 시드 안정 정책 0개
  - H2 미개방, 해당 라우터의 제출 CSV 미생성

직접 잔차 보정과 위험 라우팅 하위 분기는 종료했다. 그러나 트리 잔차모델이 놓친
단조 물리 경로를 별도 검증한 결과, `experiments/kma_um_power_curve_gate.py`의 제한된
공동 power curve가 승격 계약을 통과했다.

## 제한된 공동 power curve

현재 공개 최고 제출의 정확한 `p=0.545`, `alpha=0.50` rolling OOF 표면을 기준으로
다음 계약을 사용한다.

1. Q1에서 풍속과 발전량 사이의 단조 isotonic curve만 학습한다.
2. 직접 그룹3, 세 그룹 정규화 평균, 그룹1·2 정규화 평균, 세 그룹 정규화 중앙값을
   사전 정의된 proxy 후보로 둔다.
3. Q2에서 총 세 지표가 양수이고 동일 GFS 850 hPa power-curve 통제군 대비 세 지표가
   각각 최소 `+0.0001`인 proxy·게이트·혼합 강도만 선택한다.
4. 각 행의 최대 이동은 그룹3 설비용량의 5%인 `1,050 kWh`로 제한한다.
5. H1으로 power curve를 다시 학습한 뒤 H2를 한 번 평가하고, 월별 FiCR·이슈 블록
   부트스트랩·통제군 순증분을 모두 검사한다.

Q2에서는 세 그룹 정규화 중앙값 proxy가 선택됐다. H1 재학습 잠금 H2 결과는 다음과
같다.

- 현재 공개 최고 OOF 대비: 점수 `+0.00699607`, 1-NMAE `+0.00304475`,
  FiCR `+0.01094739`
- 동일 GFS 통제군 대비 KMA 순증분: 점수 `+0.00233477`,
  1-NMAE `+0.00073861`, FiCR `+0.00393093`
- 7~12월 각각의 총 FiCR 증분: 모두 양수
- 2,000회 이슈 블록 부트스트랩 KMA 순증분 하위 5%:
  점수 `+0.00032543`, 1-NMAE `+0.00004736`, FiCR `+0.00039537`
- 최대 이동 `1,050 kWh`, 잠금 구간 평균 절대 이동 `175.34 kWh`

따라서 2025 UMRG 수집만 승격됐다. 아직 2025 원천자료가 없으므로 제출 CSV는 만들지
않았다. test의 마지막 `2026-01-01 00:00` 목표를 빠뜨리지 않도록 종료 경계를 한 시간
뒤로 둔다.

```powershell
python -m experiments.fetch_kma_um_context `
  --metadata data/test/gfs_test.csv `
  --start "2025-01-01 00:00:00" `
  --end "2026-01-01 01:00:00" `
  --group UMRG `
  --nwp N512 `
  --output-dir artifacts_final/external_weather/kma_um_regional_context_2025

python -m experiments.kma_um_power_curve_gate --write-submission-if-qualified
```

두 번째 명령은 2025 매니페스트·전체 시계열·공개시각을 다시 검증하고 모든 잠금
게이트가 참일 때만 `submissions/blend_best_kma_um_power_curve_gate.csv`를 쓴다.
