# 초기 배포 계약 검사

확률 범위·행 순서·묶음 분할·오프라인 실행·시간과 메모리를 점검한 기록입니다. 실행 계약 통과와 모델 성능의 독립 확증은 다른 판단입니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Deployment gate

- parent: `submit_v2.zip`
- n_sample_rows: `5`
- n_benchmark_rows: `245789`
- finite_and_probability_range: `True`
- order_checks: `{'reverse_order_max_abs': 1.1102230246251565e-16, 'shuffled_order_max_abs': 1.1102230246251565e-16, 'chunked_max_abs': 1.1102230246251565e-16}`
- offline_static_check: `True`
- row_local_static_check: `True`
- benchmark_seconds: `10.26701900002081`
- peak_rss_mb_observed: `330.13671875`
- parent_prediction_parity_note: `parent itself is the reference; correction-off candidate parity is not applicable because no candidate was promoted`
