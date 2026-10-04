# 초기 데이터 계약과 스키마 검사

입력 자료의 크기·열 구성·해시·연도별 정답 비율을 집계한 감사입니다. 원본 행은 공개하지 않으며 작은 공식 테스트 예제를 전체 서버 평가 자료로 해석하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Top-1100 data contract

Created: `2026-08-09T16:39:32.997301+09:00`

## Files
- **train**: `1,475,092` rows × `49` columns; `368,527,723` bytes; SHA-256 `D2081186B458B49F60B082BE480C273135833E15BA59A76D033AF28BCF8763FF`
- **test**: `5` rows × `48` columns; `1,894` bytes; SHA-256 `478D10B20C00443F6FE8270AB7348DE49D14395526130F0B2D915DA496821A19`
- **trackman_history**: `1,793,078` rows × `30` columns; `353,823,031` bytes; SHA-256 `F7818F9EE0CCEFE7C2CF69FA99EFE6E5CB882D8B886DD96D2394BCF3B53F33A9`
- **sample_submission**: `5` rows × `2` columns; `112` bytes; SHA-256 `B2CF6BA6745C74C46DB23E620D74705F9B327EFB8C8A14DC0DD2AFAB0E898775`

## Schema checks

- `train_test_same_input_columns`: **PASS**
- `sample_columns`: **['row_id', 'control_success']**
- `trackman_has_target`: **False**
- `row_id_unique_train`: **PASS**
- `row_id_unique_test`: **PASS**

## Target rate by season

```text
          rows  target_sum  target_count  target_rate
2019  237413.0    134060.0      237413.0     0.564670
2020  244087.0    130028.0      244087.0     0.532712
2021  247088.0    131639.0      247088.0     0.532762
2022  247472.0    130893.0      247472.0     0.528920
2023  245525.0    122752.0      245525.0     0.499957
2024  253507.0    123231.0      253507.0     0.486105
```

All hashes and counts above were rebuilt from the original CSVs. No external outcome data were read.
