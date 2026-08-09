# Top-1100 data contract

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
