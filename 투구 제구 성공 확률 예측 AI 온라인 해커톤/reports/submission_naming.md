# 제출 ZIP 파일명 및 제출 이력 규칙

## 고정 규칙

- 최종 제출 ZIP 파일명은 확장자를 포함해 40자 이내로 한다.
- 파일명은 제출 버전을 즉시 확인할 수 있도록 `submit_vN.zip` 형식을 기본으로 한다.
- 모델 구성·실험명은 `artifacts/candidates/*/manifest.json`과 보고서에 기록하고, 최종 파일명에는 넣지 않는다.
- 다음 사용 가능한 제출은 `submit_v6.zip`이다. 기존 제출 파일은 덮어쓰지 않는다. 새 ZIP은 고정 promotion gate 통과 시에만 만든다.

## 현재 제출 기록

| 버전/제목 | 제출 ID | 제출 일시 | 제출 선택 | 점수 | 소요 시간 | 파일명 |
|---|---:|---|---|---:|---:|---|
| submission1 | 35748 | 2026-08-05 22:54:39 | - | 749.5249490965 | 5초 | `submit.zip` |
| submission2 | 36908 | 2026-08-06 20:45:27 | - | 704.1257475401 | 5초 | `submit_gt_v1.zip` |
| submission3 edit | 39023 | 2026-08-08 16:08:48 | schedule | 763.2665303697 | 9초 | `submit_v2.zip` |
| submission4 edit | 40116 | 2026-08-09 13:30:02 | - | 761.9846367188 | - | `submit_v3.zip` |
| submission5 edit | 40149 | 2026-08-09 14:20:28 | schedule | 743.6520325296 | 5초 | `submit_v4.zip` |
| submission6 edit | 40151 | 2026-08-09 14:21:09 | - | 741.0198690607 | 9초 | `submit_v5.zip` |

`submit_v2.zip`은 기존 기준 제출 대비 `+13.7415812732`점(약 `+1.83%`)이지만, v3~v5는 각각 v2보다 하락했다. Trackman 10% 확대와 R-only branch 패키지는 폐기했으며, 다음 유효 버전은 `submit_v6.zip`으로 고정한다. v2의 5% Trackman과 R-recency 성분 자체는 보존한다.
