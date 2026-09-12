# Target 1180 후속: v167 Public 개선 결과

기록 시각: `2026-08-25 01:44 KST`

## 결과

| 항목 | v148 | v167 | 차이 |
|---|---:|---:|---:|
| Public | 1170.3014697177 | **1172.0772380321** | **+1.7757683144** |
| 제출 ID | `1544757` | `1547707` | - |
| 확인 시점 순위 | 14위 | **11위** | +3 |
| 1180까지 | 9.6985302823 | **7.9227619679** | -1.7757683144 |

- DACON 접수 시각: `2026-08-25 01:39:32 KST`
- API 응답: `isSubmitted=true`, `detail=Success`
- 공식 대회 ID: `236743`
- 제출 전 quota: `5`, 제출 후 예상 잔여 quota: `4`
- 공식 리더보드에서 팀명, 제출 ID, 점수, 제출수 `49`를 재확인했다.

## 최종 패키지

- GitHub 전달 파일: `submissions/releases/v167/submit_v167.zip`
- 로컬 빌드 원본: `artifacts/v167_h1_affine_submission_package_20260825_01/submit_v167.zip`
- SHA-256: `30DD28F56723EC0F560C9101FC5A94EF78568F874DCA88BF808879831E61C8C1`
- 크기: `46,354,018 bytes`
- 구성: `89`개 파일, 루트 `model/`, `script.py`, `requirements.txt`
- v148 대비 변경 멤버: `model/h1/model/rf.pkl` 하나

v167은 v148의 부모·C3·라우팅·가중치를 모두 보존하고, R_CORE에서 H1의 원 공개 구현에
포함됐던 고정 affine만 복원한다.

```text
H1_post = clip(0.5854452601930041
               + 1.09 * (H1_raw - 0.5854452601930041),
               0.001, 0.999)
```

R_ANCHOR와 F는 v148과 동일하다. 이 상수는 이번 연구에서 탐색한 값이 아니라 원 H1
패키지에 고정돼 있던 값이다.

## 선택 근거와 한계

배포 H1과 동일한 depth-8, seed 42/43/44 앙상블을 strict-forward OOF로 다시 만든 후
v148 수치 계약을 재구축했다. v160의 고정 affine은 exact-v148 2024 축에서 `+2.1532287780`,
R_CORE에서 `+3.0566942300`이었다. 새 저복잡도 후보 중 locked gain이 가장 컸기 때문에
사용자의 명시적 요청에 따라 탐색 제출했다.

사전 안정성 gate는 통과하지 못했다. full-2022는 `+11.1109688000`이지만 late-2023은
`-7.4035640365`였고, pitcher/crossed/block bootstrap p05는 각각
`-0.4304/-1.7691/-0.9890`, White reality-check p는 `.1609`였다. 따라서 실제 Public
개선은 후보의 실측 승격 근거이지만 같은 affine 축을 추가 튜닝하거나 테스트 정답을
역추정하는 근거로 사용하지 않는다.

## 재현·안전성 검증

| 검사 | 결과 |
|---|---:|
| 부모 v148 SHA 잠금 | 통과 |
| 원 H1 SHA 잠금 | 통과 |
| ZIP 구성 차이 | H1 모델 1개만 변경 |
| H1 affine 수식 오차 | `0.0` |
| 전체 후보 수식 최대 오차 | `1.11e-16` |
| 비활성 경로 보존 최대 오차 | `1.11e-16` |
| 셔플 최대 오차 | `1.11e-16` |
| 분할 최대 오차 | `1.11e-16` |
| 245,789행 프록시 | `88.4978초` / 내부 `120초` |
| 출력 범위 | `0.3699093 ~ 0.5122415` |
| 독립 재빌드 | SHA-256 완전 일치 |
| 저장소 테스트 | `394 passed, 4 skipped` |

예측·가중치 선택에는 2025 결과, test 전체 분포·집계·순서, 다른 test 행의 정보를
사용하지 않았다. 로컬 test 사본은 패키지 실행과 행 독립성 검사에만 사용했다.

## 재구축

확정 v167 ZIP 한 개는 PRIVATE 저장소 전달 예외로 추적하지만, 대용량 부모 ZIP, 원 H1 ZIP,
OOF와 공식 데이터는 계속 Git에서 제외한다. 팀 보관소에서 아래 해시의 두 부모를 준비한 뒤
재구축을 실행한다.

- v148 ZIP: `7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337`
- 원 H1 ZIP: `D338E9EEF715C3CC84C0C5CD9C830925A93D85BF9518BFB0DCD0D0B052F995D6`

```powershell
python -m src.archive.v167_build_h1_affine_submission_package `
  --parent-zip artifacts\v148_v142_v138_blend_package_20260823_01\submit_v148.zip `
  --source-h1-zip artifacts\external_hoo_h1_exact_20260824\submissions\cand_asof_xl.zip `
  --data-dir data `
  --evidence-summary artifacts\v160_original_h1_affine_audit_20260825_01\summary.json `
  --config configs\v167_h1_affine_submission_package.json `
  --output-dir artifacts\v167_h1_affine_submission_package_20260825_01 `
  --timeout 240
```

근거 OOF 체인은 `v157_exact_deployed_h1_oof` → `v158_exact_h1_c3_contract` →
`v160_original_h1_affine_audit` 순서다. 각 결과 디렉터리는 Git에서 제외되고 코드와 config만
버전 관리한다.

## 남은 위험과 다음 연구

- 목표 1180까지 `7.9227619679`가 남았다.
- late-2023 반전과 음수 cluster-bootstrap p05 때문에 이 affine을 더 강하게 하는 탐색은
  우선순위가 낮다.
- 다음 후보는 동일 affine 재튜닝보다, exact deployed-contract와 같은 row-aligned
  multi-origin OOF를 갖는 독립 구현 신호여야 한다.
- 이후 Public 점수는 새 후보의 승격·기각에만 쓰며 test 정답 또는 행별 오차를 추론하지
  않는다.
