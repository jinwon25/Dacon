# 산출물 정책

이 디렉터리는 학습 모델, OOF 예측과 실험 캐시를 보관하며 기본적으로 Git에서 제외한다.
예외적으로 `standalone_champion_1161/`의 확정 ZIP과 manifest,
`oof_champion_1161/`의 fidelity-labelled evidence만 private Git LFS에 보관한다.
`standalone_champion_1159/`과 `standalone_champion_1158/`은 과거 챔피언 감사용으로 보존한다.
`v61_final_gate_oof_20260822_01/`은 후속 후보를 같은 최종 부모에서 비교하기 위한
로컬 연구 cache이며 전달·실행 의존성은 아니다.

## 현재 기준

- champion: `standalone_champion_1161/standalone_champion_1161.zip`
- Public 기준: `1161.2020600422`
- champion SHA-256: `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`
- 계보: v25 → v26 R_ANCHOR `.15` → TrackMan-ASOF `.03` → EXP-021 R_CORE `.10` → shared FM F `.10`
- 실행·재빌드: [`../docs/STANDALONE_CHAMPION.md`](../docs/STANDALONE_CHAMPION.md)
- manifest: `standalone_champion_1161/standalone_manifest.json`
- OOF evidence: [`oof_champion_1161/README.md`](oof_champion_1161/README.md)
- private artifact 감사: `python -m src.audit_champion_private_artifacts --help`

현재 릴리스 실행에는 `standalone_champion_1161/` 이외의 과거 artifact가 필요하지
않다. 과거 artifact는 연구 재현이 필요할 때만 개인 보관소에서 사용한다. 실패한
후보 캐시, 이름에 `tmp_`가 붙은 압축 해제본, 빈 실행 폴더와 로그는 정리 대상이다.

현재 보존하는 연구 cache:

- `standalone_champion_1158/`: 직전 Public `1158.0745556751` 복원·회귀 감사용이다.
- `standalone_champion_1159/`: v84 직접 부모 Public `1159.3352239501` 회귀 감사용이다.
- `v84_fixed_v56_probe_20260822_01/`: 실제 제출본과 상세 학습·패키지 manifest다.
- `v83_post_public_rebase_20260822_01/`: 새 1159 OOF 부모 위 기존 frozen shift의 compact
  조건부 headroom CSV와 summary만 보존한다.
- `v61_final_gate_oof_20260822_01/`: 2025 TrackMan profile exact 재현과
  2023/2024 최종 gate OOF. 다음 직교 후보의 고정 부모로 사용한다.
- v62 ExtraTrees 예측 cache는 기각 후 삭제했다. 결과는
  [`../reports/target1170_research_update_20260822.md`](../reports/target1170_research_update_20260822.md)에
  남아 있다.
- v63 count interaction과 v64 physical-teacher student의 예측·학습 모델 cache도
  기각 후 삭제했다. 코드·테스트·정확한 결과는 같은 최신 연구 보고서에 남아 있다.
- v66/v68/v69/v70/v71/v73/v74 cache는 기각 후 삭제한다. v70은 2024 재사용 축에서
  양수였지만 고정 recipe 과거축 v74에서 source eta가 모두 0이라 최종 기각했다.
  원시 TrackMan census와 정확한 성능은
  [`../reports/trackman_deep_dive_20260822.md`](../reports/trackman_deep_dive_20260822.md),
  코드와 테스트에 보존한다.
- v75 calibration/headroom과 v76 domain×count contrast cache도 판단 확정 후 삭제한다.
  compact 수치와 결론은
  [`../reports/evaluation_reaudit_20260822.md`](../reports/evaluation_reaudit_20260822.md)에
  보존한다.
- v78 환경 안정 잔차의 predictions NPZ와 상세 JSON도 기각 후 삭제한다. compact 결과는
  [`../reports/v78_environment_stable_residual_metrics.csv`](../reports/v78_environment_stable_residual_metrics.csv),
  판단은 [`../reports/target1170_v77_v78_followup_20260822.md`](../reports/target1170_v77_v78_followup_20260822.md)에
  보존한다.
- v79 champion-offset GLM과 v81 stable shallow GBDT의 대형 predictions NPZ는 기각 후
  삭제한다. v80은 compact 공분산·headroom CSV만 보존한다. 세 실험의 정확한 판단은
  [`../reports/target1170_v79_v81_followup_20260822.md`](../reports/target1170_v79_v81_followup_20260822.md)에
  보존한다.

나머지 날짜별 폴더는 과거 실험 재현 자료다. 공간 확보를 위해 삭제할 때는 먼저 연결된 `reports/` 문서와 manifest가 있는지 확인한다.

## 보관 원칙

- 확정 v167 제출 ZIP: PRIVATE 저장소의 `submissions/releases/v167/submit_v167.zip`
- 기존 1161 챔피언과 선별 OOF evidence: private Git LFS allowlist
- v148 full-2024 OOF evidence: 해시 고정 `oof_champion_1170/` 일반 Git 예외
- 중간 OOF와 대규모 캐시: 각 실험 담당자 로컬 또는 DVC/object storage
- 일반 Git에 남길 내용: manifest, 생성 명령, 입력 기준선, SHA-256, 크기, 검증 결과
