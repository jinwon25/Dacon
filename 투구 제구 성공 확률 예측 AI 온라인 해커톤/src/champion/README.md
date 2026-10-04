> 최종 선택은 **v345 (1182.9497)입니다.** 아래 v148 안내는 과거 계보 기록이며 공개하지 않은 모델·ZIP이 필요합니다. 최종 결과와 실행 범위는 [문서 안내](../../docs/README.md)를 따릅니다.

# 실제 제출 버전의 코드 계보

실제 제출된 버전으로 이어지는 모듈을 모았습니다. 이 폴더는 `src/core/`와 다른 `src/champion/` 모듈만 불러오며 `src/archive/`를 불러오지 않습니다. 관련 없는 세대를 함께 읽지 않아도 해당 계보를 이해할 수 있습니다. 실행에는 별도로 제외된 입력 자산이 필요합니다.

종료된 개별 비교 실험은 `src/archive/`에 보관합니다.

## 실제 제출한 과거 버전

| 버전 | Public | 계보상 역할 |
|---|---:|---|
| v148 | **1170.3014697177** | 2026-08-23 당시 기준 모델, 보수적인 v142→v138 결합 |
| v142 | 1169.6277932822 | 독립 H1 + 부호가 안정된 C3 |
| v124 | 1164.2949203402 | 여러 축의 공개 결과 기반 이차식 결합 |
| v104 | 1162.6302840289 | 자료원 안정성의 다수 조건 마스크 |
| v84 | 1161.2020600422 | 고정 v56 공유 FM, F 경로 |
| v26 | 1157.9736407889 | 신호 가중치 15% 고정 |
| v25 | 1155.8293405409 | 변화점 이후 R_ANCHOR 직접 확률 |
| v22 | 1153.0436023798 | 낮은 분산의 도메인 보정 + 행별 ASOF |

## 과거 v148 실행·재조립 기록

`submit_v148.zip`은 당시 전달한 독립 실행 패키지입니다. 평가 서버는 ZIP을 풀어 내부 `script.py`를 실행했고, 평가 시 저장소 코드 자체는 필요하지 않았습니다. ZIP은 현재 공개본에 포함하지 않습니다.

원래 ZIP에서 단일 구조 패키지를 다시 만드는 당시 명령입니다.

```bash
python -m src.champion.v148_flat_build_package \
  --original-zip artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip \
  --runtime-script src/champion/v148_flat_runtime_script.py \
  --output-dir artifacts/v148_flat_20260823_01
```

`v148_flat_runtime_script.py`는 단일 구조 패키지의 실행 코드입니다. 하나의 `main()`과 정적 import를 사용해 원래의 여섯 단계 동적 부모 스크립트 호출을 대체합니다. 예측은 부동소수점 오차 범위에서 원본과 일치했으며 [당시 비교 보고서](../../research/reports/v148_flat_refactor_20260823.md)에 근거가 있습니다.

별도 자산을 확보한 경우 다음 명령으로 예측 일치를 확인합니다.

```bash
python -m src.champion.v148_flat_parity_audit \
  --original-zip artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip \
  --flat-zip artifacts/v148_flat_20260823_01/submit_v148_flat.zip \
  --train-csv data/train.csv --rows 100000
```

## 이전 세대 재조립

각 빌더는 입력을 명령행에서 명시적으로 받습니다. 이전 버전의 작업 폴더를 자동 참조하지 않습니다.

```bash
# v142 — independent H1 and sign-stable C3
python -m src.champion.v142_build_submission_package \
  --parent-zip <v124 zip> --h1-zip <H1 zip> \
  --runtime-script src/champion/v142_runtime_script.py \
  --train-csv data/train.csv --oof-path <oof npz> \
  --data-dir data --config research/configs/v142_v141_submission_package.json \
  --output-dir artifacts/<run>

# v124 — public quadratic stack
python -m src.champion.v124_public_quadratic_stack \
  --parent-zip <v104 zip> --data-dir data \
  --config research/configs/v124_public_quadratic_stack.json \
  --output-dir artifacts/<run>

# v104 — source-stability mask
python -m src.champion.v104_source_stability_mask \
  --train-csv data/train.csv --contract-dir <contract dir> \
  --v103-dir <v103 run dir> --config research/configs/<v104 config> \
  --output-dir artifacts/<run>
```

`<...>` 입력은 고정된 외부 산출물이며 저장소 파일이 아닙니다. Git에서 제외한 자산의 당시 전달 절차는 [과거 현황 기록](../../docs/PROJECT_STATUS.md)에 남아 있습니다. 해시를 확인한 별도 자산이 있어야 재조립할 수 있습니다.

## 코드·자산 관리 기준

- 선택된 기준 자산을 덮어쓰지 않습니다. 새 `artifacts/<name>_<date>_NN/`에 빌드합니다.
- 당시 새 후보는 패키징 전 `research/configs/evaluation_v3.json`을 통과하고, 공개 결과를 확인한 뒤 기준 버전을 바꿨습니다.
- 이 폴더에 `src/archive/` import를 추가하지 않습니다. 종료 실험의 기능을 재사용해야 하면 공통 구현을 `src/core/`에 둡니다.
