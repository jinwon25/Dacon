# v82 Public 결과와 1159 챔피언 승격 — 2026-08-22

## 결론

사전에 고정된 v57 strict 혼합을 v82 단일 ZIP으로 제출했고, 공식 Public
**1159.33522**를 기록했다. 기존 `1158.0745556751` 대비 **+1.2606643249**이므로
새 standalone 챔피언으로 승격한다.

- 제출 파일명: `submit_v82_probe.zip`
- 현재 전달 파일: `artifacts/standalone_champion_1159/standalone_champion_1159.zip`
- SHA-256: `6F6B208EAA7D71BA8F9AC1E60C5CF0E23011F09FD35A639CD66B2912FCDB4AB1`
- 크기: `34,557,910` bytes
- API 응답: `isSubmitted=true`, `detail=Success`
- 제출 ID: API·공개 페이지에서 미제공
- 확인 당시 순위/누적 제출: **13위 / 28회**
- 1170까지: **10.66478**

다른 DACON 대회 폴더의 인증 파일 재사용 위험을 사용자에게 고지한 뒤 명시 승인을
받아 제출했다. 토큰·팀 이름 값은 문서·로그에 기록하지 않았다.

## 로컬→Public 전이

| 항목 | gain |
|---|---:|
| final-parent full-2024 OOF | +1.2596972212 |
| 실제 Public | +1.2606643249 |
| Public - OOF | +0.0009671037 |
| 전이 비율 | 1.000767727 |

이번 한 건에서는 full-2024 OOF가 Public 효과 크기까지 거의 정확히 예측했다. 그러나
clean transfer 관측은 아직 충분하지 않으므로 이 비율을 다른 family의 점수 환산계수로
사용하지 않는다.

## 동결 레시피

`R_CORE = game_type == "R"`이면서 pitcher/batter team 어느 쪽도 `13`이 아닌 행으로
정의한다. 각 행은 다른 test 행을 참조하지 않는다.

```text
R_CORE: p_v82 = 0.90 * p_champion + 0.10 * p_EXP021_strict
그 외: p_v82 = p_champion
```

혼합 공간은 probability이고 weight `0.10`, route `R_CORE`, anchor team `13`은 2024
평가축을 열기 전에 고정된 v57 레시피다. Public 점수와 test 분포로 weight를 다시
선택하지 않았다.

## 선택 및 감사 근거

사전 선택축에서 strict 혼합은 두 축 모두 양수였다.

| 선택축 | gain | 최악 월 gain |
|---|---:|---:|
| full-2022 | +39.673598 | +3.280710 |
| late-2023 | +7.195258 | +5.479019 |

현재 최종 챔피언을 정확한 부모로 다시 계산한 OOF 결과는 다음과 같다.

| 감사축 | 전체 gain | R_CORE gain | 양수 월 비율 | 최악 월 gain |
|---|---:|---:|---:|---:|
| full-2024 | +1.259697 | +1.788249 | 50.0% | -9.176155 |
| late-2024 | +0.487466 | +0.696216 | 33.3% | -9.064120 |

R_ANCHOR와 F는 예측을 바꾸지 않아 domain gain이 정확히 `0`이다. 두 감사축 평균은
양수지만 월 안정성이 `configs/evaluation_v3.json`의 승격 기준에는 못 미친다. 따라서
이 결과는 사전 엄격 게이트를 통과하지 못한 탐색적 probe였지만 실제 Public 상회를
확인했다. 향후 일반 승격 기준은 유지하고 이번 결과만 예외적으로 기록한다.

## 단일 실행·규정 감사

ZIP 루트는 아래 세 항목뿐이며 과거 제출 ZIP이나 저장소 코드를 요구하지 않는다.

```text
model/
requirements.txt
script.py
```

챔피언 모델과 EXP-021 strict 모델, 두 추론 스크립트가 모두 `model/` 아래 포함되어
있다. 공식 train으로 만든 동결 history/profile만 읽으며 test 전체의 평균·빈도·순위·
rolling·lag·누적값을 사용하지 않는다. 공식 행 독립성 공지와 이상 답안 경고를 다시
확인했다.

- ZIP CRC: 통과
- 파일 수: 67
- 로컬 공식 smoke: 5행, 모두 R_CORE 경로 실행
- wrapper 대 직접 구성요소 수식 최대 오차: `0.0`
- 원본 대 shuffle 최대 오차: `0.0`
- 원본 대 partition 최대 오차: `0.0`
- 유효한 2025 smoke 행을 고유 row_id로 반복한 245,789행 규모 추론: `58.7940초`
- 확률 범위와 finite 검사: 통과
- 전체 회귀 테스트: `276 passed, 4 skipped`
- 공식 제한: 600초; 내부 soft guard: 120초

규정 근거:

- [DACON 평가 안내](https://dacon.io/competitions/official/236743/overview/evaluation)
- [평가 데이터 행별 독립 예측 재안내](https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent)
- [이상 답안·제출 주의 공지](https://dacon.io/competitions/official/236743/talkboard/417157?dtype=recent&page=1)

## 재현

```powershell
python -m src.v82_build_public_probe `
  --project "<private_project>" `
  --external-root "<private_project>/artifacts/external_mk_lg9" `
  --champion-zip artifacts/standalone_champion_1158/standalone_champion_1158.zip `
  --config configs/v82_public_probe.json `
  --output-dir artifacts/v82_public_probe_20260822_01 `
  --timeout 120
```

`manifest.json`에 고정 recipe, 구성요소 검증값, package SHA-256과 감사 결과가 저장된다.
성공 후 중간 strict ZIP·압축 해제본·외부 빌드 보고서는 제거되고 최종 ZIP과 manifest만
남는다. 제출 확인 뒤에는 동일 해시 파일을 `standalone_champion_1159/`로 승격했다.

## 후속 판정

1. v57 strict 같은 방향을 한 번 더 더하면 full/late-2024 gain이
   `-0.0763/-0.7871`이므로 weight 증량을 금지한다.
2. v50은 새 부모 위 full `+2.5390`이지만 late `+0.1340`, 최소 domain `-3.4279`라
   패키징하지 않는다.
3. v56 shared-horizon FM은 v82와 겹치지 않는 F-only 방향이다. 새 부모 위 full/late
   gain `+1.0181/+2.9166`, 양수 월 `75%/100%`, 최악 월 `-4.7126/+0.3596`이라
   2025 최종 모델 복원·standalone 검증의 다음 1순위로 둔다.
4. v56도 2022 선택축 최악 월 `-6.5616`, consensus gate `0/32`였다는 위험을 보존한다.
   Public을 보고 damping·rank·route를 바꾸지 않고 기존 고정 recipe만 재현한다.
5. 팀원 exact OOF, 새 직교 기반모형과 nested temporal runner는 병행한다.
