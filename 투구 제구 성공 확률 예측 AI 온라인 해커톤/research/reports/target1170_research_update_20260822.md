# 1170 심층 연구 업데이트 — 2026-08-22

## 결론

- 현재 운영 champion은 계속 **Public 1158.0745556751**이다.
- 전달 파일은 과거 제출 ZIP을 요구하지 않는
  `artifacts/standalone_champion_1158/standalone_champion_1158.zip` 하나다.
- 2025 TrackMan profile 792개를 원본 공식 데이터에서 다시 만들었고, 배포 payload의
  gate 필드와 **완전히 일치**했다.
- 최종 TrackMan gate는 2024 전체에서 `+0.0138`, 2024 후반에서 `-0.1612`였다. Public
  `+0.1009`는 보존하지만 gate 강도를 확대할 근거는 없다.
- 새 직교 후보 v62 group-balanced ExtraTrees는 2024 전체 `-1.1415`, 후반
  `-0.0248`로 반전해 기각했다. 제출·standalone 후보 ZIP은 만들지 않았다.
- v63 `pitcher×balls>strikes` 순수 상호작용은 v50과 상관이 `0.006~0.009`로
  직교했지만, 두-origin 합의 후보가 `0/240`이고 2024 전체 gain도 `+0.0860`뿐이었다.
- v64 물리 교사→ID 제거 학생은 late-2023의 강한 선택 성적(`+12.2269`)을 2022와
  2024에 재현하지 못했다. full-2024 `-1.3183`, late-2024 `-1.5045`이며 세 seed가
  모두 음수여서 physical-first 학생 경로를 기각했다.
- 2026-08-22 기준 확인한 공개 대회 저장소 중 우리 점수보다 높은 검증 가능한 공개
  결과는 없었다. 공개 아이디어의 대부분은 이미 v17–v62에서 재구성했거나 시간 감사에서
  기각됐다.

1170까지 필요한 상승폭은 **11.9254443249**다. 현재 증거로는 calibration 미세조정이나
단일 lookup만으로 이 격차를 메울 가능성이 낮다. 다음 고가치 경로는 팀원별 exact OOF를
같은 행 순서로 모은 constrained blend와 StablePFN 원칙의 환경 안정 feature 선별이다.

## 1. 대회 규칙 감사

[DACON 공식 독립 예측 공지](https://dacon.io/competitions/official/236743/talkboard/417123)를
다시 확인했다. 이후 모든 후보에 다음을 코드 게이트로 적용한다.

- 한 test 행은 현재 행, 공식 train, train-only frozen artifact만 사용한다.
- 다른 test 행의 평균·빈도·순위·그룹·시퀀스·rolling/lag/누적값은 사용하지 않는다.
- 파일 앞쪽 행을 과거 투구처럼 간주하지 않는다.
- test 분포로 center, alpha, threshold, blend weight를 다시 정하지 않는다.
- 전체·singleton·shuffle·partition 실행의 동일 행 예측 차이는 `1e-12` 이하여야 한다.
- 외부 데이터는 사용하지 않는다. 공개 사전학습 weight도 라이선스와 운영진 허용 범위를
  모두 확인하기 전에는 후보에 넣지 않는다.

추가로 [DACON 운영진의 학습·추론 과정 검증 답변](https://dacon.io/competitions/official/236743/talkboard/417157)을
확인했다. Phase 3에서는 코드 재현뿐 아니라 주요 feature, 후처리값, 앙상블 비율과 모델
설정이 어떤 정보와 실험에서 도출됐는지까지 확인한다. 따라서 모든 배포 상수는 Public
점수 역산이 아닌 시간 OOF 근거와 선택 시점을 보고서에 남긴다.

현재 standalone champion은 정적 금지 연산 0개, singleton/full 최대 차이
`5.55e-17`, shuffle/full `5.55e-17`, partition/full `0`으로 통과했다.

## 2. v61 — 최종 0819 gate exact temporal OOF

### 재현 무결성

`src/archive/v61_final_gate_oof.py`는 공식 train과 TrackMan history에서 2023·2024·2025
시점 profile을 strict pre-origin 방식으로 다시 만든다. 2025 결과를 배포 payload와
비교한 뒤에만 과거 OOF를 신뢰한다.

| 검사 | 결과 |
|---|---:|
| 재구성 profile | 792 |
| 배포 profile | 792 |
| `(season, pitcher_id)` | exact match |
| `tm_linked` | exact match |
| `tm_pitcher_n` 최대 차이 | `0` |

### 시간 감사

| 축 | 전체 gain | 양수 월 | 최악 월 | 적용 도메인 gain |
|---|---:|---:|---:|---:|
| late-2023 선택 | +0.8309 | 66.7% | -0.6475 | R_ANCHOR +4.7782 |
| full-2024 감사 | **+0.0138** | 42.9% | -0.6103 | R_ANCHOR +0.0783 |
| late-2024 복제 | **-0.1612** | 50.0% | -0.7073 | R_ANCHOR -0.9222 |

gate는 평균 절대 이동이 `1e-4`보다 작은 보수적 보정이다. 실제 Public 이득은 유지하되,
OOF상 확대·다른 도메인 확장의 근거로 사용하지 않는다.

## 3. v62 — season×domain group-balanced ExtraTrees

### 가설과 설계

- 선수 ID를 제외해 새 선수·선수 이동에 대한 의존을 줄였다.
- 현재 행의 경기 상황, 공식 ASOF, 최근 경기 상태, 구종 구성만 사용했다.
- 세 직전 시즌을 사용하고 `uniform`과 `season×domain equal-risk`를 비교했다.
- 2022와 late-2023에서 동일한 risk/route/weight가 통과해야만 2024를 열었다.
- 64-tree squared-error ExtraTrees를 사용해 기존 boosting/lookup과 다른 분할 구조를
  만들었다.

선택상 최선은 `season_domain_equal`, `R_CORE`, weight `0.02`였다. 2022 gain은
`+4.4464`, late-2023은 `+4.5936`였지만 2022 최악 월이 `-2.4793`이라 exact consensus
통과 후보는 **0/40**이었다.

| 감사 축 | gain | 양수 월 | 최악 월 | R_CORE gain |
|---|---:|---:|---:|---:|
| full-2024 | **-1.1415** | 37.5% | -2.4854 | -1.6205 |
| late-2024 | **-0.0248** | 66.7% | -0.7387 | -0.0355 |

결론은 기각이다. group balancing이 과거 두 선택축의 평균은 개선했지만 2024 concept
shift를 해결하지 못했다. 실패 cache와 모델 예측은 삭제하고 코드·테스트·이 보고서만
남긴다.

## 4. v63 — pitcher×balls>strikes 순수 상호작용

공개 상위 해법에서 반복된 저차원 count 축을 외부 fitted value 없이 재구현했다. 각 완료
시즌의 train-only wave0 OOF 잔차에서 투수별 `balls > strikes`와 나머지 상태의 차이를
추정하고, 투수별 가중 평균이 정확히 0이 되도록 주효과를 제거했다. 전체 과거·최근 2년·
직전 1년, EB smoothing 4개, 도메인 4개, weight 5개를 2022와 late-2023에서 같은
레시피로 비교했다.

- 두-origin consensus 통과: **0/240**
- 선택상 최선: `all / smoothing=600 / F / weight=0.25`
- 선택 gain: 2022 `+0.1431`, late-2023 `-0.0385`
- v50 저랭크 correction 상관: full-2024 `0.0064`, late-2024 `0.0091`

| 감사 축 | gain | 양수 월 | 최악 월 | 적용 F gain |
|---|---:|---:|---:|---:|
| full-2024 | **+0.0860** | 50.0% | -0.4530 | +0.7306 |
| late-2024 | **+0.0594** | 33.3% | -0.3686 | +0.4782 |

신호는 기존 v50과 거의 무상관이지만 크기와 시간 안정성이 모두 부족하다. 2024에서 보인
F 소폭 이득을 보고 F 전용 강도나 smoothing을 다시 고르는 것은 사후 선택이므로 금지한다.

## 5. v64 — frozen physical-teacher student closure

v44의 물리 privileged-distillation 실험에서 `official_without_ids / ALL / weight=0.4`가
2022 학습→late-2023 선택에서 gain `+12.2269`, 전 월·전 도메인 양수였지만, 당시 목적이
profile 추가 효과 확인이라 2024가 열리지 않았다. v64는 이 레시피를 그대로 동결하고
2021→2022 강건성, 2023→2024 전체, 2024 전반→후반을 새로 학습했다. TrackMan 물리값은
labelled source의 교사에만 들어가고 학생 추론에는 공식 현재 행 특성만 사용했으며 선수 ID도
제거했다.

| 축 | gain | 양수 월 | 최악 월 | 최소 도메인 gain |
|---|---:|---:|---:|---:|
| historical full-2022 | **-1.2240** | 42.9% | -6.3011 | -4.1660 |
| full-2024, 3-seed 평균 | **-1.3183** | 37.5% | -3.0095 | -2.9877 |
| late-2024, 3-seed 평균 | **-1.5045** | 33.3% | -2.2047 | -4.3659 |

full-2024 seed gain은 `-1.2001/-1.3920/-1.3665`, late-2024는
`-1.6662/-1.4868/-1.3642`로 모두 음수였다. 교사 자체의 privileged gain은 각 source에서
`+348~+404`였지만, 위치·의도·현재 물리값이 없는 ID-free 학생으로는 안정적으로 전달되지
않았다. 이 경로는 가중치 축소 재탐색 없이 종료한다.

## 6. 팀 브랜치·OOF 감사

- 1158.07 팀 브랜치 커밋은 이미 `team/main`에 병합돼 있으며,
  현재 champion TrackMan gate의 코드·보고서다.
- 다른 팀 브랜치는 GitHub Actions workflow 삭제만 포함하고 모델 차이는 없다.
- 팀 정책상 모델·OOF 바이너리는 Git에 넣지 않으며, 로컬에도 현재 champion과 독립적인
  팀원 final OOF가 없다. 따라서 Public 점수만으로 blend weight를 정하지 않고 exact OOF
  전달 전까지 constrained blend를 보류한다.

## 7. 공개 대회 자료 감사

공개 저장소는 최신 방법의 구현 증거로만 참고했다. 공개 점수로 우리 blend weight를
역산하거나 공개 저장소의 test 분포 추정값을 가져오지 않았다.

| 공개 자료 | 확인한 결과·아이디어 | 우리 판단 |
|---|---|---|
| [x2-qp-cheese/LGAimers](https://github.com/x2-qp-cheese/LGAimers) | V18 Public 1081.68246; `pitcher×batter_hand`, `pitcher×pressure×batter_hand` | 우리 v17 핵심 계보와 동일. v32 temporal consensus까지 재검증 완료 |
| [hoo743-ui/LG_Aimers09](https://github.com/hoo743-ui/LG_Aimers09) | champion 1071.8146; pitcher/batter main effect와 `pitcher×balls>strikes` | 저차원 count 축은 참고 가치가 있으나 README 평가식이 공식 BSS와 불일치. LB 역산·test mean 경로는 공식 규칙상 제외 |
| [mk-isos/lg-aimers-9-pitch-control](https://github.com/mk-isos/lg-aimers-9-pitch-control) | Public 1053.861552; low-rank, TrackMan physical lookup, AR state. EXP-110은 2024 개선·2023 악화 | v50/v51로 핵심을 재구성했으나 기준 미달. EXP-110 recent pooled Brier 개선은 `6.8e-7`로 사실상 중립이고 공개 OOF/state가 없어 즉시 이식하지 않음 |
| [thisisacrane/lg-aimers9-pitch-control](https://github.com/thisisacrane/lg-aimers9-pitch-control) | CatBoost+LGB+isotonic, cold-start shrink, 2023 F break | pre-2023 F 제거가 Public을 개선하지 않아 hard regime drop은 제외 |
| [danny010712/LG-Aimers-9th-Hackathon](https://github.com/danny010712/LG-Aimers-9th-Hackathon) | 실패 유형 보조 모델과 global logit shift | failure mode는 v20/v52에 반영. test 성공률 추정·사후 shift는 사용하지 않음 |
| [IROHA0508/LG_Aimers_Hackathon](https://github.com/IROHA0508/LG_Aimers_Hackathon) | random CV 2411→Public 786 붕괴, walk-forward에서 RF/MLP 다양성 재평가 | 시간 검증 필요성을 재확인. v45/v62에서 neural/bagging 직교 후보를 별도 감사 |
| [ihabeu/lg-aimers-9th-kbo-pitch-control](https://github.com/ihabeu/lg-aimers-9th-kbo-pitch-control) | Bayesian hierarchy, regime, sequence, uncertainty 등 다수 실험 | 확인 가능한 Public 우위와 완전 OOF가 없어 아이디어 목록만 참고 |

공개 구현에서 가장 반복되는 교훈은 모델 복잡도보다 temporal validation, 확률 보정,
오차 다양성이다. 다만 우리 v45 TabM-mini, v50 저랭크, v51 AR, v57 공개 모델 blend,
v62 ExtraTrees가 모두 최종 외부 게이트를 통과하지 못했으므로 “다른 모델 하나 추가”만으로
1170을 주장할 수 없다.

## 8. 최신 1차 문헌과 적용 가능성

### 즉시 반영한 원칙

- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html):
  parameter-efficient MLP ensemble은 직교 오차 후보로 타당하다. 그러나 우리 v45
  TabM-mini는 full-2024 `-8.271`로 반전했으므로 같은 구조 확대는 후순위다.
- [TabDPT, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html):
  real-data pretraining과 retrieval/ICL의 강한 공개 TFM이다. 대회 외부 데이터 금지와
  pretrained-weight 허용 범위를 먼저 해결해야 하며, 147만 행과 L4 단일 ZIP 환경 때문에
  즉시 배포하지 않는다.
- [TabPFN v2, Nature 2025](https://www.nature.com/articles/s41586-024-08328-6):
  소규모·중간 크기 표에서 강하지만 핵심 benchmark는 10,000행 수준이다. 우리 전체 표를
  그대로 처리하기보다 train-only prototype/partition expert로만 검토 가능하며, v55
  prototype retrieval 실패 때문에 우선순위가 낮다.
- [TabSTAR, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/faf6e23e198314c7728eaa6ac44ae079-Paper-Conference.pdf):
  semantic/text field가 강점인데 이 대회는 익명 ID와 수치 상태 중심이라 적합도가 낮다.
- [StablePFN, KDD 2026](https://doi.org/10.1145/3770855.3818030):
  causal-aware stable prediction은 이번 핵심 병목인 season/domain shift와 방향이 맞다.
  다만 새 논문·사전학습 모델 자체를 바로 넣기보다, 안정적 원인 후보만 남기는 환경별
  feature selection과 worst-group 목적함수로 원칙을 이식하는 편이 안전하다.

### 야구 도메인 근거

- [xCTRL, 2025](https://arxiv.org/abs/2508.19184)는 투수별·구종별·타자 손·상황별로
  의도를 개인화해 제구 실행을 측정한다. 우리 데이터에는 현재 구종·실제 위치·포수 target이
  없으므로 xCTRL 자체는 구현할 수 없다. 대신 투수별 context partial pooling 원칙은
  v17/v50에 반영돼 있다.
- [Context-enhanced pitch location, Sports Engineering 2025](https://link.springer.com/article/10.1007/s12283-025-00497-5)는
  release position과 spin, 물리 문맥이 위치 예측에 중요함을 보인다. 이는 TrackMan을
  약한 보조 신호로 쓰는 근거지만, 위치·의도 정보가 없는 현재 profile의 한계도 설명한다.
- [Tabular data with temporal shift, ICML 2025](https://proceedings.mlr.press/v267/cai25j.html)는
  연속 timestamp의 추세·주기를 사용한다. 공식 행에는 season/month/weekday만 있고 정확한
  timestamp가 없으며 test 순서 복원은 금지되므로 논문의 핵심 방법은 적용하지 않는다.

문헌은 모델 이름을 채택하는 근거가 아니라 가설을 만드는 근거로 사용했다. 최종 채택은
항상 공식 데이터의 forward OOF와 행 독립성 검사로 결정한다.

## 9. 1170 후속 실행 우선순위

1. **팀원 exact OOF constrained blend**
   - 동일 `row_id`의 2022·2023·2024 OOF를 팀원 모델별로 확보한다.
   - 평균 gain이 아니라 season×month×domain 최악 손실을 제약한 nonnegative blend를 fit한다.
   - Public 점수만 있는 모델은 weight 학습에 사용하지 않는다.

2. **StablePFN 원칙을 이용한 shift-stable feature selection**
   - 사전학습 weight를 쓰지 않고, 2020→2021부터 2023→2024까지 계수/SHAP 방향이 같은
     row-local 변수만 남긴다.
   - ERM, season-domain equal risk, worst-group regularization을 같은 용량에서 비교한다.

3. **독립 기반모형 확보 후 worst-group constrained stack**
   - v45/v62/v64처럼 구조만 다른 후보가 아니라, 두 선택 origin에서 잔차 상관과 gain이
     함께 안정적인 모델만 stack bank에 넣는다.
   - 팀원 OOF가 없으면 먼저 feature-family 단위 OOF를 동일 row order로 생성한다.

4. **교차적합 calibration은 마지막 단계**
   - 새 직교 기반 모델이 확보된 뒤에만 beta/logit calibration을 fold 내부에서 fit한다.
   - 현재 champion 단독의 scalar spread/center 재탐색은 v59/v60 실패로 중단한다.

물리 교사 학생(v64), 물리 profile(v44), `pitcher×balls>strikes`(v63)는 완료·기각
목록으로 이동한다. 같은 구조의 강도·seed·도메인 재탐색은 하지 않는다.

## 10. 승격과 전달

새 후보는 다음을 모두 통과할 때만 처음부터 standalone ZIP으로 만든다.

- full-2024 gain `>= +5`
- 양수 월 `>= 75%`, 최악 월 `> -5`
- R_CORE/R_ANCHOR/F 모두 gain `>= 0`
- late-2024 gain `> 0`, 모든 월 양수
- 전체·singleton·shuffle·partition 행 독립성 통과
- ZIP 내부 `script.py`, `requirements.txt`, `model/`만으로 실행
- 부모 ZIP, 저장소 `src/`, 외부 API, test 집계 의존 없음

이번 주기의 v61만 기준 OOF로 유지하고 v62–v64 실패 cache와 학습 모델은 삭제한다.
현재 전달 대상은 standalone champion 1158 하나뿐이다.

## 11. 재현 명령

```powershell
python -m src.archive.v63_pitcher_balls_ahead_interaction `
  --project <private-project> --final-parent-dir artifacts/v61_final_gate_oof_20260822_01 `
  --output-dir artifacts/v63_pitcher_balls_ahead_20260822_01 --v50-dir <private-v50-dir>

python -m src.archive.v64_frozen_physical_teacher_student `
  --project <private-project> --alignment-dir <private-alignment-dir> `
  --final-parent-dir artifacts/v61_final_gate_oof_20260822_01 `
  --selection-metrics <private-v44-selection-metrics.csv> `
  --output-dir artifacts/v64_frozen_physical_teacher_student_20260822_01
```
