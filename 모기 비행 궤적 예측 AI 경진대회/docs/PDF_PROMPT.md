# claude.ai 웹에 붙여넣을 "방법론 PDF(PPT) 제작" 프롬프트

> 아래 `=== 프롬프트 시작 ===` ~ `=== 프롬프트 끝 ===` 사이를 **그대로 복사**해서 claude.ai(웹)에 붙여넣으세요.
> claude.ai가 PPTX(또는 슬라이드형 PDF)를 만들어 줍니다. 표지의 닉네임/날짜만 본인 것으로 바꾸면 됩니다.

---

=== 프롬프트 시작 ===

너는 데이터 사이언스 발표자료 디자이너야. 아래 내용을 바탕으로 **데이콘 코드 공유 게시판에 첨부할 "방법론 발표자료"** 를 만들어 줘.

**산출물 요구사항**
- 형식: **PPTX 슬라이드 덱** (가능하면 .pptx 파일로. 어려우면 슬라이드형 PDF). 14장 내외.
- 언어: **한국어**. 청중: 데이콘 참가자/리뷰어(머신러닝 이해 있음).
- 톤: 기술적·간결·자신감 있게. 불릿 위주, 슬라이드당 핵심 3~5개. 숫자(점수)는 크게 강조.
- 디자인: 깔끔한 테크 컨퍼런스 스타일. 포인트 컬러 1~2개(예: 블루 계열). 표/다이어그램 적극 사용. 슬라이드 하단에 페이지 번호.
- 핵심 메시지가 한눈에 보이도록 제목을 "결론형"으로 (예: "Neural ODE — 1차 plateau 돌파").

**대회 한 줄 소개**: 모기의 3D 비행 궤적에서 40ms 간격 11개 관측점으로 마지막 +80ms 좌표를 예측. 평가지표 **R-Hit@1cm**(예측-정답 유클리드 거리 1cm 이내면 hit, 비율이 높을수록 좋음). 최종 **Private 2위, 0.703151**.

---

아래 슬라이드 구성과 내용을 그대로 채워서 만들어 줘 (문구는 자연스럽게 다듬어도 됨):

**1. 표지**
- 제목: "모기 비행 궤적 예측 — Private 2위 (0.703151)"
- 부제: "직교 메커니즘 다양성의 앙상블"
- 작성자: [닉네임], 날짜: 2026-06 / 데이콘 코드 공유

**2. 문제 정의 & 지표**
- 입력: 40ms 간격 11개 시점의 3D 좌표 (−400ms~0ms). 출력: 마지막 관측 +80ms의 (x,y,z). 데이터 train/test 각 10,000.
- 지표 R-Hit@1cm = 1cm 이내 hit 비율.
- **핵심 성질(전략 결정의 근거)**: 1cm 임계값의 **이진 hit 지표**라, 평균 오차(mm) 최소화보다 **1cm 경계에 걸친 샘플을 넘기는 것**이 점수를 좌우 → "단일 모델 정확도 < 앙상블 다양성" 전략으로 직결.

**3. 핵심 통찰 — 데이터 천장의 정체**
- 40여 개 모델을 쌓아도 LB 0.6888에서 막힘.
- 원인: 전부 같은 **Kalman 잔차 base** 위에서 학습 → 예측이 **상관계수 ~0.99**로 묶임.
- 앙상블 이득은 멤버 간 **직교성**에서 나온다 → 같은 메커니즘 변종은 아무리 많아도 새 정보 0.
- 결론: **근본적으로 다른 예측 메커니즘(paradigm)을 새로 도입**할 때만 점수가 움직인다.

**4. 전체 아키텍처 (다이어그램으로)**
- DE 블렌드가 4개의 직교 풀을 통합 → base(OOF 0.6831) → 최종 CREE 수동 주입.
- 다이어그램 텍스트:
  - Pool A: Kalman 잔차 프레임 (BiGRU/TCN/Transformer/MDN)
  - Pool B: Neural ODE (위치+속도 6D 상태, RK4 적분)
  - Pool C: Frenet 3D-프레임 ODE + control-head 닫힌형 적분
  - → DE 블렌드(base) → **v157 = (1−α)·base + α·CREE_ens3** (α=0.40/0.45)
  - CREE 회전물리 3-앙상블 (base와 2.82mm 직교)

**5. Pool A — Kalman 잔차 프레임 (baseline)**
- 마지막 속도로 yaw 정렬한 canonical local frame에서 **Kalman 잔차**를 타깃으로 NN 학습.
- 백본: BiGRU/TCN/Transformer/MDN-WTA. yaw+y-mirror 증강.
- 한계: 멤버 간 corr ~0.99 → 이 풀만으로는 **LB 0.6888 천장**.

**6. Pool B — Neural ODE (1차 돌파, LB 0.6912)**
- 타깃을 Kalman 무관 `y − last_obs`로 → 완전히 다른 base.
- 6D 상태(위치·속도) + 학습된 감쇠 + neural acceleration field, 80ms를 **RK4 4-eval**로 적분.
- 단독 OOF 0.66로 낮지만 base 풀과 **L2 ~2.2mm 직교** → DE가 ~40% 가중 → **0.6888 → 0.6912**.

**7. Pool C — Frenet 프레임 / control-head (2차 돌파, LB 0.697)**
- **Frenet 3D 프레임**: tangent·normal·binormal 완전 3D 직교 프레임 → z 처리가 근본적으로 달라 decorrelation 최대.
- **control-head**: RK4 대신 NN이 가속도를 출력하고 `p = v₀·T + ½·a·T²` **닫힌형 적분** → 다른 오차 구조.
- 신규 멤버 가중 58% → **LB 0.697** (변환률 +0.0165, 역대 최고).

**8. CREE 회전물리 (최종 돌파, LB 0.7016 → 0.7022)**
- 공개 Dacon 코드공유 baseline(HyperPhysics, 회전 turn-rate 물리)을 우리 5-fold OOF 파이프라인에 포팅.
- Rodrigues 회전 + 학습 angular velocity + EMA 필터 + world-up 프레임 → **base와 2.82mm 직교**.
- `dirnet(s42)+dirnet(s1)+3step-heading` **3-앙상블**(내부 분산만 줄이고 교차-paradigm 직교성 보존).

**9. 결정적 인사이트 — "OOF는 직교 멤버의 LB 프록시가 아니다" (가장 중요한 슬라이드, 강조)**
- OOF 최고 블렌드(OOF 0.6831, CREE weight 0.082) → LB 0.6996.
- OOF가 **더 낮은** 블렌드(raw CREE 25% **수동 주입**, OOF 0.6808) → LB **0.7016**.
- 즉 **더 낮은 OOF + 더 직교한 변종이 LB를 +0.0020 이긴다.** 1cm hit 지표는 OOF blend CV가 못 잡는 다양성을 보상.
- **over-conversion**: DE는 OOF-greedy라 CREE에 0.08만 주지만, 수동 α=0.40까지 키우니 Public 0.7016→0.7020→0.7022 **단조 상승**.

**10. 최종 레시피 (코드 블록 스타일)**
```
cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)     # CREE 회전물리 3-앙상블
base      = frenet/neural conservative DE 블렌드 (OOF 0.6831)
v157_a040 = 0.60·base + 0.40·cree_ens3                 # Public 0.7022
v157_a045 = 0.55·base + 0.45·cree_ens3                 # Public 0.7022  → Private 0.703151
```
- 첨부 노트북의 마지막 셀 하나로 최종 제출이 **외부 의존 없이 1초 만에 재현**(원본과 오차 < 0.001mm).

**11. 점수 진행 (표)**
| 단계 | LB |
|---|---|
| 시작 baseline (candidate-selection) | 0.6306 |
| Kalman 잔차 NN 풀 | 0.6888 (plateau) |
| + Neural ODE | 0.6912 |
| + Frenet / control-head | 0.697 |
| + CREE 회전물리 (수동 α 주입) | 0.7016 → 0.7022 |
| **최종 (Private)** | **0.703151 (2위)** |

**12. 검증된 dead-end (시간 아끼시라고 공유 — 표)**
| 시도 | 결과 |
|---|---|
| Disagreement selector (per-sample 모델 선택) | DEAD (route-acc ≈ 무작위) |
| Mode-seeking / geometric-median | Δ ≤ 0 (동질 군집) |
| IMM / analytic Constant-Turn 필터 | naive linear보다 나쁨 |
| Neural CDE (torchcde) | 학습 실패 (OOF 0.28) |
| Flow/SONODE 추가 주입 | 순수 CREE보다 나쁨, 폐기 |
| 같은 frenet 프레임 encoder 변종 | DE weight 0 (포화) |
| pseudo-label | OOF 과적합, LB 변환률 붕괴 |

**13. 규칙 준수**
- 회전물리 멤버(내부 코드네임 CREE — **동명의 본 대회 참가자와 무관**)는 대회 기간 중 코드 공유 게시판에 공개됐던 회전물리 baseline(HyperPhysics 계열)의 **모델 구조를 참고**(규칙 8조 B항 공개 코드 공유 허용). 게시물은 현재 삭제되어 링크 제시 불가, 메커니즘은 표준 물리(Rodrigues 회전 + EMA 필터).
- **공개 모델 구조 코드를 포팅(구조 보존)** 해 우리 CV·데이터에 연결, **가중치 차용 없이 train만으로 from-scratch 학습**. 직교 멤버 앙상블·over-conversion 주입·전체 파이프라인이 독자 기여.
- test 학습 없음 · 외부 데이터 없음 · 원격 API 없음(전부 로컬) · 시드 고정 재현 가능. 라이브러리 전부 오픈소스(PyTorch/NumPy/SciPy/scikit-learn/LightGBM).

**14. 회고 / 교훈**
- 통한 것: 단일 모델 정확도가 아니라 **직교 메커니즘의 다양성**. 점수를 움직인 건 항상 "새 base paradigm".
- 안 통한 것: 같은 메커니즘의 encoder/하이퍼파라미터 변종, selector류, OOF만 보는 블렌드 튜닝.
- 한 줄 교훈: **1cm hit 지표에서는 직교 멤버를 수동 α로 over-convert하라. OOF/nested-CV가 보수적으로 보는 lift를 실측 LB가 크게 초과 달성한다.**

마지막으로, 슬라이드 9(결정적 인사이트)와 슬라이드 4(아키텍처 다이어그램)를 시각적으로 가장 임팩트 있게 디자인해 줘. 완성되면 .pptx(또는 PDF) 파일로 다운로드할 수 있게 해 줘.

=== 프롬프트 끝 ===
