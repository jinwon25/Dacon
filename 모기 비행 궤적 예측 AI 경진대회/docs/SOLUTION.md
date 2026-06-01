# 모기 비행 궤적 예측 AI 경진대회 — 솔루션 (Private LB 0.703151, 2위)

> 데이콘 모기 비행 궤적 예측 AI 경진대회 · 2차 평가 제출 솔루션 자료 (2026-06)

---

## 1. 문제 정의

- **입력**: 40ms 간격으로 관측된 11개 시점의 3D 좌표 `(x, y, z)` (총 −400ms ~ 0ms 구간).
- **출력**: 마지막 관측 시점 **+80ms** 의 3D 좌표 `(x, y, z)`.
- **평가 지표**: **R-Hit@1cm** — 예측과 정답의 유클리드 거리가 **1cm 이내**인 샘플의 비율.
- **데이터 규모**: train 10,000 / test 10,000 궤적.

핵심 성질: 1cm 임계값의 **이진 hit 지표**이므로, 평균 오차(mm)를 줄이는 것보다 **1cm 경계에 걸친 샘플을 넘기는 것**이 점수를 좌우한다. 이 점이 전체 전략(앙상블 다양성 > 단일 모델 정확도)을 결정했다.

---

## 2. 최종 결과

| 구분 | 점수 | 순위 |
|---|---:|---:|
| **Private LB (최종)** | **0.703151** | **2위** |
| Public LB (최종 선택) | 0.7022 | — |

- 최종 선택 2슬롯: `submission_v157_ens3a0.40` / `submission_v157_ens3a0.45` (둘 다 Public 0.7022 → Private 0.703151).
- 시작점 대비: 단순 candidate-selection baseline(LB 0.6306) → **0.703151** (+0.073).
- 주요 plateau 돌파: 0.6888 → 0.6912 → 0.697 → 0.7016 → **0.7022/0.703151**.

---

## 3. 접근 개요 — "직교 메커니즘 다양성"의 앙상블

이 대회의 본질적 난점은 **데이터 천장**이었다. 단일 모델 계열을 아무리 정교화해도 LB 0.6888 부근에서 막혔다. 진단 결과 원인은 명확했다:

> 우리가 쌓은 40여 개 모델이 전부 **같은 baseline(Kalman 잔차)** 위에서 학습돼, 서로의 예측이 **상관계수 ~0.99** 로 묶여 있었다. 앙상블의 이득은 멤버 간 **직교성(다양성)** 에서 나오는데, 같은 메커니즘의 변종들은 아무리 많아도 새 정보를 주지 못한다.

따라서 점수를 움직인 모든 돌파는 **근본적으로 다른 예측 메커니즘(paradigm)을 새로 도입**한 순간이었다. 최종 솔루션은 **서로 직교하는 물리/학습 메커니즘들의 보수적 앙상블 + 직교 멤버의 수동 주입**이다.

```
                  ┌─ Pool A: Kalman 잔차 프레임 (BiGRU/TCN/Transformer/MDN)
   DE 블렌드 ─────┼─ Pool B: Neural ODE (위치+속도 6D 상태, RK4 적분)
   (base, OOF      ├─ Pool C: Frenet 3D-프레임 ODE + control-head 적분
    0.6831)        └─ (boundary refinement 변종들)
        │
        │   v157 = (1−α)·base + α·CREE_ensemble       ← 최종 수동 주입
        ▼
   CREE 회전물리 3-앙상블 (별도 메커니즘, base와 2.8mm 직교)
```

---

## 4. 핵심 빌딩 블록

### 4.1 Pool A — Kalman 잔차 프레임 (baseline paradigm)
- 마지막 속도 벡터로 yaw 정렬한 **canonical local frame** 에서, **Kalman 잔차**를 타깃으로 학습.
- 백본: BiGRU, TCN, Transformer, MDN-WTA(K-way Winner-Take-All).
- yaw 회전 + y-mirror 증강으로 회전 불변성 확보.
- **한계**: 멤버 간 상관 ~0.99 → 이 풀만으로는 LB 0.6888 천장.

### 4.2 Pool B — Neural ODE (1차 돌파, LB 0.6912)
- 타깃을 Kalman과 무관한 `y − last_obs` 로 바꿔 **완전히 다른 base**.
- **6D 상태(위치, 속도) + 학습된 감쇠 + neural acceleration field**, 80ms 구간을 **RK4 4-eval** 로 적분.
- 단독 OOF는 낮지만(0.66), base 풀과 **L2 ~2.2mm 직교** → DE 블렌더가 ~40% 가중치 부여 → plateau 0.6888 → **0.6912** 돌파.

### 4.3 Pool C — Frenet 프레임 / control-head (2차 돌파, LB 0.697)
- **Frenet 3D 프레임**: tangent(속도) · normal(가속도) · binormal 으로 만든 완전 3D 직교 프레임. yaw(xy)만 회전하는 v120과 달리 **z 처리가 근본적으로 달라** decorrelation 최대.
- **control-head 적분**: RK4 대신 NN이 가속도(control)를 출력하고 `p = v₀·T + ½·a·T²` **닫힌형 적분** → 다른 오차 구조.
- conservative 블렌드 OOF 0.6807, 신규 멤버 가중치 58% → LB **0.697** (변환률 +0.0165, 역대 최고).

### 4.4 CREE 회전물리 멤버 (최종 돌파, LB 0.7016 → 0.7022)
- **공개 Dacon 코드공유 baseline (HyperPhysics, 회전 기반 turn-rate 물리모델)** 을 우리 5-fold OOF 파이프라인에 포팅.
  - Rodrigues 회전으로 속도 벡터를 회전 + 학습된 angular velocity(omega) + EMA 속도/가속도 필터 + world-up 기반 프레임.
  - 우리 RK4 적분 계열과 메커니즘이 완전히 달라 **base와 2.82mm 직교**(우리 frenet 멤버끼리는 0.6mm).
- **3-CREE 앙상블** 로 강화: `dirnet(seed42)` + `dirnet(seed1)` + `3step-heading` 의 단순 평균. 앙상블은 내부 분산만 줄이고 **교차-paradigm 직교성은 보존**.

> **규칙 준수 메모 (회전물리 멤버 출처)**: 회전물리 멤버(내부 코드네임 `CREE` — **동명의 본 대회 참가자와는 무관**)는 대회 기간 중 Dacon **코드 공유 게시판에 공개되었던** 한 회전 기반 turn-rate 물리 baseline(HyperPhysics 계열)의 모델 구조를 참고했다(대회 규칙 8조 B항: 데이콘 플랫폼 공개 코드 공유 사용 허용). 해당 게시물은 **현재 삭제되어 링크를 제시할 수 없으나**, 핵심 메커니즘은 교과서적 물리(Rodrigues 회전 공식 + EMA 속도/가속도 필터)다. **공개된 모델 구조 코드를 포팅(구조 보존)하여 우리 5-fold CV 파이프라인·데이터에 연결**했고, **가중치는 차용하지 않고 본 대회 train만으로 from-scratch 학습**했다. **test 데이터는 학습에 일절 사용하지 않음**. 즉 회전물리 아키텍처는 공개 코드 기반이고(8조 B항 허용·인용), 그 위의 직교 멤버 앙상블·over-conversion 주입·전체 파이프라인이 본 솔루션의 독자 기여다. 사용 라이브러리(PyTorch/NumPy/SciPy/scikit-learn/LightGBM)는 모두 오픈소스(BSD/MIT/Apache)이며 외부 API·원격 모델을 사용하지 않았다.

---

## 5. 결정적 인사이트 — "OOF는 직교 멤버의 LB 프록시가 아니다"

이 대회에서 우리가 발견한 가장 중요한 사실:

1. **DE 블렌더(OOF 최적화)는 직교 멤버를 과소평가한다.**
   - OOF가 가장 높은 블렌드(`v148blend`, OOF **0.6831**, CREE 가중치 0.082) → LB 0.6996.
   - OOF가 **더 낮은** 블렌드(`v148creefwd α=0.25`, OOF 0.6808, raw CREE 25% **수동 주입**) → LB **0.7016**.
   - 즉 **OOF가 낮아도 더 직교한 변종이 LB를 +0.0020 이겼다.** 1cm hit 지표는 OOF blend CV가 잡지 못하는 다양성을 보상한다.

2. **수동 α 주입(over-conversion)이 정답.**
   - DE는 greedy하게 OOF에 맞추느라 CREE에 0.082밖에 주지 않지만, 1cm hit 지표에서는 직교 메커니즘을 더 많이(α=0.4) 섞을수록 경계 샘플을 더 많이 넘긴다.
   - α를 0.25 → 0.30 → 0.40 으로 키우며 Public 점수가 0.7016 → 0.7020 → 0.7022 로 **단조 상승**(over-conversion peak ≈ α 0.45 부근).

3. **레버는 "프레임/메커니즘 직교성"이지 encoder/integration 변종이 아니다.**
   - 같은 frenet 프레임에서 encoder만 바꾼 변종(Transformer/LRU/TCN, 0.6mm)은 DE가 전부 거부(포화).
   - CREE(2.82mm), Flow-직접(out-of-cluster) 처럼 **프레임/inductive-bias가 직교**한 것만 점수를 움직였다.

---

## 6. 최종 레시피 (재현 검증 완료)

```
base       = v148_reblend.py 의 DE 블렌드 출력 (frenet/neural conservative blend, OOF 0.6831)
cree_ens3  = mean(cree_xy2, cree_xy2s1, cree_xy2h3)
             = CREE HyperPhysics 회전물리 3-앙상블 (dirnet seed42 + dirnet seed1 + 3step-heading)

submission_v157_ens3a0.40 = 0.60·base + 0.40·cree_ens3     (Public 0.7022)
submission_v157_ens3a0.45 = 0.55·base + 0.45·cree_ens3     (Public 0.7022)
   └─ 둘 중 하나가 Private 0.703151 (최종 2위)
```

- 위 식은 `submissions/rebuild.py` 로 **외부 의존 없이 즉시 재현** 가능하며, 원본 제출과 **오차 < 0.001mm** 로 일치함을 확인했다.
- DE 블렌드(`scipy.differential_evolution`)는 seed 고정으로 **결정론적**, 멤버 신경망 학습도 seed 고정.

---

## 7. 검증된 실패(Dead-end) — 재시도 가치 없음

천장의 모든 각도를 막았음을 확인한 음성 결과(이후 같은 시도 금지):

| 시도 | 결과 |
|---|---|
| Disagreement selector (per-sample 모델 선택) | DEAD — route-acc 0.17 ≈ 무작위. 1cm 경계 불일치의 정답쪽은 피처로 예측 불가한 노이즈 |
| Mode-seeking / geometric-median 집계 | Δ ≤ 0 — active 멤버가 동질 군집이라 mode ≈ mean |
| IMM / analytic Constant-Turn 필터 | 0.24~0.55 < naive linear 0.58 — turn 신호 노이즈 취약 |
| Neural CDE (torchcde) | DEAD — OOF 0.2768, 학습 자체 실패 |
| Flow/SONODE 추가 주입 (4-mechanism) | Public 0.6994 < 순수 CREE 0.7022 — 해로움 확인, 폐기 |
| pseudo-label | OOF 과적합, LB 변환률 붕괴 |

교훈: **신규 멤버는 OOF-vs-TEST 예측 L2 일관성을 필수 검증**해야 한다(OOF는 정상인데 TEST만 깨지는 버그 클래스 존재).

---

## 8. 재현 방법 (요약)

상세는 `README.md` 의 "재현 방법" 절 참조.

- **빠른 재현 (권장, <1초, GPU 불필요)**: `python submissions/rebuild.py`
  → `inputs/` 의 base + 3-CREE 예측만으로 최종 제출 2개를 재생성하고 원본과 일치 검증.
- **전체 재현 (from scratch)**: `src/` 의 멤버 학습 스크립트 → `v148_reblend.py`(base) → `v157_final_submission.py`(최종).

환경: Python 3.11, PyTorch 2.7 (CPU), NumPy 2.x, SciPy, scikit-learn, LightGBM. (`requirements.txt`)

---

## 9. 회고

- **무엇이 통했나**: 단일 모델 정확도가 아니라 **직교 메커니즘의 다양성**. 점수를 움직인 건 항상 "새 base paradigm"(Kalman→ODE→Frenet→회전물리)이었다.
- **무엇이 안 통했나**: 같은 메커니즘의 encoder/하이퍼파라미터 변종, selector류, OOF만 보는 블렌드 튜닝.
- **핵심 기법**: 1cm hit 지표에서 직교 멤버를 **수동 α로 over-convert**. OOF/nested-CV가 보수적으로 보는 lift를 실측 LB가 크게 초과 달성.
