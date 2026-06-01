# 아침 브리프 (2026-06-01, 마감 10:00) — #1 도전 제출 플랜

## 현재 상태
- **최고: `submission_v148creefwd_a0.25` = LB 0.7016 (~#5)**. 이미 제출됨.
- #1 CREE = 0.7026 (**+0.0010 차이**). 우리가 #5까지 올라옴 (시작 #27).

## ⚠️ 먼저 할 것 (무손실)
**Dacon 최종 선택 2슬롯 중 하나에 `submission_v148creefwd_a0.25_oof0.6808.csv`(0.7016) 지정.** floor 확보.

## 핵심 레버 (밤새 검증)
**OOF 무시, decorrelated 메커니즘을 수동 주입 → over-convert.** 3개 독립 메커니즘 확보:
1. **frenet-RK4** (우리 기존, 적분)
2. **CREE-회전물리** (HyperPhysics, 공개 baseline) — frenet과 2.82mm 직교, 0.7016의 주역
3. **Flow-직접** (밤새 신규, 적분/회전 아님) — frenet 2.17mm, CREE 3.04mm 직교 (out-of-cluster 성공)
- 강화 CREE(풀증강)는 효과 없어 폐기. boundary는 직교성 죽이므로 **raw 멤버가 주입원**.

## 추천 제출 순서 (5슬롯, 전부 free-roll — 0.7016 floor 보존)
**업데이트: CREE 시드앙상블(seed42+1) 완성 — OOF 0.6701→0.6723↑ + frenet 직교 2.50mm 보존 = 더 강한 주입원.** 아래는 앙상블 CREE 기반(v153) 후보.

**최종: 4개 독립 메커니즘 + 3-CREE 앙상블** — frenet-RK4 + CREE-회전 + Flow-직접 + SONODE-학습v0. CREE는 **3-앙상블(dirnet seed42+1 + 3step heading) OOF 0.6744**(단일 0.6701보다↑, frenet 2.46mm 직교) = 역대 최강 주입원.

### ★ 추천 5슬롯 (v156 = 최강 3-CREE 앙상블 기반)
| # | 파일 | 구성 | 의도 |
|---|---|---|---|
| 1 | `submission_v156_ens3a0.25.csv` | 3CREE앙상블 0.25 (0.7016 recipe, CREE 최강화) | **가장 안전 ≥0.7016, 최유력** |
| 2 | `submission_v156_4way.csv` | frenet+3CREE0.22+Flow0.12+SONODE0.10 | **4-메커니즘 #1 베팅** (최대 다양성) |
| 3 | `submission_v156_ens3a0.3.csv` | 3CREE앙상블 0.30 | α>0.25 효과 |
| 4 | `submission_v156_4wayB.csv` | frenet+3CREE0.26+Flow0.14+SONODE0.12 | 대담한 4-way |
| 5 | `submission_v153_3way.csv` | frenet+2CREE0.28+Flow0.15 | 3-메커니즘 (대안) |

**학습 흐름:** 1(최강CREE가 0.7016 상회?) + 2(4메커니즘이 더?) → 2>1이면 다양성↑(4번) / 1>2면 α 미세조정. 점수 회신 주시면 제가 최적 조합 추가 생성.

> 백업 다수: `v154_*`, `v153_ensa0.35`, `v148creefwd_a0.3`, `v143bcree_*` 등 `open/submission_v15*.csv`.

## 제출 후
점수 알려주시면:
- 0.7016 넘는 것 발견 → 그걸 + 0.7016을 최종 2슬롯 (or 더 높은 둘)
- 다 0.7016 이하 → 0.7016 유지 (floor)
- 패턴 보고 **추가 주입비(α/β) 미세조정** 또는 **4번째 out-of-cluster 메커니즘** 투입 가능

## 백업 후보 (필요시)
- CREE α: a0.18, a0.4 / v143b base: v143bcree a0.2/0.25/0.3 / Flow: winnerflow b0.08/0.15 / cree30flow b0.1/0.12

## 규칙 메모
- CREE/Flow 모두 from-scratch 또는 **공개 Dacon 코드공유 기반**(정당). test 학습 없음. 2차평가 시 CREE 공개baseline 출처 명시 필요.
