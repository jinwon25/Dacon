# [Private 2위 0.703151] 직교 메커니즘 앙상블 + CREE 수동주입 코드·솔루션 공유

> 데이콘 코드 공유 게시판 게시용 본문 초안. 상세 솔루션은 첨부 PDF(`SOLUTION.pdf`), 전체 코드는 첨부 패키지 참고.

안녕하세요. 모기 비행 궤적 예측 대회 **Private 2위(0.703151)** 솔루션을 공유합니다.

## 한 줄 요약
이 대회의 지표(R-Hit@1cm)는 **1cm 경계에 걸친 샘플을 넘기느냐**가 점수를 좌우하는 이진 hit 지표라, 단일 모델 정확도보다 **서로 직교한 예측 메커니즘의 다양성**이 핵심이라고 판단했습니다. 그래서 "근본적으로 다른 base paradigm"을 단계적으로 추가하며 plateau를 넘었습니다.

## 가장 중요했던 깨달음 — "OOF는 직교 멤버의 LB 프록시가 아니다"
40여 개 모델을 쌓아도 LB 0.6888에서 막혔는데, 원인은 전부 같은 Kalman 잔차 base 위에 있어 예측이 **상관 ~0.99**로 묶였기 때문이었습니다. 점수를 움직인 건 항상 **새 메커니즘**이었습니다:

1. **Neural ODE** (위치·속도 6D 상태 RK4 적분) → 0.6912
2. **Frenet 3D-프레임 / control-head 적분** → 0.697
3. **CREE 회전물리**(Rodrigues 회전 + 학습 angular velocity) → 0.7016 → **0.7022**

특히 결정적이었던 건, **OOF가 더 낮아도 더 직교한 멤버를 수동으로 더 많이 섞을수록(over-conversion) LB가 올라간다**는 점입니다. DE 블렌더는 OOF-greedy라 직교 멤버(CREE)에 weight를 0.08밖에 안 줬지만, 수동으로 α=0.40까지 주입했더니 1cm 경계 샘플을 더 많이 넘겨 LB가 단조 상승했습니다.

## 최종 레시피
```
cree_ens3 = mean(cree_xy2, cree_xy2_s1, cree_xy2_h3)   # CREE 회전물리 3-앙상블
base      = frenet/neural conservative DE 블렌드 (OOF 0.6831)
최종 제출  = (1 - α)·base + α·cree_ens3      (α = 0.40 / 0.45)
```
첨부 코드의 `submissions/rebuild.py` 하나로 최종 제출이 **외부 의존 없이 1초 만에 재현**(원본과 오차 < 0.001mm)됩니다.

## 규칙 준수
- **CREE 멤버**는 대회 기간 중 코드 공유 게시판에 공개되었던 HyperPhysics 회전물리 baseline의 모델 구조를 참고했습니다(규칙 8조 B항 공개 코드 공유 허용). 해당 게시물은 현재 삭제되어 링크를 제시할 수 없으나, 메커니즘은 표준 물리(Rodrigues 회전 + EMA 필터)이며 구조는 재구현 후 본 대회 train만으로 from-scratch 학습했습니다(가중치 차용 없음).
- test 데이터 학습 없음, 외부 데이터 없음, 원격 API 모델 없음(전부 로컬 실행), 시드 고정 재현 가능.

## 검증된 dead-end (시간 아끼시라고 공유)
disagreement selector(route-acc ≈ 무작위), mode-seeking/geometric-median, IMM/analytic-turn, Neural CDE(학습 실패), Flow/SONODE 추가 주입, 같은 프레임의 encoder 변종(전부 blend weight 0). 자세한 이유는 PDF §7.

읽어주셔서 감사합니다. 질문 환영합니다 🙏
