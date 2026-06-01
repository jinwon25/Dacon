# 최종 제출 패키지

Private LB **0.703151 (2위)** 를 기록한 최종 제출과 자급식 재현 패키지.

## 파일

| 파일 | 설명 |
|---|---|
| `submission_v157_ens3a0.40_FINAL.csv` | ★ 최종 선택 슬롯 1 (`0.60·base + 0.40·CREE_ens3`, Public 0.7022) |
| `submission_v157_ens3a0.45_FINAL.csv` | ★ 최종 선택 슬롯 2 (`0.55·base + 0.45·CREE_ens3`, Public 0.7022) |
| `rebuild.py` | 자급식 재현/검증 (외부 의존 없음) |
| `inputs/base_v148blend.csv` | base 블렌드 예측 (frenet/neural conservative DE blend, OOF 0.6831) |
| `inputs/cree_xy2.csv` | CREE 회전물리 멤버 (dirnet, seed42) |
| `inputs/cree_xy2s1.csv` | CREE 회전물리 멤버 (dirnet, seed1) |
| `inputs/cree_xy2h3.csv` | CREE 회전물리 멤버 (3step-heading) |
| `CHECKSUMS.sha256` | 입력/출력 무결성 |
| `historical/` | 과거 LB-실측 후보 (0.6770~0.697 시기) |

## 재현

```bash
python rebuild.py
```

`inputs/` 의 base + 3-CREE 예측만으로 두 최종 제출을 재생성하고 원본과 일치(오차 < 0.001mm)를 검증한다. GPU 불필요, <1초.

레시피:

```
cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)
v157_a040 = 0.60 * base + 0.40 * cree_ens3
v157_a045 = 0.55 * base + 0.45 * cree_ens3
```

학습부터의 전체 재현은 상위 `README.md` 의 "재현 방법 §2" 참고.
