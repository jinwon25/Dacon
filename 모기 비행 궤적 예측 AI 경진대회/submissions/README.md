# 최종 제출 레시피와 재현 코드

기존 최종 결과: Private LB **0.703151 (2위)**.
공개본에는 `rebuild.py`와 `CHECKSUMS.sha256` 등 설명·무결성 기록만 남깁니다. 행별 입력·출력 CSV 및 과거 제출 CSV는 이력에서도 제외했습니다. 체크섬 목록은 원 비공개 파일의 기록입니다.

```text
cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)
v157_a040 = 0.60 * base + 0.40 * cree_ens3
v157_a045 = 0.55 * base + 0.45 * cree_ens3
```

`python rebuild.py`는 허가된 로컬 `inputs/base_v148blend.csv`, `cree_xy2.csv`, `cree_xy2s1.csv`, `cree_xy2h3.csv` 및 비교용 원 제출을 갖춘 경우에만 사용할 수 있습니다. 공개 clone만으로 실행되지 않습니다. 학습 코드와 공식 데이터 취득 절차는 [상위 README](../README.md)를 참고하세요.
