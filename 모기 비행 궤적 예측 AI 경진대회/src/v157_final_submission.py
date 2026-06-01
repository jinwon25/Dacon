"""v157_final_submission — 최종 제출(Private LB 0.703151, 2등) 재현 스크립트.

최종 레시피 (역설계로 검증 완료, 오차 < 0.001mm):

    base       = v148_reblend.py 의 DE 블렌드 출력 (frenet/neural conservative blend, OOF 0.6831)
    cree_ens3  = mean(cree_xy2, cree_xy2s1, cree_xy2h3)
                 = CREE HyperPhysics 회전물리 3-앙상블 (dirnet seed42 + dirnet seed1 + 3step-heading)
    v157_aX    = (1 - X) * base + X * cree_ens3       # 순수 CREE α 주입 (over-conversion)

최종 선택 2슬롯 (Dacon):
    submission_v157_ens3a0.4.csv   = 0.60*base + 0.40*cree_ens3   (Public 0.7022)
    submission_v157_ens3a0.45.csv  = 0.55*base + 0.45*cree_ens3   (Public 0.7022)
    -> 둘 중 하나가 Private 0.703151 (최종 2등) 을 기록.

입력(상대경로):
    cache/cree_xy2_state.npz, cache/cree_xy2s1_state.npz, cache/cree_xy2h3_state.npz  (CREE 멤버 test 예측)
    open/submission_v148blend_oof0.6831.csv  (base 블렌드; v148_reblend.py 산출물)
    open/sample_submission.csv               (id 컬럼)

사용:
    python scripts/v157_final_submission.py            # 최종 제출 CSV 2개 생성 + 검증
    python scripts/v157_final_submission.py --verify   # 기존 제출과 일치 검증만

개발환경: Python 3.11 / numpy 2.x / pandas 2.x  (CPU only, GPU 불필요)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA = Path("data")
CACHE = Path("data/cache")
BASE_CSV = DATA / "submission_v148blend_oof0.6831.csv"
CREE_TAGS = ["cree_xy2", "cree_xy2s1", "cree_xy2h3"]   # dirnet s42, dirnet s1, 3step-heading
ALPHAS = [0.40, 0.45]


def _load_xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path)[["x", "y", "z"]].to_numpy()


def load_cree_ensemble() -> np.ndarray:
    """3개 CREE 멤버 test 예측의 단순 평균. cache npz 우선, 없으면 open/ CSV로 폴백."""
    preds = []
    for tag in CREE_TAGS:
        npz = CACHE / f"{tag}_state.npz"
        csv = DATA / f"submission_{tag}.csv"
        if npz.exists():
            preds.append(np.load(npz)["test_global"].astype(np.float64))
        elif csv.exists():
            preds.append(_load_xyz(csv).astype(np.float64))
        else:
            raise FileNotFoundError(f"CREE 멤버 누락: {npz} / {csv} (먼저 v148_cree_xy2.py 학습 필요)")
    return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="기존 제출 CSV와 일치만 검증")
    args = ap.parse_args()

    if not BASE_CSV.exists():
        raise FileNotFoundError(
            f"base 누락: {BASE_CSV}\n  -> 먼저 `python scripts/v148_reblend.py` 로 생성하세요."
        )
    sub = pd.read_csv(DATA / "sample_submission.csv")
    base = _load_xyz(BASE_CSV).astype(np.float64)
    ens3 = load_cree_ensemble()
    assert base.shape == ens3.shape, (base.shape, ens3.shape)

    for a in ALPHAS:
        pred = (1.0 - a) * base + a * ens3
        out = DATA / f"submission_v157_ens3a{a:.2g}.csv"
        df = pd.DataFrame({"id": sub["id"], "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
        existing = out if out.exists() else None
        if not args.verify:
            df.to_csv(out, index=False)
            print(f"[saved] {out.name}  (alpha={a})", flush=True)
        if existing is not None:
            old = _load_xyz(existing)
            d = np.linalg.norm(pred - old, axis=-1).max() * 1000
            print(f"[verify] {out.name}: max diff vs existing = {d:.5f} mm "
                  f"({'MATCH' if d < 0.01 else 'MISMATCH'})", flush=True)

    print("\n[done] v157 최종 제출 재현 완료. submission_v157_ens3a0.4 / a0.45 가 "
          "Private 0.703151(2등) 후보입니다.", flush=True)


if __name__ == "__main__":
    main()
