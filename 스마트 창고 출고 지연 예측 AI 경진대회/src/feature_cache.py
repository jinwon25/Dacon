"""피처 캐시.

train/test 피처를 한 번만 add_features로 빌드하고 parquet으로 저장.
이후 모델 학습은 캐시된 parquet을 바로 로드 → add_features 시간 절약.
"""

from pathlib import Path

import pandas as pd

from train_solution import add_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data/cache"


def build_and_cache(data_dir: Path = PROJECT_ROOT / "data", force: bool = False) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_pq = CACHE_DIR / "train_features.parquet"
    test_pq = CACHE_DIR / "test_features.parquet"

    if train_pq.exists() and test_pq.exists() and not force:
        print(f"cache exists at {CACHE_DIR}, skipping (use force=True to rebuild)")
        return

    print("loading raw data")
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    layout = pd.read_csv(data_dir / "layout_info.csv")

    print("building train features")
    train = add_features(train_raw, layout)
    print("building test features")
    test = add_features(test_raw, layout)

    print(f"saving train ({train.shape}) -> {train_pq}")
    train.to_parquet(train_pq, index=False)
    print(f"saving test ({test.shape}) -> {test_pq}")
    test.to_parquet(test_pq, index=False)
    print("done")


def load_cached(data_dir: Path = PROJECT_ROOT / "data", cluster: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster:
        train_pq = CACHE_DIR / "train_features_cluster.parquet"
        test_pq = CACHE_DIR / "test_features_cluster.parquet"
        if not (train_pq.exists() and test_pq.exists()):
            raise FileNotFoundError(f"cluster cache not found, run: python layout_cluster.py")
    else:
        train_pq = CACHE_DIR / "train_features.parquet"
        test_pq = CACHE_DIR / "test_features.parquet"
        if not (train_pq.exists() and test_pq.exists()):
            build_and_cache(data_dir)
    train = pd.read_parquet(train_pq)
    test = pd.read_parquet(test_pq)
    # categorical 복원
    from train_solution import CAT_COLS
    extra_cat = ["layout_cluster"] if cluster else []
    for c in list(CAT_COLS) + extra_cat:
        if c in train.columns:
            train[c] = train[c].astype("category")
        if c in test.columns:
            test[c] = test[c].astype("category")
    return train, test


if __name__ == "__main__":
    build_and_cache(force=True)
