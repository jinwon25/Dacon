"""DACON 식음업장 메뉴 수요 예측 — 공식 베이스라인 (메뉴별 개별 MultiOutput LSTM).

원본: 대회 제공 baseline (`trial.py`). 데이터 경로를 `data/` 구조에 맞게 정정.
프로젝트 루트 어디에서 실행해도 동작하도록 pathlib 기준 경로 사용.

    python src/baseline_lstm.py   # -> submissions/baseline_submission.csv
"""
import os
import random
import glob
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SUBM = ROOT / "submissions"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 16, 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiOutputLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=7):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # (B, output_dim)


def train_lstm(train_df):
    trained_models = {}
    for store_menu, group in tqdm(train_df.groupby("영업장명_메뉴명"), desc="Training LSTM"):
        store_train = group.sort_values("영업일자").copy()
        if len(store_train) < LOOKBACK + PREDICT:
            continue

        features = ["매출수량"]
        scaler = MinMaxScaler()
        store_train[features] = scaler.fit_transform(store_train[features])
        train_vals = store_train[features].values  # (N, 1)

        X_train, y_train = [], []
        for i in range(len(train_vals) - LOOKBACK - PREDICT + 1):
            X_train.append(train_vals[i:i + LOOKBACK])
            y_train.append(train_vals[i + LOOKBACK:i + LOOKBACK + PREDICT, 0])

        X_train = torch.tensor(np.array(X_train)).float().to(DEVICE)
        y_train = torch.tensor(np.array(y_train)).float().to(DEVICE)

        model = MultiOutputLSTM(input_dim=1, output_dim=PREDICT).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(EPOCHS):
            idx = torch.randperm(len(X_train))
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = idx[i:i + BATCH_SIZE]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                output = model(X_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        trained_models[store_menu] = {
            "model": model.eval(),
            "scaler": scaler,
            "last_sequence": train_vals[-LOOKBACK:],
        }
    return trained_models


def predict_lstm(test_df, trained_models, test_prefix: str):
    results = []
    for store_menu, store_test in test_df.groupby("영업장명_메뉴명"):
        if store_menu not in trained_models:
            continue
        model = trained_models[store_menu]["model"]
        scaler = trained_models[store_menu]["scaler"]

        store_test_sorted = store_test.sort_values("영업일자")
        recent_vals = store_test_sorted["매출수량"].values[-LOOKBACK:]
        if len(recent_vals) < LOOKBACK:
            continue

        recent_vals = scaler.transform(recent_vals.reshape(-1, 1))
        x_input = torch.tensor([recent_vals]).float().to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(x_input).squeeze().cpu().numpy()

        restored = []
        for i in range(PREDICT):
            dummy = np.zeros((1, 1))
            dummy[0, 0] = pred_scaled[i]
            restored.append(max(scaler.inverse_transform(dummy)[0, 0], 0))

        pred_dates = [f"{test_prefix}+{i + 1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, restored):
            results.append({"영업일자": d, "영업장명_메뉴명": store_menu, "매출수량": val})
    return pd.DataFrame(results)


def convert_to_submission_format(pred_df, sample_submission):
    pred_dict = dict(zip(zip(pred_df["영업일자"], pred_df["영업장명_메뉴명"]), pred_df["매출수량"]))
    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = str(final_df.loc[row_idx, "영업일자"])
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)
    return final_df


def main():
    train = pd.read_csv(DATA / "train" / "train.csv")
    trained_models = train_lstm(train)

    all_preds = []
    for path in sorted(glob.glob(str(DATA / "test" / "TEST_*.csv"))):
        test_df = pd.read_csv(path)
        test_prefix = re.search(r"(TEST_\d+)", os.path.basename(path)).group(1)
        all_preds.append(predict_lstm(test_df, trained_models, test_prefix))
    full_pred_df = pd.concat(all_preds, ignore_index=True)

    sample_submission = pd.read_csv(DATA / "sample_submission.csv")
    submission = convert_to_submission_format(full_pred_df, sample_submission)
    SUBM.mkdir(exist_ok=True)
    out = SUBM / "baseline_submission.csv"
    submission.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
