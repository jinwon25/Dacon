"""Bidirectional GRU over 25-slot scenario sequence (PyTorch CPU).

CNN과 다른 시퀀스 모델링 — RNN의 sequential dependency가 다른 패턴 학습 가능.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from train_solution import (ID_COL, GROUP_COL, TARGET, CAT_COLS,
                            LAYOUT_STATIC_COLS, LAYOUT_DERIVED_COLS,
                            make_folds, seed_everything)
from feature_cache import load_cached
from train_seq_cnn import ScenarioDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


class SeqGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden, hidden_size=hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        h = self.encoder(x)
        h, _ = self.gru(h)
        return self.head(h).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "models/seqgru_seed42"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--clip-percentile", type=float, default=99.0)
    parser.add_argument("--drop-layout-features", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_path(args.data_dir)

    print("loading cached features")
    train, test = load_cached(data_dir)
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    drop_cols = [ID_COL, GROUP_COL, TARGET, "layout_id", "layout_type"]
    if args.drop_layout_features:
        drop_cols.extend(LAYOUT_STATIC_COLS)
        drop_cols.extend(LAYOUT_DERIVED_COLS)
    feature_cols = [c for c in train.columns if c not in drop_cols]
    print(f"features: {len(feature_cols)}")

    fill = train[feature_cols].median(numeric_only=True)
    Xall_train = train[feature_cols].fillna(fill).to_numpy(dtype=np.float32)
    Xall_test = test[feature_cols].fillna(fill).to_numpy(dtype=np.float32)
    upper = np.percentile(Xall_train, args.clip_percentile, axis=0)
    lower = np.percentile(Xall_train, 100 - args.clip_percentile, axis=0)
    Xall_train = np.clip(Xall_train, lower, upper)
    Xall_test = np.clip(Xall_test, lower, upper)
    mean = Xall_train.mean(axis=0)
    std = Xall_train.std(axis=0)
    std[std < 1e-6] = 1.0
    Xall_train = (Xall_train - mean) / std
    Xall_test = (Xall_test - mean) / std

    y_raw = train[TARGET].to_numpy(dtype=np.float32)
    y_log = np.log1p(np.clip(y_raw, 0, None))
    train_scen = train[GROUP_COL].to_numpy()
    train_slot = train["slot"].astype(int).to_numpy()
    test_scen = test[GROUP_COL].to_numpy()
    test_slot = test["slot"].astype(int).to_numpy()
    slot_count = max(train_slot.max(), test_slot.max()) + 1

    test_ds = ScenarioDataset(Xall_test, y=None, scen_ids=test_scen, slots=test_slot, slot_count=slot_count)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 4, shuffle=False, num_workers=0)

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred_acc = np.zeros(len(test), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(make_folds(train, args.n_splits, args.seed), start=1):
        print(f"\n=== fold {fold} ===")
        Xtr, ytr_log = Xall_train[tr_idx], y_log[tr_idx]
        Xval, yval_log, yval_raw = Xall_train[val_idx], y_log[val_idx], y_raw[val_idx]
        scen_tr = train_scen[tr_idx]; slot_tr = train_slot[tr_idx]
        scen_val = train_scen[val_idx]; slot_val = train_slot[val_idx]

        tr_ds = ScenarioDataset(Xtr, ytr_log, scen_tr, slot_tr, slot_count)
        val_ds = ScenarioDataset(Xval, yval_log, scen_val, slot_val, slot_count)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, shuffle=False, num_workers=0)

        model = SeqGRU(n_features=Xtr.shape[1], hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout)
        loss_fn = nn.L1Loss(reduction="none")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(args.epochs):
            model.train()
            t0 = time.time()
            train_loss = 0.0
            n_total = 0
            for x, yt, mk in tr_loader:
                optimizer.zero_grad()
                pred = model(x)
                loss_per = loss_fn(pred, yt)
                loss = (loss_per * mk).sum() / mk.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * mk.sum().item()
                n_total += mk.sum().item()
            scheduler.step()

            model.eval()
            preds_log = np.zeros_like(val_ds.y)
            with torch.no_grad():
                for i, (x, _, _) in enumerate(val_loader):
                    p = model(x).cpu().numpy()
                    s = i * val_loader.batch_size
                    e = s + p.shape[0]
                    preds_log[s:e] = p
            row_pred = np.expm1(np.clip(preds_log[val_ds.row_to_scen, val_ds.row_to_slot], 0, None))
            val_mae = mean_absolute_error(yval_raw, row_pred)
            print(f"  ep {epoch+1}: train L1 {train_loss/n_total:.4f}, val MAE {val_mae:.4f} ({time.time()-t0:.0f}s)")

            if val_mae < best_val - 1e-4:
                best_val = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  early stopping")
                    break

        model.load_state_dict(best_state)
        model.eval()
        preds_log = np.zeros_like(val_ds.y)
        with torch.no_grad():
            for i, (x, _, _) in enumerate(val_loader):
                p = model(x).cpu().numpy()
                s = i * val_loader.batch_size
                e = s + p.shape[0]
                preds_log[s:e] = p
        row_pred = np.expm1(np.clip(preds_log[val_ds.row_to_scen, val_ds.row_to_slot], 0, None))
        oof[val_idx] = row_pred

        tst_preds_log = np.zeros((test_ds.n_scen, slot_count), dtype=np.float32)
        with torch.no_grad():
            for i, (x, _, _) in enumerate(test_loader):
                p = model(x).cpu().numpy()
                s = i * test_loader.batch_size
                e = s + p.shape[0]
                tst_preds_log[s:e] = p
        tst_row = np.expm1(np.clip(tst_preds_log[test_ds.row_to_scen, test_ds.row_to_slot], 0, None))
        test_pred_acc += tst_row / args.n_splits
        print(f"fold {fold} best val MAE: {best_val:.6f}")

    oof = np.clip(oof, 0, None)
    test_pred = np.clip(test_pred_acc, 0, None)
    overall_mae = mean_absolute_error(y_raw, oof)
    print(f"\nOOF MAE: {overall_mae:.6f}")

    sample[TARGET] = test_pred
    sample.to_csv(out_dir / "submission.csv", index=False)
    train_raw = pd.read_csv(data_dir / "train.csv")
    oof_df = train_raw[[ID_COL, GROUP_COL, "layout_id", TARGET]].copy()
    oof_df["pred"] = oof
    oof_df["abs_error"] = (oof_df[TARGET] - oof_df["pred"]).abs()
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    metadata = {
        "seed": args.seed, "n_splits": args.n_splits, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "hidden": args.hidden,
        "n_layers": args.n_layers, "dropout": args.dropout,
        "oof_mae": float(overall_mae), "n_features": len(feature_cols),
        "python": os.sys.version, "torch": torch.__version__,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
