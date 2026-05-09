#!/usr/bin/env python3
"""
DeFN training with per-seed prediction NPZs saved.

Reproduces the architecture and training loop from train.py while additionally
writing per-seed validation and test predictions for downstream multi-seed
bootstrap and Welch's t-test analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.defn.defn.config import DeFNConfig
from src.baselines.defn.defn.model import DeFN
from src.models.eval.metrics import (
    confusion_at_threshold,
    distance_to_corner,
    tss_at_threshold,
)
ALL_FEATURE_COLS = [
    'USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'MEANJZH',
    'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'AREA_ACR', 'TOTBSQ', 'TOTFX', 'TOTFY', 'TOTFZ',
    'Bmax', 'Bmin', 'Bave', 'MaxdxBz', 'MaxdyBz', 'TotNL', 'NumNL', 'MaxNL',
    'CHArea', 'CHAll', 'CHMax', 'Xflux1h', 'Xflux4h', 'Xmax1d', 'Xhis', 'Mhis',
    'Xhis1d', 'Mhis1d', 'dt24_SAVNCPP', 'dt24_TotNL', 'dt24_TOTBSQ', 'dt24_TOTFY',
    'dt24_TOTFX', 'dt24_TOTFZ', 'dt24_USFLUX', 'dt24_AREA_ACR', 'dt24_ABSNJZH',
    'dt24_TOTUSJZ', 'dt24_Bmax', 'dt24_CHArea', 'dt24_MaxGraB', 'dt24_MaxdzBy',
    'dt24_TOTUSJH', 'dt24_NumNL', 'dt24_MaxdxBz', 'dt24_MEANJZH', 'dt24_MaxNL',
    'dt24_CHAll', 'dt24_CHMax', 'dt24_MEANGBZ', 'dt24_MEANGBH', 'dt24_MEANGBT',
    'dt24_MEANGAM', 'dt24_MEANJZD', 'dt12_AREA_ACR', 'dt12_Bmax', 'dt12_USFLUX',
    'dt02_AREA_ACR', 'dt02_Bmax', 'CRArea', 'CRAll', 'CRMax', 'CRArea_1h',
    'CRAll_1h', 'CRMax_1h', 'CRArea_2h', 'CRAll_2h', 'CRMax_2h',
    'Xflux_1hbef', 'Xflux_2hbef', 'dt24_CRArea', 'dt24_CRAll', 'dt24_CRMax',
]
assert len(ALL_FEATURE_COLS) == 79, f"expected 79 cols, got {len(ALL_FEATURE_COLS)}"


def load_and_split(cfg: DeFNConfig):
    features = pd.read_parquet(REPO_ROOT / cfg.data.features_path)
    train_val_windows = pd.read_parquet(REPO_ROOT / cfg.data.train_val_windows)
    test_windows = pd.read_parquet(REPO_ROOT / cfg.data.test_windows)

    train_val_uids = set(train_val_windows["window_uid"])
    test_uids = set(test_windows["window_uid"])

    train_val_feat = features[features["window_uid"].isin(train_val_uids)].copy()
    test_feat = features[features["window_uid"].isin(test_uids)].copy()

    label_cols = [f"y_geq_M_{h}" for h in cfg.train.horizons]
    for lc in label_cols:
        train_val_feat.drop(columns=[lc], inplace=True, errors="ignore")
        test_feat.drop(columns=[lc], inplace=True, errors="ignore")
    train_val_feat = train_val_feat.merge(
        train_val_windows[["window_uid"] + label_cols], on="window_uid", how="left",
    )
    test_feat = test_feat.merge(
        test_windows[["window_uid"] + label_cols], on="window_uid", how="left",
    )
    train_val_feat = train_val_feat.sort_values("t0").reset_index(drop=True)
    test_feat = test_feat.sort_values("t0").reset_index(drop=True)

    n_val = int(len(train_val_feat) * cfg.data.val_fraction)
    train_feat = train_val_feat.iloc[:-n_val]
    val_feat = train_val_feat.iloc[-n_val:]

    available_cols = [c for c in ALL_FEATURE_COLS if c in features.columns]
    print(f"  Using {len(available_cols)}/{len(ALL_FEATURE_COLS)} features"
          f"  | train={len(train_feat)} val={len(val_feat)} test={len(test_feat)}")

    def extract(df):
        X = df[available_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        labels = {h: df[f"y_geq_M_{h}"].values.astype(np.int64) for h in cfg.train.horizons}
        return X, labels

    return extract(train_feat), extract(val_feat), extract(test_feat)


def _weighted_ce_loss(logits, targets, class_weights):
    probs = torch.clamp(torch.softmax(logits, dim=1), min=1e-10, max=1.0)
    y_onehot = torch.zeros_like(probs)
    y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return -torch.mean(class_weights * y_onehot * torch.log(probs))


def train_seed_horizon(
    model_cfg: DeFNConfig,
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed: int, max_iters: int = 16000,
):
    cfg = model_cfg.train
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    scaler = StandardScaler()
    X_tr = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32))
    X_v = torch.from_numpy(scaler.transform(X_val).astype(np.float32))
    X_te = torch.from_numpy(scaler.transform(X_test).astype(np.float32))
    y_tr = torch.from_numpy(y_train)

    model = DeFN(X_tr.shape[1], model_cfg.model).to(device)
    class_weights = torch.tensor([1.0, cfg.pos_class_weight], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    n = len(X_tr)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    X_shuf = X_tr[perm]
    y_shuf = y_tr[perm]
    bs = cfg.batch_size
    n_batches = (n + bs - 1) // bs
    batches_X = [X_shuf[i * bs:(i + 1) * bs] for i in range(n_batches)]
    batches_y = [y_shuf[i * bs:(i + 1) * bs] for i in range(n_batches)]

    model.train()
    step = 0
    while step < max_iters:
        for bx, by in zip(batches_X, batches_y):
            if step >= max_iters:
                break
            optimizer.zero_grad()
            loss = _weighted_ce_loss(model(bx), by, class_weights)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 4000 == 0:
                print(f"      iter {step}/{max_iters} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(model(X_v), dim=1)[:, 1].cpu().numpy()
        test_probs = torch.softmax(model(X_te), dim=1)[:, 1].cpu().numpy()
    d2c_thr = float(distance_to_corner(y_val, val_probs))
    test_tss = tss_at_threshold(y_test, test_probs, d2c_thr)
    return val_probs, test_probs, d2c_thr, test_tss


@click.command()
@click.option("--seeds", default="24,10,100,42,123",
              help="Comma-separated training seeds. Default reproduces the published 5-seed DeFN ensemble.")
@click.option("--max-iters", default=16000, show_default=True)
@click.option("--output-dir", default=str(REPO_ROOT / "outputs/baselines/defn"))
def main(seeds, max_iters, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in seeds.split(",")]

    cfg = DeFNConfig()
    train_data, val_data, test_data = load_and_split(cfg)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    horizons = ["6h", "12h", "24h"]
    horizons_int = np.array([6, 12, 24])

    n_test = len(X_test)
    n_val = len(X_val)
    test_labels = np.stack([y_test[h] for h in horizons], axis=1).astype(np.float32)
    val_labels = np.stack([y_val[h] for h in horizons], axis=1).astype(np.float32)

    for seed in seed_list:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        seed_test = np.zeros((n_test, 3), dtype=np.float32)
        seed_val = np.zeros((n_val, 3), dtype=np.float32)
        seed_thr = np.zeros(3, dtype=np.float32)
        for hi, h in enumerate(horizons):
            print(f"\n  Horizon {h}")
            vp, tp_, thr, tss = train_seed_horizon(
                cfg, X_train, y_train[h], X_val, y_val[h], X_test, y_test[h],
                seed=seed, max_iters=max_iters,
            )
            print(f"    seed={seed} {h}  d2c_TSS={tss:.4f}  thr={thr:.4f}")
            seed_val[:, hi] = vp
            seed_test[:, hi] = tp_
            seed_thr[hi] = thr

        np.savez(out_dir / f"seed{seed}_test.npz",
                 probs=seed_test, labels=test_labels,
                 thresholds=seed_thr, horizons=horizons_int)
        np.savez(out_dir / f"seed{seed}_val.npz",
                 probs=seed_val, labels=val_labels, horizons=horizons_int)
        print(f"  Saved seed{seed}_test.npz + seed{seed}_val.npz")

    print(f"\nAll seeds saved to {out_dir}")


if __name__ == "__main__":
    main()
