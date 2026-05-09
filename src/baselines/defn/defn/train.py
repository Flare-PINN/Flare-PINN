#!/usr/bin/env python3
"""
DeFN (Deep Flare Net) — faithful reimplementation, fair evaluation.

Architecture: 8-layer residual DNN with BN + skip connections
  (matches https://github.com/komeisugiura/defn18 line-for-line)

Training differences from original (and why):
  - Original selects checkpoints by TEST TSS (data snooping).
    We report VALIDATION D2C thresholding for the primary metrics — same as our PINN protocol.
  - Original uses 16k mini-batch iterations with no early stopping; we match that
    (``--max-iters`` default 16000).
  - Architecture, loss, optimizer, and class weights match the released TensorFlow code.

Usage:
    python -m src.baselines.defn.train
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.baselines.defn.defn.config import DeFNConfig, DeFNRunResult, DeFNSummaryRow
from src.baselines.defn.defn.model import DeFN
from src.models.eval.metrics import (
    confusion_at_threshold,
    distance_to_corner,
    tss_at_threshold,
)
from tools.defn.audit_defn_state import build_report, readiness_blockers
from tools.defn.build_defn_features import ALL_FEATURE_COLS


class SplitData:
    def __init__(self, X: np.ndarray, labels: dict[str, np.ndarray]):
        self.X = X
        self.labels = labels


def load_and_split(cfg: DeFNConfig) -> tuple[SplitData, SplitData, SplitData]:
    features = pd.read_parquet(project_root / cfg.data.features_path)
    train_val_windows = pd.read_parquet(project_root / cfg.data.train_val_windows)
    test_windows = pd.read_parquet(project_root / cfg.data.test_windows)

    train_val_uids = set(train_val_windows["window_uid"])
    test_uids = set(test_windows["window_uid"])

    train_val_feat = features[features["window_uid"].isin(train_val_uids)].copy()
    test_feat = features[features["window_uid"].isin(test_uids)].copy()

    # Override labels from the canonical windows files (scalar_features.parquet
    # has stale labels that differ from the PINN's ground truth).
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

    print(f"  Train: {len(train_feat)}, Val: {len(val_feat)}, Test: {len(test_feat)}")

    available_cols = [c for c in ALL_FEATURE_COLS if c in features.columns]
    print(f"  Using {len(available_cols)} of {len(ALL_FEATURE_COLS)} features")

    for h in cfg.train.horizons:
        lc = f"y_geq_M_{h}"
        trn_pos = train_feat[lc].sum()
        val_pos = val_feat[lc].sum()
        tst_pos = test_feat[lc].sum()
        print(f"  Labels {h}: train={trn_pos}, val={val_pos}, test={tst_pos}")

    def extract(df: pd.DataFrame) -> SplitData:
        X = df[available_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        labels = {}
        for h in cfg.train.horizons:
            labels[h] = df[f"y_geq_M_{h}"].values.astype(np.int64)
        return SplitData(X, labels)

    return extract(train_feat), extract(val_feat), extract(test_feat)


def _argmax_tss(y_true: np.ndarray, probs_2d: np.ndarray) -> float:
    """Original DeFN TSS: argmax thresholding (class with higher prob wins)."""
    y_pred = (probs_2d[:, 1] > probs_2d[:, 0]).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return tpr - fpr


def _weighted_ce_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Matches original TF: -mean(w * y_onehot * log(clip(softmax, 1e-10)))"""
    probs = torch.clamp(torch.softmax(logits, dim=1), min=1e-10, max=1.0)
    y_onehot = torch.zeros_like(probs)
    y_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
    return -torch.mean(class_weights * y_onehot * torch.log(probs))


def _ensure_exact_defn_ready(cfg: DeFNConfig) -> None:
    features_path = project_root / cfg.data.features_path
    defn_dir = features_path.parent
    report = build_report(defn_dir=defn_dir, windows_path=features_path)
    blockers = readiness_blockers(report)
    active = blockers["aia"] + blockers["training"]
    if not active:
        return
    lines = "\n".join(f"  - {msg}" for msg in active)
    raise click.ClickException(
        "Exact DeFN replication inputs are not ready.\n"
        f"{lines}\n"
        f"Run: python tools/defn/audit_defn_state.py --strict --defn-dir {defn_dir.relative_to(project_root)}"
    )


def train_single(
    model_cfg: DeFNConfig,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int, horizon: str,
    max_iters: int = 16000,
) -> DeFNRunResult:
    """
    Faithful DeFN: 16k mini-batch iterations, no early stopping,
    final model evaluated with argmax — exactly as original TF code.
    """
    cfg = model_cfg.train
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    scaler = StandardScaler()
    X_tr = torch.from_numpy(scaler.fit_transform(X_train).astype(np.float32))
    X_te = torch.from_numpy(scaler.transform(X_test).astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train)

    model = DeFN(X_tr.shape[1], model_cfg.model).to(device)
    class_weights = torch.tensor([1.0, cfg.pos_class_weight], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999))

    # Original TF code: shuffle once, create fixed batches, cycle through
    # for 16000 mini-batch iterations.
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
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = _weighted_ce_loss(model(bx), by, class_weights)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 2000 == 0:
                print(f"    iter {step}/{max_iters} loss={loss.item():.4f}")

    # Evaluate FINAL model (no model selection — original takes last checkpoint)
    model.eval()
    X_v = torch.from_numpy(scaler.transform(X_val).astype(np.float32)).to(device)
    with torch.no_grad():
        val_probs = torch.softmax(model(X_v), dim=1)[:, 1].cpu().numpy()
        test_probs_2d = torch.softmax(model(X_te), dim=1).cpu().numpy()
        test_probs = test_probs_2d[:, 1]

    # Argmax eval (original protocol)
    argmax_tss = _argmax_tss(y_test, test_probs_2d)

    # D2C eval (same protocol as PINN — threshold from val, apply to test)
    d2c_thr = distance_to_corner(y_val, val_probs)
    d2c_tss = tss_at_threshold(y_test, test_probs, d2c_thr)
    tp, fp, fn, tn = confusion_at_threshold(y_test, test_probs, d2c_thr)
    pod = tp / max(1, tp + fn)
    far = fp / max(1, fp + tp)
    csi = tp / max(1, tp + fn + fp)

    print(f"    argmax_TSS={argmax_tss:.4f}  d2c_TSS={d2c_tss:.4f} (thr={d2c_thr:.4f})")

    return DeFNRunResult(
        seed=seed,
        horizon=horizon,
        iters=max_iters,
        argmax_tss=argmax_tss,
        threshold=d2c_thr,
        test_tss=d2c_tss,
        test_pod=pod,
        test_far=far,
        test_csi=csi,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


@click.command()
@click.option("--seeds", default="24,10,100,42,123",
              help="Comma-separated training seeds. Default reproduces the published 5-seed DeFN ensemble.")
@click.option("--max-iters", default=16000, show_default=True)
@click.option("--output-dir", default="final_results/defn")
@click.option(
    "--allow-incomplete-features",
    is_flag=True,
    help="Train even if AIA 1600 or other exact-replication checks are incomplete.",
)
def main(
    seeds: str,
    max_iters: int,
    output_dir: str,
    allow_incomplete_features: bool,
):
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s.strip()) for s in seeds.split(",")]

    cfg = DeFNConfig()
    cfg.train.seeds = seed_list
    if not allow_incomplete_features:
        _ensure_exact_defn_ready(cfg)

    print("=" * 80)
    print("DeFN (Deep Flare Net) — FAITHFUL REIMPL, FAIR EVALUATION")
    print("=" * 80)
    print(f"  Architecture: 8-layer residual DNN (matching original TF)")
    print(f"  Loss: weighted CE [1, {cfg.train.pos_class_weight}] (original)")
    print(f"  Optimizer: Adam lr={cfg.train.lr} (original)")
    print(f"  Batch: {cfg.train.batch_size} (original)")
    print(f"  Iterations: {max_iters} (original: 16000)")
    print(f"  No early stopping, final model evaluated with argmax")
    print(f"  Seeds: {seed_list}")

    print(f"\n{'=' * 80}")
    print("LOADING DATA")
    print("=" * 80)
    train_data, val_data, test_data = load_and_split(cfg)

    all_results: list[DeFNRunResult] = []
    # Weak-form paper reporting now uses the 3-seed paper-lock mean.
    pinn_tss = {"6h": 0.8170, "12h": 0.8106, "24h": 0.7898}

    for horizon in cfg.train.horizons:
        print(f"\n{'=' * 80}")
        print(f"HORIZON: {horizon}")
        print("=" * 80)

        y_train = train_data.labels[horizon]
        y_val = val_data.labels[horizon]
        y_test = test_data.labels[horizon]

        n_pos = y_train.sum()
        print(f"  Train pos: {n_pos}/{len(y_train)} ({100*n_pos/len(y_train):.2f}%)")
        print(f"  Val pos:   {y_val.sum()}/{len(y_val)}")
        print(f"  Test pos:  {y_test.sum()}/{len(y_test)}")

        for seed in seed_list:
            print(f"\n  --- Seed {seed} ---")
            r = train_single(
                cfg, train_data.X, y_train,
                val_data.X, y_val, test_data.X, y_test,
                seed=seed, horizon=horizon,
                max_iters=max_iters,
            )
            all_results.append(r)
            print(f"    => Test TSS: {r.test_tss:.4f}, "
                  f"POD: {r.test_pod:.4f}, FAR: {r.test_far:.4f}, "
                  f"thr: {r.threshold:.4f}")

        h_tss = [r.test_tss for r in all_results if r.horizon == horizon]
        print(f"\n  {horizon} SUMMARY: TSS = {np.mean(h_tss):.4f} ± {np.std(h_tss):.4f}"
              f"  (PINN: {pinn_tss[horizon]:.4f})")

    # Save per-seed results and per-horizon summary (summary derived from results only)
    df = pd.DataFrame([r.model_dump() for r in all_results])
    df.to_csv(out_dir / "defn_results.csv", index=False)

    summary_rows: list[DeFNSummaryRow] = []
    for h in cfg.train.horizons:
        hd = df[df["horizon"] == h]
        summary_rows.append(
            DeFNSummaryRow(
                horizon=h,
                n_seeds=len(hd),
                mean_test_tss=float(hd["test_tss"].mean()),
                std_test_tss=float(hd["test_tss"].std()),
                mean_argmax_tss=float(hd["argmax_tss"].mean()),
                std_argmax_tss=float(hd["argmax_tss"].std()),
                mean_pod=float(hd["test_pod"].mean()),
                std_pod=float(hd["test_pod"].std()),
                mean_far=float(hd["test_far"].mean()),
                std_far=float(hd["test_far"].std()),
                mean_csi=float(hd["test_csi"].mean()),
                std_csi=float(hd["test_csi"].std()),
            )
        )
    pd.DataFrame([s.model_dump() for s in summary_rows]).to_csv(
        out_dir / "defn_summary.csv", index=False,
    )

    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON (D2C TSS — same thresholding protocol as Flare-PINN)")
    print("=" * 80)
    print(f"{'Horizon':<8} {'DeFN (D2C)':<18} {'PINN (D2C)':>12} {'Δ':>8}")
    print("-" * 50)
    for h in cfg.train.horizons:
        hd = df[df["horizon"] == h]
        m, s = hd["test_tss"].mean(), hd["test_tss"].std()
        p = pinn_tss[h]
        delta = p - m
        print(f"{h:<8} {m:.4f}±{s:.4f}   {p:>10.4f} {delta:>+8.4f}")

    print(f"\n{'=' * 80}")
    print("DeFN original protocol (argmax on class logits — test set only)")
    print("=" * 80)
    for h in cfg.train.horizons:
        hd = df[df["horizon"] == h]
        m, s = hd["argmax_tss"].mean(), hd["argmax_tss"].std()
        print(f"  {h}: argmax TSS = {m:.4f} ± {s:.4f}")

    print(f"\nSaved: {out_dir}/defn_results.csv")
    print(f"Saved: {out_dir}/defn_summary.csv")


if __name__ == "__main__":
    main()
