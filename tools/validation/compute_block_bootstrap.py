#!/usr/bin/env python3
"""
Final Table 4 bootstrap. Single protocol (6h blocks, 100k iters), but the
unit of comparison varies by row:

  - Flare-PINN vs Strong-Form:  seed-averaged TSS (both have 3 seeds)
  - Flare-PINN vs DeFN:         seed-averaged TSS (3 vs 5 seeds)
  - Flare-PINN vs Baseline:     single-seed paper-lock checkpoints (1 vs 1)
  - Strong-Form vs Baseline:    single-seed paper-lock checkpoints (1 vs 1)

Rationale: only seed-average when both sides have multi-seed runs available;
otherwise use the publicly released paper-lock checkpoints for an
artifact-to-artifact comparison.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


WEAK_SEEDS = [
    PROJECT_ROOT / "outputs/checkpoints/weak_form/final/validation_results/checkpoint_step_0044000_test.npz",
    PROJECT_ROOT / "outputs/checkpoints/weak_form/seed1/validation_results/checkpoint_step_0044000_test.npz",
    PROJECT_ROOT / "outputs/checkpoints/weak_form/seed42/validation_results/checkpoint_step_0045000_test.npz",
]
STRONG_SEEDS = [
    PROJECT_ROOT / "outputs/checkpoints/strong_form/final/validation_results/checkpoint_step_0044000_test.npz",
    PROJECT_ROOT / "outputs/checkpoints/strong_form/seed1/validation_results/checkpoint_step_0044000_test.npz",
    PROJECT_ROOT / "outputs/checkpoints/strong_form/seed42/validation_results/checkpoint_step_0044000_test.npz",
]
DEFN_SEEDS = [
    PROJECT_ROOT / f"outputs/baselines/defn/seed{s}_test.npz"
    for s in (24, 10, 100, 42, 123)
]

# Single-seed paper-lock artifacts
WEAK_FINAL = WEAK_SEEDS[0]
STRONG_FINAL = STRONG_SEEDS[0]
BENCH = PROJECT_ROOT / "outputs/checkpoints/benchmark_classifier/validation_results/checkpoint_step_0040000_test.npz"


def build_block_indices(t0_series, block_size_hours):
    import pandas as pd
    t0_sorted = pd.to_datetime(t0_series).sort_values(ignore_index=True)
    blocks, current = [], [0]
    block_start = t0_sorted.iloc[0]
    for pos in range(1, len(t0_sorted)):
        diff_h = (t0_sorted.iloc[pos] - block_start).total_seconds() / 3600.0
        if diff_h < block_size_hours:
            current.append(pos)
        else:
            blocks.append(np.asarray(current, dtype=np.int64))
            current = [pos]
            block_start = t0_sorted.iloc[pos]
    if current:
        blocks.append(np.asarray(current, dtype=np.int64))
    return blocks


def build_moving_block_ends(t0_sorted_array, block_size_hours):
    """For each starting window index s, return the half-open end index e such that
    block [s, e) contains all windows j with t0[j] in [t0[s], t0[s]+block_size_hours)."""
    import pandas as pd
    t0_dt64 = pd.to_datetime(t0_sorted_array).to_numpy().astype("datetime64[s]")
    seconds = int(block_size_hours * 3600)
    t0_seconds = t0_dt64.astype("int64")
    target = t0_seconds + seconds
    end_idx = np.searchsorted(t0_seconds, target, side="left")
    return end_idx.astype(np.int64)


def block_counts(y_true, y_prob, thr, blocks):
    pred = (y_prob >= thr).astype(np.int8)
    counts = np.zeros((len(blocks), 4), dtype=np.int32)
    for bi, idx in enumerate(blocks):
        yb = y_true[idx]; pb = pred[idx]
        counts[bi, 0] = int(((pb == 1) & (yb == 1)).sum())
        counts[bi, 1] = int(((pb == 1) & (yb == 0)).sum())
        counts[bi, 2] = int(((pb == 0) & (yb == 1)).sum())
        counts[bi, 3] = int(((pb == 0) & (yb == 0)).sum())
    return counts


def moving_block_counts(y_true, y_prob, thr, end_idx):
    """For each starting position s, return (tp, fp, fn, tn) counts in [s, end_idx[s])."""
    N = len(y_true)
    pred = (y_prob >= thr).astype(np.int8)
    is_tp = ((pred == 1) & (y_true == 1)).astype(np.int32)
    is_fp = ((pred == 1) & (y_true == 0)).astype(np.int32)
    is_fn = ((pred == 0) & (y_true == 1)).astype(np.int32)
    is_tn = ((pred == 0) & (y_true == 0)).astype(np.int32)
    cum_tp = np.concatenate([[0], np.cumsum(is_tp)])
    cum_fp = np.concatenate([[0], np.cumsum(is_fp)])
    cum_fn = np.concatenate([[0], np.cumsum(is_fn)])
    cum_tn = np.concatenate([[0], np.cumsum(is_tn)])
    starts = np.arange(N)
    counts = np.stack([
        cum_tp[end_idx] - cum_tp[starts],
        cum_fp[end_idx] - cum_fp[starts],
        cum_fn[end_idx] - cum_fn[starts],
        cum_tn[end_idx] - cum_tn[starts],
    ], axis=-1).astype(np.int32)
    return counts


def tss_from_counts(counts):
    tp = counts[..., 0].astype(np.float64)
    fp = counts[..., 1].astype(np.float64)
    fn = counts[..., 2].astype(np.float64)
    tn = counts[..., 3].astype(np.float64)
    return tp / np.maximum(1.0, tp + fn) - fp / np.maximum(1.0, fp + tn)


def load_seed_blocks(npzs, h_idx, blocks_or_endidx, time_order, row_perm=None, moving=False):
    """Stack of per-seed block_counts → shape (n_seeds, n_blocks_or_N, 4),
    plus per-seed observed totals (n_seeds, 4) computed directly from raw data."""
    out = []
    obs_totals = []
    thrs = []
    for p in npzs:
        d = np.load(p)
        labels = d["labels"]
        probs = d["probs"]
        if row_perm is not None:
            labels = labels[row_perm]
            probs = probs[row_perm]
        labels = labels[time_order, h_idx]
        probs = probs[time_order, h_idx]
        thr = float(d["thresholds"][h_idx])
        pred = (probs >= thr).astype(np.int8)
        obs_totals.append(np.array([
            int(((pred == 1) & (labels == 1)).sum()),
            int(((pred == 1) & (labels == 0)).sum()),
            int(((pred == 0) & (labels == 1)).sum()),
            int(((pred == 0) & (labels == 0)).sum()),
        ], dtype=np.int32))
        if moving:
            out.append(moving_block_counts(labels, probs, thr, blocks_or_endidx))
        else:
            out.append(block_counts(labels, probs, thr, blocks_or_endidx))
        thrs.append(thr)
    return np.stack(out, axis=0), np.array(thrs), np.stack(obs_totals, axis=0)


def per_iter_tss_avg(model_blockcounts, sampled_blocks_idx):
    """Mean of per-seed TSS at each bootstrap iteration."""
    gathered = model_blockcounts[:, sampled_blocks_idx]
    summed = gathered.sum(axis=2)
    tss = tss_from_counts(summed)
    return tss.mean(axis=0)


def per_iter_tss_single(model_blockcounts, sampled_blocks_idx):
    """Single-seed TSS at each bootstrap iteration (model_blockcounts has shape (1, n_blocks, 4))."""
    gathered = model_blockcounts[0, sampled_blocks_idx]
    summed = gathered.sum(axis=1)
    return tss_from_counts(summed)


def observed_tss(obs_totals):
    """Mean of per-seed observed TSS — equals single-seed TSS when n_seeds=1.
    Operates on per-seed total confusion counts (n_seeds, 4)."""
    return float(tss_from_counts(obs_totals).mean())


def per_seed_tss_array(obs_totals):
    """Return shape (n_seeds,) array of per-seed TSS values."""
    return tss_from_counts(obs_totals).astype(np.float64)


def welch_t_test_one_sided(a, b):
    """Welch's t-test with one-sided alternative H1: mean(a) > mean(b).
    Returns (t_stat, df_welch, p_one_sided)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan"), float("nan")
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    se2 = s1 ** 2 / n1 + s2 ** 2 / n2
    if se2 <= 0:
        return float("nan"), float("nan"), float("nan")
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    df = se2 ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    from scipy.stats import t as t_dist
    p_one = float(t_dist.sf(t, df))  # P(T > t)
    return float(t), float(df), p_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-parquet", default=str(PROJECT_ROOT / "data/windows_test_15.parquet"))
    ap.add_argument("--block-size-hours", type=int, default=48,
                    help="Block size in hours. 48 matches the input lookback window length.")
    ap.add_argument("--moving", action="store_true", default=True,
                    help="Use Künsch (1989) moving block bootstrap. Default on.")
    ap.add_argument("--non-overlapping", action="store_true",
                    help="Use non-overlapping fixed blocks instead of moving blocks.")
    ap.add_argument("--n-boot", type=int, default=100000)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--fdr", action="store_true",
                    help="Apply Benjamini-Hochberg FDR correction across all 12 cells.")
    ap.add_argument("--fdr-alpha", type=float, default=0.05,
                    help="BH-FDR target FDR level (only used with --fdr).")
    ap.add_argument("--output", default=str(
        PROJECT_ROOT / "final_results/paper/metrics/bootstrap_block_ALL_FINAL.csv"
    ))
    args = ap.parse_args()

    import pandas as pd
    windows = pd.read_parquet(args.windows_parquet)
    time_order = np.argsort(pd.to_datetime(windows["t0"]).to_numpy())
    sorted_t0 = pd.to_datetime(windows["t0"]).iloc[time_order].reset_index(drop=True)

    use_moving = not args.non_overlapping
    if use_moving:
        t0_arr = sorted_t0.to_numpy()
        end_idx = build_moving_block_ends(t0_arr, args.block_size_hours)
        block_index = end_idx
        N = len(end_idx)
        block_lengths = end_idx - np.arange(N)
        mean_block_len = block_lengths.mean()
        n_resamples_per_iter = int(np.ceil(N / mean_block_len))
        print(f"Test windows: {len(windows)}  Moving blocks: N={N} starting positions, "
              f"mean block size = {mean_block_len:.1f} windows ({args.block_size_hours}h)")
        print(f"Resamples per iter: {n_resamples_per_iter}")
        sample_high = N
    else:
        nonov_blocks = build_block_indices(sorted_t0, args.block_size_hours)
        block_index = nonov_blocks
        N = len(nonov_blocks)
        n_resamples_per_iter = N
        sample_high = N
        print(f"Test windows: {len(windows)}  Non-overlapping blocks: {N} ({args.block_size_hours}h)")
    print(f"Bootstrap iterations: {args.n_boot:,}\n")

    defn_perm_path = PROJECT_ROOT / "outputs/baselines/defn/defn_to_windows_perm.npy"
    defn_perm = np.load(defn_perm_path) if defn_perm_path.exists() else None

    horizons = [(0, "6h"), (1, "12h"), (2, "24h")]
    rng = np.random.default_rng(42)
    results = []

    for h_idx, h_name in horizons:
        print(f"\n{'='*72}\n{h_name.upper()}\n{'='*72}")

        # Multi-seed sets
        weak_avg_bc, _, weak_avg_obs = load_seed_blocks(WEAK_SEEDS, h_idx, block_index, time_order, moving=use_moving)
        strong_avg_bc, _, strong_avg_obs = load_seed_blocks(STRONG_SEEDS, h_idx, block_index, time_order, moving=use_moving)
        defn_avg_bc, _, defn_avg_obs = load_seed_blocks(DEFN_SEEDS, h_idx, block_index, time_order, row_perm=defn_perm, moving=use_moving)

        # Single-seed paper-lock
        weak_one_bc, _, weak_one_obs = load_seed_blocks([WEAK_FINAL], h_idx, block_index, time_order, moving=use_moving)
        strong_one_bc, _, strong_one_obs = load_seed_blocks([STRONG_FINAL], h_idx, block_index, time_order, moving=use_moving)
        bench_bc, _, bench_obs = load_seed_blocks([BENCH], h_idx, block_index, time_order, moving=use_moving)

        comps = [
            ("Flare-PINN vs Baseline",   weak_one_bc, bench_bc, weak_one_obs, bench_obs, "single", "single"),
            ("Strong-Form vs Baseline",  strong_one_bc, bench_bc, strong_one_obs, bench_obs, "single", "single"),
            ("Flare-PINN vs Strong",     weak_avg_bc, strong_avg_bc, weak_avg_obs, strong_avg_obs, "avg", "avg"),
            ("Flare-PINN vs DeFN",       weak_avg_bc, defn_avg_bc, weak_avg_obs, defn_avg_obs, "avg", "avg"),
        ]

        for name, bc_a, bc_b, obs_a_arr, obs_b_arr, mode_a, mode_b in comps:
            obs_a = observed_tss(obs_a_arr)
            obs_b = observed_tss(obs_b_arr)
            delta_obs = obs_a - obs_b

            # Welch's t-test on per-seed TSS — only meaningful when both sides have >=2 seeds
            if mode_a == "avg" and mode_b == "avg":
                tss_a_seeds = per_seed_tss_array(obs_a_arr)
                tss_b_seeds = per_seed_tss_array(obs_b_arr)
                welch_t, welch_df, welch_p = welch_t_test_one_sided(tss_a_seeds, tss_b_seeds)
            else:
                welch_t, welch_df, welch_p = float("nan"), float("nan"), float("nan")

            deltas_parts = []
            for start in range(0, args.n_boot, args.chunk_size):
                batch = min(args.chunk_size, args.n_boot - start)
                samp = rng.integers(0, sample_high, size=(batch, n_resamples_per_iter))
                tss_a = (per_iter_tss_avg(bc_a, samp) if mode_a == "avg"
                         else per_iter_tss_single(bc_a, samp))
                tss_b = (per_iter_tss_avg(bc_b, samp) if mode_b == "avg"
                         else per_iter_tss_single(bc_b, samp))
                deltas_parts.append(tss_a - tss_b)
            deltas = np.concatenate(deltas_parts)

            ci_lo, ci_hi = np.percentile(deltas, [5, 95])
            p_one = float(np.mean(deltas <= 0))
            sig = p_one < 0.05

            n_a = bc_a.shape[0]
            n_b = bc_b.shape[0]
            print(f"  [{name}]  n_A={n_a} n_B={n_b}  TSS_A={obs_a:.4f}  TSS_B={obs_b:.4f}")
            extra = (f"  Welch t={welch_t:.2f} df={welch_df:.1f} p={welch_p:.4f}"
                     if not np.isnan(welch_p) else "")
            print(f"    ΔTSS={delta_obs:+.4f}  90% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                  f"boot_p={p_one:.4f}{extra}")

            results.append(dict(
                Comparison=name,
                Horizon=h_name,
                N_Seeds_A=str(n_a),
                N_Seeds_B=str(n_b),
                TSS_A=f"{obs_a:.6f}",
                TSS_B=f"{obs_b:.6f}",
                Delta_TSS=f"{delta_obs:+.6f}",
                CI_Lower=f"{ci_lo:+.6f}",
                CI_Upper=f"{ci_hi:+.6f}",
                Bootstrap_P_Value=f"{p_one:.6f}",
                Bootstrap_Significant="True" if sig else "False",
                Welch_t=f"{welch_t:.6f}" if not np.isnan(welch_t) else "",
                Welch_df=f"{welch_df:.6f}" if not np.isnan(welch_df) else "",
                Welch_P_Value=f"{welch_p:.6f}" if not np.isnan(welch_p) else "",
                Welch_Significant=("True" if (not np.isnan(welch_p) and welch_p < 0.05) else
                                   ("False" if not np.isnan(welch_p) else "")),
            ))

    if args.fdr:
        pvals = np.array([float(r["Bootstrap_P_Value"]) for r in results])
        m = len(pvals)
        order = np.argsort(pvals)
        sorted_p = pvals[order]
        raw_q = sorted_p * m / np.arange(1, m + 1)
        bh_q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
        bh_q = np.empty_like(bh_q_sorted)
        bh_q[order] = np.clip(bh_q_sorted, 0.0, 1.0)
        for i, r in enumerate(results):
            r["P_BH_q"] = f"{bh_q[i]:.6f}"
            r["Significant_BH"] = "True" if bh_q[i] < args.fdr_alpha else "False"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_path}")

    print(f"\n{'='*100}\nFINAL TABLE — hybrid headline test "
          f"(Welch's t for multi-seed rows, block bootstrap for single-seed rows)"
          f"\n{'='*100}")
    print(f"{'Comparison':<28s} {'Hor':>4s} {'A/B':>5s} {'ΔTSS':>8s} "
          f"{'90% CI':>22s} {'boot_p':>7s} {'Welch_p':>8s} {'sig':>4s}")
    for r in results:
        ab = f"{r['N_Seeds_A']}/{r['N_Seeds_B']}"
        is_multi = int(r["N_Seeds_A"]) > 1 and int(r["N_Seeds_B"]) > 1
        if is_multi:
            head_p = float(r["Welch_P_Value"])
            head_label = "W"
        else:
            head_p = float(r["Bootstrap_P_Value"])
            head_label = "B"
        sig = "*" if head_p < 0.05 else ""
        welch_disp = f"{float(r['Welch_P_Value']):.4f}" if r["Welch_P_Value"] else "    -   "
        print(f"{r['Comparison']:<28s} {r['Horizon']:>4s} {ab:>5s} "
              f"{float(r['Delta_TSS']):>+.3f}    "
              f"[{float(r['CI_Lower']):+.3f}, {float(r['CI_Upper']):+.3f}]  "
              f"{float(r['Bootstrap_P_Value']):.4f}  {welch_disp:>8s}  {sig:>1s}{head_label}")


if __name__ == "__main__":
    main()
