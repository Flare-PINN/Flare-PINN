#!/usr/bin/env python3
"""
Audit DeFN replication readiness from local parquet/progress state.

This script answers two questions:
  1. Is AIA 1600 Angstrom fully refetched and merged into ``aia_features.parquet``?
  2. Is ``defn_features.parquet`` ready for exact 79-feature DeFN training?

It prints a compact report and can fail with a non-zero exit code in strict mode.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from tools.defn.build_defn_features import (
    AIA_131_1H,
    AIA_131_2H,
    AIA_131_T0,
    AIA_1600_FEATURES,
    ALL_FEATURE_COLS,
    DT24_131,
)

CR_COLS = ["CR131Area", "CR131All", "CR131Max"]
DT24_AIA_1600 = [f"dt24_{col}" for col in AIA_1600_FEATURES]
AIA_131_FEATURES = AIA_131_T0 + AIA_131_1H + AIA_131_2H + DT24_131


def _to_timestamp_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _compute_required_timestamps(windows_df: pd.DataFrame) -> pd.DataFrame:
    records: list[tuple[int, int, pd.Timestamp]] = []
    for _, row in windows_df.iterrows():
        harpnum = int(row["harpnum"])
        t0 = _to_timestamp_utc(row["t0"])
        records.append((harpnum, 1600, t0))
        records.append((harpnum, 1600, t0 - pd.Timedelta(hours=24)))
        records.append((harpnum, 131, t0))
        records.append((harpnum, 131, t0 - pd.Timedelta(hours=1)))
        records.append((harpnum, 131, t0 - pd.Timedelta(hours=2)))
        records.append((harpnum, 131, t0 - pd.Timedelta(hours=24)))
    out = pd.DataFrame(records, columns=["harpnum", "wavelength", "t_need"])
    return out.drop_duplicates().reset_index(drop=True)


def _group_into_batches(required: pd.DataFrame, max_span_hours: float) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    if required.empty:
        return batches
    for (harpnum, wavelength), grp in required.groupby(["harpnum", "wavelength"]):
        times = sorted(pd.to_datetime(grp["t_need"], utc=True).tolist())
        current = [times[0]]
        for ts in times[1:]:
            span_h = (ts - current[0]).total_seconds() / 3600.0
            if span_h <= max_span_hours:
                current.append(ts)
                continue
            batches.append(
                {
                    "harpnum": int(harpnum),
                    "wavelength": int(wavelength),
                    "t_start": min(current) - pd.Timedelta(minutes=15),
                    "t_end": max(current) + pd.Timedelta(minutes=15),
                    "timestamps": current,
                }
            )
            current = [ts]
        batches.append(
            {
                "harpnum": int(harpnum),
                "wavelength": int(wavelength),
                "t_start": min(current) - pd.Timedelta(minutes=15),
                "t_end": max(current) + pd.Timedelta(minutes=15),
                "timestamps": current,
            }
        )
    return batches


def _batch_key(batch: dict[str, Any]) -> str:
    t_start = pd.Timestamp(batch["t_start"]).strftime("%Y%m%d%H%M")
    return f"{batch['harpnum']}_{batch['wavelength']}_{t_start}"


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def _column_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for col in cols:
        if col not in df.columns:
            stats[col] = {"present": 0, "nonnull": 0, "nonzero": 0}
            continue
        series = _numeric_series(df, col)
        stats[col] = {
            "present": 1,
            "nonnull": int(series.notna().sum()),
            "nonzero": int((series.fillna(0.0) != 0.0).sum()),
        }
    return stats


def _group_nonzero_windows(df: pd.DataFrame, cols: list[str]) -> int:
    present = [col for col in cols if col in df.columns]
    if not present or df.empty:
        return 0
    numeric = df[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return int((numeric != 0.0).any(axis=1).sum())


def _load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return set(data.get("done", []))


def _compute_expected_batches(
    windows_path: Path,
    num_shards: int,
    max_batch_span_hours: float,
) -> dict[str, Any] | None:
    if not windows_path.exists():
        return None
    win_df = pd.read_parquet(windows_path, columns=["harpnum", "t0"])
    win_df["t0"] = pd.to_datetime(win_df["t0"], utc=True)
    with redirect_stdout(io.StringIO()):
        required = _compute_required_timestamps(win_df)
        batches = _group_into_batches(required, max_span_hours=max_batch_span_hours)
    out: dict[str, Any] = {
        "total": len(batches),
        "by_wavelength": {
            "131": int(sum(int(b["wavelength"]) == 131 for b in batches)),
            "1600": int(sum(int(b["wavelength"]) == 1600 for b in batches)),
        },
        "per_shard": {},
    }
    if num_shards > 1:
        per_shard: dict[int, dict[str, int]] = {
            shard: {"total": 0, "131": 0, "1600": 0} for shard in range(num_shards)
        }
        for batch in batches:
            shard = int(batch["harpnum"]) % num_shards
            per_shard[shard]["total"] += 1
            per_shard[shard][str(batch["wavelength"])] += 1
        out["per_shard"] = per_shard
    return out


def build_report(
    defn_dir: Path,
    windows_path: Path,
    num_shards: int = 12,
    max_batch_span_hours: float = 48.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "defn_dir": str(defn_dir),
        "windows_path": str(windows_path),
        "progress": {},
        "aia_shards": {},
        "aia_merged": {},
        "defn_features": {},
    }

    expected = _compute_expected_batches(windows_path, num_shards, max_batch_span_hours)
    progress_paths = [defn_dir / f"aia_targeted_progress_s{i}.json" for i in range(num_shards)]
    progress_items: list[dict[str, Any]] = []
    done_total = 0
    done_131 = 0
    done_1600 = 0
    for shard_id, path in enumerate(progress_paths):
        done = _load_progress(path)
        shard_131 = sum("_131_" in key for key in done)
        shard_1600 = sum("_1600_" in key for key in done)
        progress_items.append(
            {
                "path": path.name,
                "exists": path.exists(),
                "shard_id": shard_id,
                "done_total": len(done),
                "done_131": int(shard_131),
                "done_1600": int(shard_1600),
            }
        )
        done_total += len(done)
        done_131 += int(shard_131)
        done_1600 += int(shard_1600)
    report["progress"] = {
        "files": progress_items,
        "done_total": done_total,
        "done_131": done_131,
        "done_1600": done_1600,
        "expected": expected,
    }
    if expected is not None:
        report["progress"]["remaining_131"] = max(0, expected["by_wavelength"]["131"] - done_131)
        report["progress"]["remaining_1600"] = max(0, expected["by_wavelength"]["1600"] - done_1600)
        report["progress"]["remaining_total"] = max(0, expected["total"] - done_total)

    shard_paths = [defn_dir / f"aia_features_targeted_s{i}.parquet" for i in range(num_shards)]
    existing_shards = [path for path in shard_paths if path.exists()]
    shard_summaries: list[dict[str, Any]] = []
    shard_rows = 0
    shard_harps: set[int] = set()
    shard_ch_nonnull = 0
    shard_cr_nonnull = 0
    for path in existing_shards:
        df = pd.read_parquet(path)
        shard_rows += len(df)
        if "harpnum" in df.columns:
            shard_harps.update(pd.to_numeric(df["harpnum"], errors="coerce").dropna().astype(int).tolist())
        ch_stats = _column_stats(df, AIA_1600_FEATURES)
        cr_stats = _column_stats(df, CR_COLS)
        shard_ch_nonnull += sum(v["nonnull"] for v in ch_stats.values())
        shard_cr_nonnull += sum(v["nonnull"] for v in cr_stats.values())
        shard_summaries.append(
            {
                "path": path.name,
                "rows": len(df),
                "ch_nonnull": {col: ch_stats[col]["nonnull"] for col in AIA_1600_FEATURES},
                "cr_nonnull": {col: cr_stats[col]["nonnull"] for col in CR_COLS},
            }
        )
    report["aia_shards"] = {
        "found": len(existing_shards),
        "expected": num_shards,
        "rows": shard_rows,
        "harps": len(shard_harps),
        "ch_nonnull_total": shard_ch_nonnull,
        "cr_nonnull_total": shard_cr_nonnull,
        "files": shard_summaries,
    }

    merged_path = defn_dir / "aia_features.parquet"
    if merged_path.exists():
        merged = pd.read_parquet(merged_path)
        merged_stats = _column_stats(merged, AIA_1600_FEATURES + CR_COLS)
        report["aia_merged"] = {
            "exists": True,
            "path": merged_path.name,
            "rows": len(merged),
            "cols": len(merged.columns),
            "harps": int(merged["harpnum"].nunique()) if "harpnum" in merged.columns else 0,
            "column_stats": merged_stats,
        }
    else:
        report["aia_merged"] = {"exists": False, "path": merged_path.name}

    features_path = defn_dir / "defn_features.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        feature_stats = _column_stats(
            features,
            AIA_1600_FEATURES + DT24_AIA_1600 + AIA_131_FEATURES,
        )
        missing_cols = [col for col in ALL_FEATURE_COLS if col not in features.columns]
        report["defn_features"] = {
            "exists": True,
            "path": features_path.name,
            "rows": len(features),
            "cols": len(features.columns),
            "feature_cols_present": len([col for col in ALL_FEATURE_COLS if col in features.columns]),
            "missing_feature_cols": missing_cols,
            "column_stats": feature_stats,
            "group_nonzero_windows": {
                "aia_1600": _group_nonzero_windows(features, AIA_1600_FEATURES),
                "dt24_aia_1600": _group_nonzero_windows(features, DT24_AIA_1600),
                "aia_131": _group_nonzero_windows(features, AIA_131_FEATURES),
            },
        }
    else:
        report["defn_features"] = {"exists": False, "path": features_path.name}

    return report


def readiness_blockers(report: dict[str, Any]) -> dict[str, list[str]]:
    aia: list[str] = []
    training: list[str] = []

    progress = report.get("progress", {})
    expected = progress.get("expected")
    if expected is not None and progress.get("remaining_1600", 0) > 0:
        aia.append(
            "1600 A JSOC batches are incomplete: "
            f"{progress['done_1600']}/{expected['by_wavelength']['1600']} done"
        )

    shards = report.get("aia_shards", {})
    if shards.get("found", 0) not in (0, shards.get("expected", 0)):
        aia.append(
            f"found {shards.get('found', 0)}/{shards.get('expected', 0)} AIA shard parquets"
        )

    merged = report.get("aia_merged", {})
    if not merged.get("exists"):
        aia.append("merged AIA parquet is missing")
    else:
        merged_stats = merged.get("column_stats", {})
        missing_ch = [
            col for col in AIA_1600_FEATURES if merged_stats.get(col, {}).get("nonnull", 0) == 0
        ]
        if missing_ch:
            aia.append(
                "merged AIA parquet has no 1600 coverage for "
                + ", ".join(missing_ch)
            )

    features = report.get("defn_features", {})
    if not features.get("exists"):
        training.append("defn_features.parquet is missing")
    else:
        missing_feature_cols = features.get("missing_feature_cols", [])
        if missing_feature_cols:
            training.append(
                f"missing {len(missing_feature_cols)} of 79 DeFN feature columns"
            )
        group_nonzero = features.get("group_nonzero_windows", {})
        if group_nonzero.get("aia_1600", 0) == 0:
            training.append("AIA 1600 DeFN features are all zero")
        if group_nonzero.get("dt24_aia_1600", 0) == 0:
            training.append("dt24 AIA 1600 DeFN features are all zero")
        if group_nonzero.get("aia_131", 0) == 0:
            training.append("AIA 131 DeFN features are all zero")

    if aia and "merged AIA parquet is missing" not in aia and not training:
        training.append("training is blocked because AIA 1600 inputs are not ready")

    return {"aia": aia, "training": training}


def print_report(report: dict[str, Any], blockers: dict[str, list[str]]) -> None:
    print("== DeFN audit ==")

    progress = report["progress"]
    print("[progress]")
    if progress.get("expected") is not None:
        expected = progress["expected"]
        print(
            "  expected batches: "
            f"{expected['total']} total "
            f"(131={expected['by_wavelength']['131']}, 1600={expected['by_wavelength']['1600']})"
        )
        print(
            "  done batches: "
            f"{progress['done_total']} total "
            f"(131={progress['done_131']}, 1600={progress['done_1600']})"
        )
        print(
            "  remaining: "
            f"{progress['remaining_total']} total "
            f"(131={progress['remaining_131']}, 1600={progress['remaining_1600']})"
        )
    else:
        print(
            "  done batches: "
            f"{progress['done_total']} total "
            f"(131={progress['done_131']}, 1600={progress['done_1600']})"
        )
        print("  expected batches: skipped (windows parquet not available)")

    shards = report["aia_shards"]
    print("[aia shards]")
    print(
        f"  shard parquets: {shards['found']}/{shards['expected']} "
        f"rows={shards['rows']} harps={shards['harps']}"
    )
    print(
        f"  non-null totals: CH*={shards['ch_nonnull_total']} "
        f"CR131*={shards['cr_nonnull_total']}"
    )

    merged = report["aia_merged"]
    print("[aia merged]")
    if not merged.get("exists"):
        print("  missing: data/defn/aia_features.parquet")
    else:
        print(
            f"  rows={merged['rows']} cols={merged['cols']} harps={merged['harps']}"
        )
        stats = merged["column_stats"]
        print(
            "  CH*: "
            + ", ".join(
                f"{col}={stats[col]['nonnull']} non-null/{stats[col]['nonzero']} non-zero"
                for col in AIA_1600_FEATURES
            )
        )
        print(
            "  CR131*: "
            + ", ".join(
                f"{col}={stats[col]['nonnull']} non-null/{stats[col]['nonzero']} non-zero"
                for col in CR_COLS
            )
        )

    features = report["defn_features"]
    print("[defn features]")
    if not features.get("exists"):
        print("  missing: data/defn/defn_features.parquet")
    else:
        print(
            f"  rows={features['rows']} cols={features['cols']} "
            f"feature_cols={features['feature_cols_present']}/{len(ALL_FEATURE_COLS)}"
        )
        print(
            "  non-zero windows: "
            f"AIA1600={features['group_nonzero_windows']['aia_1600']} "
            f"dt24_AIA1600={features['group_nonzero_windows']['dt24_aia_1600']} "
            f"AIA131={features['group_nonzero_windows']['aia_131']}"
        )
        if features["missing_feature_cols"]:
            print(
                "  missing feature columns: "
                + ", ".join(features["missing_feature_cols"])
            )

    combined_blockers = blockers["aia"] + blockers["training"]
    print("[status]")
    if not combined_blockers:
        print("  READY: exact DeFN replication inputs are complete")
        return
    for blocker in combined_blockers:
        print(f"  BLOCKED: {blocker}")


@click.command()
@click.option("--defn-dir", default="data/defn", show_default=True)
@click.option("--windows", default="data/defn/defn_features.parquet", show_default=True)
@click.option("--num-shards", default=12, show_default=True)
@click.option("--max-batch-span-hours", default=48.0, show_default=True)
@click.option("--strict", is_flag=True, help="Fail if exact DeFN replication is not ready.")
@click.option("--strict-aia", is_flag=True, help="Fail if AIA 1600 refetch/merge is incomplete.")
@click.option("--strict-training", is_flag=True, help="Fail if defn_features is not exact-training ready.")
def main(
    defn_dir: str,
    windows: str,
    num_shards: int,
    max_batch_span_hours: float,
    strict: bool,
    strict_aia: bool,
    strict_training: bool,
) -> None:
    report = build_report(
        defn_dir=project_root / defn_dir,
        windows_path=project_root / windows,
        num_shards=num_shards,
        max_batch_span_hours=max_batch_span_hours,
    )
    blockers = readiness_blockers(report)
    print_report(report, blockers)

    active_blockers: list[str] = []
    if strict or strict_aia:
        active_blockers.extend(blockers["aia"])
    if strict or strict_training:
        active_blockers.extend(blockers["training"])
    if active_blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
