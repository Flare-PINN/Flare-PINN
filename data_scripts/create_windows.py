#!/usr/bin/env python3
"""
Windowing + Labeling for Physics-Informed Flare Forecasting
===========================================================

Builds rolling input windows per NOAA AR from preprocessed frame metadata,
and produces classification targets at multiple horizons with limb-censoring
and AR-event matching per the blueprint.

Inputs
------
1) Frame metadata (Parquet/CSV) from convert_and_preprocess_sharp.py with:
   - harpnum (int): NOAA AR identifier  
   - date_obs (str): observation timestamp (UTC)
   - cmd_deg (float): central meridian distance in degrees
   - frame_path (str): relative path to NPZ file
   - Bx_median, By_median, Bz_median, Bx_iqr, By_iqr, Bz_iqr (normalization stats)
   - Bx_nan, By_nan, Bz_nan (NaN fractions)
   - pxscale, pxunit (spatial scale metadata)
   - signs (coordinate system convention)

2) Event catalog (Parquet/CSV) from fetch_flares.py with:
   - start (datetime): flare start time
   - peak (datetime): flare peak time  
   - end (datetime): flare end time
   - class (str): GOES class like 'C3.2','M1.0','X2.1'
   - noaa_ar (int): NOAA AR number

Outputs
-------
- Windows Parquet: one row per (AR, t0) window with columns:
  harpnum, t0, input_span_hours, stride_hours, horizons_hours (list<int>),
  frame_count_in_window, obs_coverage, window_uid,
  and per-horizon:
    y_geq_M_{H}h, event_count_{H}h, is_masked_{H}h, is_partial_ok_{H}h, horizon_coverage_{H}h

- Ambiguity log CSV (optional): events that matched multiple ARs.

Usage
-----
python -m src.data.create_windows \\
  --cfg configs/data_train.yaml \\
  --input-hours 48 --stride-hours 6 --horizons 6 12 24 \\
  --cmd-threshold 70 --partial-ok-frac 0.7
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import List, Optional, Set, Tuple

import click
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.utils.common import load_cfg, DataConfig
from spatial_temporal_matching import spatial_temporal_match, combine_mappings


# ===================== Pydantic Models =====================

class WindowConfig(BaseModel):
    """Configuration for windowing parameters."""
    input_hours: int = Field(default=48, description="Input history window length in hours")
    stride_hours: int = Field(default=6, description="Stride between windows in hours")
    horizons: List[int] = Field(default=[6, 12, 24], description="Prediction horizons in hours")
    cmd_threshold: float = Field(default=70.0, description="CMD threshold for limb masking (degrees)")
    partial_ok_frac: float = Field(default=0.7, description="Fraction of horizon observable to accept partial windows")


class FrameColumns(BaseModel):
    """Strongly typed column mapping for frame metadata."""
    harpnum: str = Field(default="harpnum", description="HARP number / AR ID")
    date_obs: str = Field(default="date_obs", description="Observation timestamp")
    cmd_deg: str = Field(default="cmd_deg", description="Central meridian distance")
    frame_path: str = Field(default="frame_path", description="Path to frame data")


class EventColumns(BaseModel):
    """Strongly typed column mapping for event catalog."""
    start: str = Field(default="start", description="Flare start time")
    peak: str = Field(default="peak", description="Flare peak time")
    end: str = Field(default="end", description="Flare end time")
    cls: str = Field(default="class", description="GOES class")
    noaa_ar: str = Field(default="noaa_ar", description="NOAA AR number")


class WindowRow(BaseModel):
    """Single window output row with strong typing."""
    harpnum: int
    t0: str  # Will be converted to datetime in DataFrame
    input_span_hours: int
    stride_hours: int
    horizons_hours: Tuple[int, ...]
    frame_count_in_window: int
    obs_coverage: float
    window_uid: str


# ===================== Helper Functions =====================

def _read_any(path: str) -> pd.DataFrame:
    """Read Parquet or CSV into a DataFrame, preserving dtypes when possible."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file extension for: {path}")
    return df


def _load_flares_from_chunks(chunks_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """Load flares from monthly chunk files for the specified year range."""
    # Find all M-class chunk files for the year range
    all_chunks = []
    for year in range(start_year, end_year + 1):
        year_chunks = list(chunks_dir.glob(f"M_{year}-*.parquet"))
        all_chunks.extend(year_chunks)
    
    print(f"    Found {len(all_chunks)} chunk files for years {start_year}-{end_year}")
    
    if not all_chunks:
        return pd.DataFrame(columns=['start', 'peak', 'end', 'class', 'noaa_ar'])
    
    dfs = []
    for chunk_file in sorted(all_chunks):
        try:
            df = pd.read_parquet(chunk_file)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not read {chunk_file.name}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['start', 'peak', 'end', 'class', 'noaa_ar'])
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates().sort_values('start').reset_index(drop=True)
    
    return combined


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convert series to datetime, UTC-aware, with error handling."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def flare_is_geq_M(cls: str) -> bool:
    """Return True if flare class is >= M (i.e., M or X)."""
    if not isinstance(cls, str) or len(cls) == 0:
        return False
    lead = cls.strip().upper()[0]
    return lead in ("M", "X")


def _stable_uid(*parts: str) -> str:
    """Generate stable hex UID from string parts."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


# ===================== Main Windowing Logic =====================

def build_windows(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    frame_cols: FrameColumns,
    event_cols: EventColumns,
    config: WindowConfig,
    harp_noaa_mapping: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main routine. Returns (windows_df, ambiguous_events_df).

    windows_df: One row per (HARP, t0).
    ambiguous_events_df: Events that matched multiple HARPs at event time.
    """
    assert 0 < config.partial_ok_frac <= 1.0
    horizons = list(sorted(int(h) for h in config.horizons))

    # Prepare frames
    df = frames.copy()
    # Required columns
    for c in [frame_cols.harpnum, frame_cols.date_obs, frame_cols.cmd_deg]:
        if c not in df.columns:
            raise KeyError(f"Frame index missing required column: '{c}'")

    # Datetimes & sorting
    df[frame_cols.date_obs] = _to_datetime(df[frame_cols.date_obs])
    df = df.sort_values([frame_cols.harpnum, frame_cols.date_obs]).reset_index(drop=True)

    # Prepare events
    ev = events.copy()
    if event_cols.start not in ev.columns or event_cols.cls not in ev.columns:
        raise KeyError(f"Events missing required columns: '{event_cols.start}', '{event_cols.cls}'")
    ev[event_cols.start] = _to_datetime(ev[event_cols.start])
    ev["is_geq_M"] = ev[event_cols.cls].apply(flare_is_geq_M)

    # Load HARP-NOAA mapping if provided
    if harp_noaa_mapping is not None and len(harp_noaa_mapping) > 0:
        print(f"  Using HARP-NOAA mapping ({len(harp_noaa_mapping)} entries)")
        
        # Create BOTH directions for efficient lookup
        # NOAA_AR -> list of HARPs (for forward lookup)
        noaa_to_harps = {}
        # HARP -> list of NOAA ARs (for reverse lookup - more efficient)
        harp_to_noaas = {}
        
        for _, row in harp_noaa_mapping.iterrows():
            noaa_ar = int(row['noaa_ar'])
            harp = int(row['harpnum'])
            
            # Forward mapping
            if noaa_ar not in noaa_to_harps:
                noaa_to_harps[noaa_ar] = []
            noaa_to_harps[noaa_ar].append(harp)
            
            # Reverse mapping (more efficient for per-HARP loop)
            if harp not in harp_to_noaas:
                harp_to_noaas[harp] = []
            harp_to_noaas[harp].append(noaa_ar)
        
        # Convert event NOAA ARs to numeric
        ev["_noaa_ar_num"] = pd.to_numeric(ev[event_cols.noaa_ar], errors='coerce')
        
        # Report mapping statistics
        harps_with_noaa = len(harp_to_noaas)
        noaas_with_harp = len(noaa_to_harps)
        print(f"    {harps_with_noaa} HARPs mapped to {noaas_with_harp} NOAA ARs")
    else:
        print(f"    No HARP-NOAA mapping provided - labels will be 0%")
        noaa_to_harps = {}
        harp_to_noaas = {}
        ev["_noaa_ar_num"] = None

    # Group frames by HARP
    windows = []
    ambiguous_rows = []

    # Precompute per-HARP time arrays for nearest lookup
    per_harp = {k: v.reset_index(drop=True) for k, v in df.groupby(frame_cols.harpnum, sort=False)}

    # Build windows per HARP
    for harp_key, g in per_harp.items():
        times = g[frame_cols.date_obs].values
        if len(times) == 0:
            continue

        # Build candidate t0's
        t_min = g[frame_cols.date_obs].min()
        t_max = g[frame_cols.date_obs].max()

        # Ensure we can cover the input history [t0 - input_hours, t0]
        t0_start = t_min + pd.Timedelta(hours=config.input_hours)
        t0_end = t_max  # Allow horizons to extend beyond frame coverage

        if t0_start > t0_end:
            continue

        stride = pd.Timedelta(hours=config.stride_hours)
        t0s = pd.date_range(t0_start, t0_end, freq=stride, inclusive="left")

        # Useful arrays for nearest lookup
        g_times = g[frame_cols.date_obs]
        g_cmd = g[frame_cols.cmd_deg].astype(float)
        g_times_np = g_times.values.astype("datetime64[ns]")

        # Per-HARP event subset to speed filtering
        max_h = max(horizons) if len(horizons) else 0
        ev_window_min = t_min
        ev_window_max = t_max + pd.Timedelta(hours=max_h)
        ev_harp = ev[
            (ev[event_cols.start] >= ev_window_min - pd.Timedelta(days=2)) & 
            (ev[event_cols.start] <= ev_window_max + pd.Timedelta(days=2))
        ]

        # Match events using HARP-NOAA mapping (improved efficiency)
        if harp_noaa_mapping is not None and len(harp_noaa_mapping) > 0:
            # Use precomputed reverse mapping: HARP -> NOAA ARs
            this_harp_noaa_ars = harp_to_noaas.get(int(harp_key), [])
            
            # Get events that match any of these NOAA ARs
            if len(this_harp_noaa_ars) > 0:
                ev_exact = ev_harp[ev_harp["_noaa_ar_num"].isin(this_harp_noaa_ars)]
            else:
                # This HARP has no NOAA AR mapping - no events can be matched
                ev_exact = pd.DataFrame(columns=ev_harp.columns)
        else:
            ev_exact = pd.DataFrame(columns=ev_harp.columns)

        # Precompute set of matched event IDs for this HARP
        def _event_id(e_row: pd.Series) -> str:
            return _stable_uid(
                str(e_row.get(event_cols.start)), 
                str(e_row.get(event_cols.cls)), 
                str(e_row.get(event_cols.noaa_ar, ""))
            )

        matched_event_ids: Set[str] = set()
        for _, r in ev_exact.iterrows():
            matched_event_ids.add(_event_id(r))

        # Build windows for this HARP
        for t0 in t0s:
            t_start = t0 - pd.Timedelta(hours=config.input_hours)
            obs_mask = (g_times >= t_start) & (g_times <= t0)
            obs_frames = g[obs_mask]

            frame_count = int(obs_frames.shape[0])
            # Expected count at 1h cadence is input_hours + 1 (inclusive)
            expected = int(config.input_hours) + 1
            obs_coverage = frame_count / max(1, expected)

            # Window UID
            window_uid = _stable_uid(str(harp_key), str(pd.Timestamp(t0).isoformat()))

            # Check if this HARP has any NOAA mapping (needed for valid labels)
            has_noaa_mapping = (harp_noaa_mapping is not None and 
                               len(harp_noaa_mapping) > 0 and 
                               int(harp_key) in harp_to_noaas)
            
            row = {
                "harpnum": int(harp_key),
                "t0": pd.Timestamp(t0),
                "input_span_hours": config.input_hours,
                "stride_hours": config.stride_hours,
                "horizons_hours": tuple(horizons),
                "frame_count_in_window": frame_count,
                "obs_coverage": obs_coverage,
                "window_uid": window_uid,
                "has_noaa_mapping": bool(has_noaa_mapping),
            }

            # Observability & labels per horizon
            for H in horizons:
                H_td = pd.Timedelta(hours=H)
                t1 = t0 + H_td

                # Observability: sample at 1h grid within [t0, t1)
                grid = pd.date_range(t0, t1, freq="1h", inclusive="left")
                if len(grid) == 0:
                    frac_in = 0.0
                    is_masked = True
                else:
                    # Nearest cmd for each grid time
                    grid_np = grid.values.astype("datetime64[ns]")
                    idx = np.searchsorted(g_times_np, grid_np, side="left")
                    idx = np.clip(idx, 0, len(g_times_np) - 1)
                    cmd_vals = g_cmd.iloc[idx].to_numpy()
                    within = np.abs(cmd_vals) <= float(config.cmd_threshold)
                    frac_in = float(within.mean())
                    is_masked = not np.all(within)

                is_partial_ok = (frac_in >= float(config.partial_ok_frac))

                # Labeling: events in [t0, t1) that match this HARP and are >= M
                ev_slice = ev_harp[
                    (ev_harp[event_cols.start] >= t0) & 
                    (ev_harp[event_cols.start] < t1) & 
                    (ev_harp["is_geq_M"])
                ].copy()

                # Count events that match (exact) for this HARP
                count = 0
                for _, e in ev_slice.iterrows():
                    eid = _event_id(e)
                    if eid in matched_event_ids:
                        count += 1

                y = count > 0

                row[f"y_geq_M_{H}h"] = bool(y)
                row[f"event_count_{H}h"] = int(count)
                row[f"is_masked_{H}h"] = bool(is_masked)
                row[f"is_partial_ok_{H}h"] = bool(is_partial_ok)
                row[f"horizon_coverage_{H}h"] = float(frac_in)

            windows.append(row)

    windows_df = pd.DataFrame(windows).sort_values(["harpnum", "t0"]).reset_index(drop=True)

    # Ambiguity detection: events matched by >1 HARP (currently just using exact matches)
    # For simplicity, we're not doing spatial matching here since HEK provides noaa_ar
    amb_df = pd.DataFrame(columns=[
        "event_id", "start_time", "class", "noaa_ar", "n_candidate_harps"
    ])

    return windows_df, amb_df


# ===================== CLI =====================

@click.command()
@click.option("--cfg", default="configs/data_train.yaml", show_default=True, 
              help="Path to config YAML (or 'all' to process train/val/test)")
@click.option("--frames", default="S:/flare_forecasting/frames_meta.parquet", 
              help="Path to frames metadata parquet")
@click.option("--events", default="data/flares_hek.parquet",
              help="Path to events parquet")
@click.option("--out-suffix", default=None,
              help="Output filename suffix (default: derived from config)")
@click.option("--out-ambig", default=None,
              help="Optional CSV for ambiguous spatial matches")
@click.option("--use-config-params", is_flag=True, default=True,
              help="Use windowing parameters from config file")
@click.option("--input-hours", type=int, default=None)
@click.option("--stride-hours", type=int, default=None)
@click.option("--horizons", type=int, multiple=True, default=None)
@click.option("--cmd-threshold", type=float, default=None)
@click.option("--partial-ok-frac", type=float, default=0.70, show_default=True)
@click.option('--spatial-fallback', is_flag=True, default=False,
              help='Enable spatial-temporal fallback matching for unmapped HARPs')
def main(cfg: str, frames: str, events: str, 
         out_suffix: Optional[str], out_ambig: Optional[str],
         use_config_params: bool,
         input_hours: Optional[int], stride_hours: Optional[int], horizons: Optional[Tuple[int, ...]],
         cmd_threshold: Optional[float], partial_ok_frac: float, spatial_fallback: bool):
    """Build AR windows + labels for flare forecasting."""
    
    # Load frames once
    print(f"[create_windows] Loading frames from: {frames}")
    frames_df_all = _read_any(frames)
    print(f"  Loaded {len(frames_df_all)} frames")
    print(f"  Date range: {frames_df_all['date_obs'].min()} to {frames_df_all['date_obs'].max()}\n")
    
    # Determine chunks directory for events
    chunks_dir = Path(events).parent / "hek_chunks"
    
    # Handle batch mode for all configs
    if cfg.lower() == "all":
        configs = [
            ("configs/data_train.yaml", "train"),
            ("configs/data_validation.yaml", "validation"),
            ("configs/data_test.yaml", "test"),
        ]
        
        print(f"[create_windows] Processing ALL configs (train, validation, test)\n")
        
        # Process each config with its own year range
        for cfg_path, split_name in configs:
            _process_single_config(
                cfg_path, split_name, frames_df_all, chunks_dir,
                out_suffix, out_ambig, use_config_params,
                input_hours, stride_hours, horizons, cmd_threshold, partial_ok_frac,
                spatial_fallback
            )
        return
    
    # Single config mode
    split_name = out_suffix or Path(cfg).stem.replace("data_", "")
    
    _process_single_config(
        cfg, split_name, frames_df_all, chunks_dir,
        out_suffix, out_ambig, use_config_params,
        input_hours, stride_hours, horizons, cmd_threshold, partial_ok_frac,
        spatial_fallback
    )


def _process_single_config(
    cfg_path: str,
    split_name: str,
    frames_df_all: pd.DataFrame,
    chunks_dir: Path,
    out_suffix: Optional[str],
    out_ambig: Optional[str],
    use_config_params: bool,
    input_hours: Optional[int],
    stride_hours: Optional[int],
    horizons: Optional[Tuple[int, ...]],
    cmd_threshold: Optional[float],
    partial_ok_frac: float,
    spatial_fallback: bool = False,
):
    """Process a single config file."""
    print(f"{'='*60}")
    print(f"Processing: {split_name.upper()} ({cfg_path})")
    print(f"{'='*60}")
    
    # Load config
    data_config = load_cfg(cfg_path)
    
    # Filter data to config time span
    span_start = pd.to_datetime(data_config.span.start, utc=True)
    span_end = pd.to_datetime(data_config.span.end, utc=True)
    
    print(f"  Time span: {span_start.date()} to {span_end.date()}")
    
    # Filter frames
    frames_df = frames_df_all[
        (pd.to_datetime(frames_df_all['date_obs'], utc=True) >= span_start) &
        (pd.to_datetime(frames_df_all['date_obs'], utc=True) <= span_end)
    ].copy()
    
    # Load events from chunks for this year range
    start_year = span_start.year
    end_year = span_end.year
    print(f"  Loading events from chunks...")
    events_df = _load_flares_from_chunks(chunks_dir, start_year, end_year)
    
    # Filter to exact time range with buffer
    events_df = events_df[
        (pd.to_datetime(events_df['start'], utc=True) >= span_start - pd.Timedelta(days=2)) &
        (pd.to_datetime(events_df['start'], utc=True) <= span_end + pd.Timedelta(days=2))
    ].copy()
    
    print(f"  Filtered to {len(frames_df)} frames, {len(events_df)} events")
    
    # Determine windowing parameters
    if use_config_params:
        _input_hours = data_config.window_hours
        _stride_hours = data_config.window_stride_hours
        _horizons = data_config.targets_hours
        _cmd_threshold = data_config.censoring.cmd_deg_threshold
    else:
        _input_hours = input_hours or 48
        _stride_hours = stride_hours or 6
        _horizons = list(horizons) if horizons else [6, 12, 24]
        _cmd_threshold = cmd_threshold or 70.0
    
    window_config = WindowConfig(
        input_hours=_input_hours,
        stride_hours=_stride_hours,
        horizons=_horizons,
        cmd_threshold=_cmd_threshold,
        partial_ok_frac=partial_ok_frac,
    )
    
    print(f"  Input window: {window_config.input_hours}h")
    print(f"  Stride: {window_config.stride_hours}h")
    print(f"  Horizons: {window_config.horizons}h")
    print(f"  CMD threshold: {window_config.cmd_threshold}°\n")
    
    # Load HARP-NOAA mapping
    mapping_path = Path(data_config.paths.interim_dir) / "harp_noaa_mapping.parquet"
    if mapping_path.exists():
        harp_noaa_mapping = pd.read_parquet(mapping_path)
        print(f"  Loaded HARP-NOAA mapping: {len(harp_noaa_mapping)} entries")
        
        # Apply spatial-temporal fallback if requested
        if spatial_fallback:
            print(f"\n  Applying spatial-temporal fallback matching...")
            try:
                spatial_matches = spatial_temporal_match(
                    flares_df=events_df,
                    frames_df=frames_df,
                    existing_mapping=harp_noaa_mapping,
                    time_window_hours=1.0,
                    margin_deg=5.0
                )
                
                if len(spatial_matches) > 0:
                    harp_noaa_mapping = combine_mappings(harp_noaa_mapping, spatial_matches)
                    print(f"  Enhanced mapping: {len(harp_noaa_mapping)} total entries")
                else:
                    print(f"  No additional spatial matches found")
            except Exception as e:
                print(f"  WARNING: Spatial matching failed: {e}")
                print(f"  Continuing with ID-based mapping only")
    else:
        print(f"    HARP-NOAA mapping not found at {mapping_path}")
        harp_noaa_mapping = None
    
    # Build windows
    frame_cols = FrameColumns()
    event_cols = EventColumns()
    
    windows_df, amb_df = build_windows(
        frames=frames_df,
        events=events_df,
        frame_cols=frame_cols,
        event_cols=event_cols,
        config=window_config,
        harp_noaa_mapping=harp_noaa_mapping,
    )
    
    # Write outputs
    interim_dir = Path(data_config.paths.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    windows_path = interim_dir / f"windows_{split_name}.parquet"
    windows_df.to_parquet(windows_path, index=False)
    print(f"\n[OK] Wrote: {windows_path}  (rows={len(windows_df)})")
    
    if out_ambig and not amb_df.empty:
        amb_path = interim_dir / f"ambiguous_{split_name}.csv"
        amb_df.to_csv(amb_path, index=False)
        print(f"[OK] Wrote ambiguous: {amb_path}  (rows={len(amb_df)})")
    
    # Summary statistics
    if not windows_df.empty:
        print(f"\n[Summary]")
        print(f"  Unique HARPs: {windows_df['harpnum'].nunique()}")
        print(f"  Time range: {windows_df['t0'].min()} to {windows_df['t0'].max()}")
        print(f"  Avg frames/window: {windows_df['frame_count_in_window'].mean():.1f}")
        print(f"  Avg obs coverage: {windows_df['obs_coverage'].mean():.2%}")
        
        # Label coverage analysis
        if 'has_noaa_mapping' in windows_df.columns:
            n_labeled = windows_df['has_noaa_mapping'].sum()
            pct_labeled = 100.0 * n_labeled / len(windows_df)
            print(f"\n[Label Coverage]")
            print(f"  Windows with NOAA mapping: {n_labeled}/{len(windows_df)} ({pct_labeled:.1f}%)")
            print(f"  WARNING: Train only on has_noaa_mapping==True to avoid false negatives!")
        
        for H in window_config.horizons:
            y_col = f"y_geq_M_{H}h"
            if y_col in windows_df.columns:
                # Overall stats
                n_pos = windows_df[y_col].sum()
                pct_pos = 100.0 * n_pos / len(windows_df) if len(windows_df) > 0 else 0.0
                
                # Stats for labeled subset only
                if 'has_noaa_mapping' in windows_df.columns:
                    labeled_df = windows_df[windows_df['has_noaa_mapping']]
                    if len(labeled_df) > 0:
                        n_pos_labeled = labeled_df[y_col].sum()
                        pct_pos_labeled = 100.0 * n_pos_labeled / len(labeled_df)
                        print(f"  {H}h horizon: {n_pos} positive ({pct_pos:.2f}% overall, {pct_pos_labeled:.2f}% in labeled subset)")
                    else:
                        print(f"  {H}h horizon: {n_pos} positive ({pct_pos:.2f}%)")
                else:
                    print(f"  {H}h horizon: {n_pos} positive ({pct_pos:.2f}%)")
    print()


if __name__ == "__main__":
    main()
