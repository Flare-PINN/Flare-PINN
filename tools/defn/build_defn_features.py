#!/usr/bin/env python3
"""
Build the DeFN feature matrix — exact 79 features from Nishizuka et al.

Matches Table 1 of Nishizuka et al. 2017 (ApJ 835:156) for the base 65, plus
14 additions from Nishizuka et al. 2018 (ApJ 858:113) for the full 79.

══════════════════════════════════════════════════════════════════════════════
EXACT FEATURE SPECIFICATION (79 total)
══════════════════════════════════════════════════════════════════════════════

INSTANTANEOUS (34):
  ── SHARP keywords (†-marked in Table 1, from Bobra & Couvidat 2015) ────
  USFlux, MeanGAM, MeanGBt, MeanGBh, MeanGBz, MeanJzd,
  TotUSJz, MeanJzh, TotUSJh, ABSnJzh, SavNCPP, Area,
  TotBSQ, TotFx, TotFY, TotFz                                     (16)

  ── Magnetogram-derived (from Bz image, NOT SHARP keywords) ────────────
  Bmax, Bmin, Bave, MaxdxBz, MaxdyBz                                (5)

  ── Magnetic neutral lines (from magnetogram PIL analysis) ─────────────
  TotNL, NumNL, MaxNL                                                (3)

  ── AIA 1600 Å UV brightening ──────────────────────────────────────────
  CHArea, CHAll, CHMax                                               (3)

  ── GOES X-ray ─────────────────────────────────────────────────────────
  Xflux1h, Xflux4h, Xmax1d                                          (3)

  ── Flare history ──────────────────────────────────────────────────────
  Xhis, Mhis, Xhis1d, Mhis1d                                        (4)

TIME DERIVATIVES (31):
  24h derivatives of 26 features (includes MaxGraB and MaxdzBy which
      appear ONLY as derivatives, not in the instantaneous list)
  12h derivatives of 3 features (Area, Bmax, USFlux)
  2h  derivatives of 2 features (Area, Bmax)

2018 ADDITIONS (14):
  ── AIA 131 Å coronal hot brightening ──────────────────────────────────
  CRArea, CRAll, CRMax                      (3: at t0)
  CRArea_1h, CRAll_1h, CRMax_1h             (3: at t0−1h)
  CRArea_2h, CRAll_2h, CRMax_2h             (3: at t0−2h)

  ── GOES at time offsets ───────────────────────────────────────────────
  Xflux_1hbef, Xflux_2hbef                  (2: X-ray at t0−1h, t0−2h)

  ── 24h derivatives of 131 Å ───────────────────────────────────────────
  dt24_CRArea, dt24_CRAll, dt24_CRMax       (3)

══════════════════════════════════════════════════════════════════════════════
TOTAL: 34 + 31 + 14 = 79
══════════════════════════════════════════════════════════════════════════════

Output: data/defn/defn_features.parquet

Prerequisites:
    python tools/defn/fetch_sharp_keywords.py
    python tools/defn/fetch_goes_xrs.py
    python tools/defn/fetch_aia.py
    Optional for Table 1 magnetogram + neutral-line terms: per-HARP consolidated NPZ
    (see README / ``data_scripts/consolidate_frames.py``). Set env ``DEFN_CONSOLIDATED_DIR``
    or rely on ``~/flare_data/consolidated`` when that path exists.

Usage:
    conda activate flare-pinn
    python tools/defn/build_defn_features.py
    # Or full sequence:  bash tools/defn/replicate_defn_pipeline.sh
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _resolve_consolidated_dir(cli_value: str | None) -> Path | None:
    """
    DeFN Table 1 magnetogram + neutral-line terms need per-HARP SHARP NPZ stacks
    (same layout as ``data_scripts/consolidate_frames.py``).

    Resolution order: ``--consolidated-dir`` CLI → env ``DEFN_CONSOLIDATED_DIR`` →
    ``~/flare_data/consolidated`` if that directory exists.
    """
    if cli_value:
        p = Path(cli_value).expanduser()
        if not p.is_dir():
            raise click.BadParameter(
                f"Consolidated NPZ directory not found: {p}",
            )
        return p
    env = os.environ.get("DEFN_CONSOLIDATED_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            print(
                f"[build_defn_features] Consolidated NPZ from DEFN_CONSOLIDATED_DIR: {p}",
            )
            return p
        print(
            f"[build_defn_features] WARN: DEFN_CONSOLIDATED_DIR={env!r} "
            "missing or not a directory; magneto/NL features will be 0",
        )
    auto = Path.home() / "flare_data" / "consolidated"
    if auto.is_dir():
        print(
            f"[build_defn_features] Using {auto} for magnetogram / PIL features "
            "(set DEFN_CONSOLIDATED_DIR to override)",
        )
        return auto
    return None


# ═══════════════════════════════════════════════════════════════════════════
# NPZ cache to avoid reloading per-HARP files repeatedly
# ═══════════════════════════════════════════════════════════════════════════

class _NPZCache:
    """LRU-style cache for consolidated NPZ data (frames + timestamps)."""

    def __init__(self, consolidated_dir: Path | None, max_size: int = 50):
        self._dir = consolidated_dir
        self._max = max_size
        self._cache: dict[int, tuple[np.ndarray, pd.DatetimeIndex] | None] = {}
        self._order: list[int] = []

    def get(self, harpnum: int) -> tuple[np.ndarray, pd.DatetimeIndex] | None:
        if harpnum in self._cache:
            return self._cache[harpnum]
        if self._dir is None:
            return None
        npz_path = self._dir / f"H{harpnum}.npz"
        if not npz_path.exists():
            self._cache[harpnum] = None
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            frames = data["frames"]
            times = pd.to_datetime(data["timestamps"], utc=True)
            entry = (frames, times)
        except Exception:
            entry = None
        self._cache[harpnum] = entry
        self._order.append(harpnum)
        if len(self._order) > self._max:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return entry

    def _find_idx(self, times: pd.DatetimeIndex, t0: pd.Timestamp, max_gap_h: float = 2.0) -> int | None:
        t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
        diffs = np.abs(times - t0_utc)
        idx = int(np.argmin(diffs))
        if diffs[idx] > pd.Timedelta(hours=max_gap_h):
            return None
        return idx


_npz_cache: _NPZCache | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Feature specification — matches Table 1 EXACTLY
# ═══════════════════════════════════════════════════════════════════════════

# 16 SHARP keywords (marked with † in Table 1; available from hmi.sharp_cea_720s)
# Paper name → JSOC keyword mapping
SHARP_FEATURES = [
    "USFLUX",   # USFlux  — Total unsigned flux
    "MEANGAM",  # MeanGAM — Mean angle of field from radial direction
    "MEANGBT",  # MeanGBt — Mean gradient of total field
    "MEANGBH",  # MeanGBh — Mean gradient of horizontal field
    "MEANGBZ",  # MeanGBz — Mean gradient of vertical field
    "MEANJZD",  # MeanJzd — Mean vertical current density
    "TOTUSJZ",  # TotUSJz — Total unsigned vertical current
    "MEANJZH",  # MeanJzh — Mean current helicity (Bz contributions)
    "TOTUSJH",  # TotUSJh — Total unsigned current helicity
    "ABSNJZH",  # ABSnJzh — Absolute value of net current per polarity
    "SAVNCPP",  # SavNCPP — Modules of net current per polarity
    "AREA_ACR", # Area    — Area of the strong field in an AR
    "TOTBSQ",   # TotBSQ  — Total magnitude of Lorentz force
    "TOTFX",    # TotFx   — Sum of X-component of Lorentz force
    "TOTFY",    # TotFY   — Sum of Y-component of Lorentz force
    "TOTFZ",    # TotFz   — Sum of Z-component of Lorentz force
]

# 5 magnetogram-derived features (computed from Bz image, NOT SHARP keywords)
MAGNETO_FEATURES = ["Bmax", "Bmin", "Bave", "MaxdxBz", "MaxdyBz"]

# 3 neutral line features (from magnetogram PIL analysis)
NEUTRAL_LINE_FEATURES = ["TotNL", "NumNL", "MaxNL"]

# 3 AIA 1600 Å UV brightening features
AIA_1600_FEATURES = ["CHArea", "CHAll", "CHMax"]

# 3 GOES X-ray features
GOES_FEATURES = ["Xflux1h", "Xflux4h", "Xmax1d"]

# 4 flare history features
FLARE_HISTORY_FEATURES = ["Xhis", "Mhis", "Xhis1d", "Mhis1d"]

# ── All 34 instantaneous features (in canonical order) ──────────────────
INSTANTANEOUS_COLS = (
    SHARP_FEATURES + MAGNETO_FEATURES + NEUTRAL_LINE_FEATURES +
    AIA_1600_FEATURES + GOES_FEATURES + FLARE_HISTORY_FEATURES
)
assert len(INSTANTANEOUS_COLS) == 34, f"Expected 34 instantaneous, got {len(INSTANTANEOUS_COLS)}"

# ── 24h derivative base features (26 total) ─────────────────────────────
# Two of these (MaxGraB, MaxdzBy) are derivative-only — they appear in
# Table 1 ONLY as dt24 derivatives, NOT as instantaneous features.
DT24_BASE = [
    "SAVNCPP", "TotNL", "TOTBSQ", "TOTFY", "TOTFX", "TOTFZ",
    "USFLUX", "AREA_ACR", "ABSNJZH", "TOTUSJZ", "Bmax", "CHArea",
    "MaxGraB",    # derivative-only: max magnitude of gradient of Bz
    "MaxdzBy",    # derivative-only: max dBy/dz
    "TOTUSJH", "NumNL", "MaxdxBz", "MEANJZH", "MaxNL", "CHAll",
    "CHMax", "MEANGBZ", "MEANGBH", "MEANGBT", "MEANGAM", "MEANJZD",
]
assert len(DT24_BASE) == 26, f"Expected 26 dt24 base features, got {len(DT24_BASE)}"

# 3 features with 12h derivatives
DT12_BASE = ["AREA_ACR", "Bmax", "USFLUX"]

# 2 features with 2h derivatives
DT02_BASE = ["AREA_ACR", "Bmax"]

# Build derivative column names
DT24_COLS = [f"dt24_{f}" for f in DT24_BASE]
DT12_COLS = [f"dt12_{f}" for f in DT12_BASE]
DT02_COLS = [f"dt02_{f}" for f in DT02_BASE]
DERIVATIVE_COLS = DT24_COLS + DT12_COLS + DT02_COLS
assert len(DERIVATIVE_COLS) == 31, f"Expected 31 derivatives, got {len(DERIVATIVE_COLS)}"

# ── 2018 additions: AIA 131 Å + GOES time offsets (14) ──────────────────
AIA_131_T0 = ["CRArea", "CRAll", "CRMax"]
AIA_131_1H = ["CRArea_1h", "CRAll_1h", "CRMax_1h"]
AIA_131_2H = ["CRArea_2h", "CRAll_2h", "CRMax_2h"]
GOES_OFFSET = ["Xflux_1hbef", "Xflux_2hbef"]
DT24_131 = ["dt24_CRArea", "dt24_CRAll", "dt24_CRMax"]
ADDITIONS_2018 = AIA_131_T0 + AIA_131_1H + AIA_131_2H + GOES_OFFSET + DT24_131
assert len(ADDITIONS_2018) == 14, f"Expected 14 additions, got {len(ADDITIONS_2018)}"

# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL 79-FEATURE VECTOR
# ═══════════════════════════════════════════════════════════════════════════
ALL_FEATURE_COLS = INSTANTANEOUS_COLS + DERIVATIVE_COLS + ADDITIONS_2018
assert len(ALL_FEATURE_COLS) == 79, f"Expected 79 features, got {len(ALL_FEATURE_COLS)}"


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction functions
# ═══════════════════════════════════════════════════════════════════════════

def _find_nearest(
    df: pd.DataFrame,
    harpnum: int,
    t_target: pd.Timestamp,
    max_gap_hours: float = 2.0,
    *,
    value_column: str | None = None,
) -> pd.Series | None:
    """
    Nearest observation in time within ``max_gap_hours`` of ``t_target``.

    DeFN (Nishizuka et al. 2017/2018) used full-disk AIA on a 1 h cadence and
    took scalars at the observation nearest each needed time. We use the same
    ±2 h tolerance as elsewhere in this module. When ``aia_features.parquet``
    stacks 1600 Å and 131 Å in one table, some rows are 1600-only or 131-only
    at a given ``t_obs``; ``value_column`` restricts the search to rows with
    valid scalars for that channel (analogous to independent nearest-hour picks
    on the two wavelength series in the original code).
    """
    mask = df["harpnum"] == harpnum
    harp_data = df.loc[mask]
    if value_column is not None and value_column in harp_data.columns:
        harp_data = harp_data.loc[harp_data[value_column].notna()]
    if harp_data.empty:
        return None

    t_utc = t_target.tz_localize("UTC") if t_target.tzinfo is None else t_target
    dt = (harp_data["t_obs"] - t_utc).abs()
    nearest_idx = dt.idxmin()
    if dt.loc[nearest_idx] > pd.Timedelta(hours=max_gap_hours):
        return None
    return harp_data.loc[nearest_idx]


def extract_sharp_features(
    sharp_df: pd.DataFrame, harpnum: int, t0: pd.Timestamp,
) -> dict[str, float]:
    """Extract 16 SHARP keyword values at t0."""
    result = {col: 0.0 for col in SHARP_FEATURES}
    if sharp_df.empty:
        return result

    row = _find_nearest(sharp_df, harpnum, t0)
    if row is None:
        return result

    for col in SHARP_FEATURES:
        v = row.get(col, np.nan)
        result[col] = float(v) if pd.notna(v) and np.isfinite(v) else 0.0
    return result


def extract_sharp_at_offset(
    sharp_df: pd.DataFrame, harpnum: int, t0: pd.Timestamp, offset_hours: float,
) -> dict[str, float]:
    """Extract SHARP values at t0 - offset_hours."""
    result = {col: 0.0 for col in SHARP_FEATURES}
    if sharp_df.empty:
        return result

    t_target = (t0.tz_localize("UTC") if t0.tzinfo is None else t0) - pd.Timedelta(hours=offset_hours)
    row = _find_nearest(sharp_df, harpnum, t_target)
    if row is None:
        return result

    for col in SHARP_FEATURES:
        v = row.get(col, np.nan)
        result[col] = float(v) if pd.notna(v) and np.isfinite(v) else 0.0
    return result


def compute_magneto_features(
    consolidated_dir: Path | None, harpnum: int, t0: pd.Timestamp,
) -> dict[str, float]:
    """
    Compute Bmax, Bmin, Bave, MaxdxBz, MaxdyBz from consolidated NPZ Bz data.
    Also compute MaxGraB (max gradient magnitude of Bz) and MaxdzBy (max dBy/dz).
    Uses the global _npz_cache for efficient repeated access.
    """
    base = {f: 0.0 for f in MAGNETO_FEATURES}
    extra = {"MaxGraB": 0.0, "MaxdzBy": 0.0}

    global _npz_cache
    if _npz_cache is None:
        return {**base, **extra}

    entry = _npz_cache.get(harpnum)
    if entry is None:
        return {**base, **extra}

    frames, times = entry
    idx = _npz_cache._find_idx(times, t0)
    if idx is None:
        return {**base, **extra}

    try:
        bz_frame = frames[idx, 2].astype(np.float64)
        base["Bmax"] = float(np.nanmax(np.abs(bz_frame)))
        base["Bmin"] = float(np.nanmin(bz_frame))
        base["Bave"] = float(np.nanmean(np.abs(bz_frame)))

        dxBz = np.gradient(bz_frame, axis=1)
        dyBz = np.gradient(bz_frame, axis=0)
        base["MaxdxBz"] = float(np.nanmax(np.abs(dxBz)))
        base["MaxdyBz"] = float(np.nanmax(np.abs(dyBz)))

        grad_mag = np.sqrt(dxBz**2 + dyBz**2)
        extra["MaxGraB"] = float(np.nanmax(grad_mag))

        by_frame = frames[idx, 1].astype(np.float64)
        dzBy = np.gradient(by_frame, axis=0)
        extra["MaxdzBy"] = float(np.nanmax(np.abs(dzBy)))
    except Exception:
        pass

    return {**base, **extra}


def compute_neutral_line_features(
    consolidated_dir: Path | None, harpnum: int, t0: pd.Timestamp,
) -> dict[str, float]:
    """Compute TotNL, NumNL, MaxNL from magnetogram PIL analysis."""
    result = {f: 0.0 for f in NEUTRAL_LINE_FEATURES}

    global _npz_cache
    if _npz_cache is None:
        return result

    entry = _npz_cache.get(harpnum)
    if entry is None:
        return result

    frames, times = entry
    idx = _npz_cache._find_idx(times, t0)
    if idx is None:
        return result

    try:
        bz_frame = frames[idx, 2].astype(np.float64)

        grad_mag = np.sqrt(
            np.gradient(bz_frame, axis=1)**2 +
            np.gradient(bz_frame, axis=0)**2
        )
        bz_range = float(np.nanmax(np.abs(bz_frame))) or 1.0
        grad_thresh = 0.03 * bz_range
        pil_mask = (grad_mag > grad_thresh) & (np.abs(bz_frame) < 0.1 * bz_range)

        from scipy import ndimage
        labeled, num_features = ndimage.label(pil_mask)
        result["NumNL"] = float(num_features)

        if num_features > 0:
            lengths = []
            for label_id in range(1, num_features + 1):
                lengths.append(float(np.sum(labeled == label_id)))
            result["TotNL"] = float(sum(lengths))
            result["MaxNL"] = float(max(lengths))

    except Exception:
        pass

    return result


def compute_aia_1600_features(
    aia_df: pd.DataFrame | None, harpnum: int, t0: pd.Timestamp,
) -> dict[str, float]:
    """Extract CHArea, CHAll, CHMax from AIA 1600 Å (nearest valid 1600 row to t0)."""
    result = {f: 0.0 for f in AIA_1600_FEATURES}
    if aia_df is None or aia_df.empty:
        return result

    t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    row = _find_nearest(aia_df, harpnum, t0_utc, max_gap_hours=2.0, value_column="CHArea")
    if row is None:
        return result
    for col in AIA_1600_FEATURES:
        v = row.get(col, 0.0)
        result[col] = float(v) if pd.notna(v) else 0.0
    return result


def compute_aia_131_features(
    aia_df: pd.DataFrame | None, harpnum: int, t0: pd.Timestamp,
) -> dict[str, float]:
    """
    Extract AIA 131 Å coronal hot brightening features at t0, t0-1h, t0-2h.
    Returns 9 features: CRArea/CRAll/CRMax at each time offset.
    """
    result = {f: 0.0 for f in AIA_131_T0 + AIA_131_1H + AIA_131_2H}
    if aia_df is None or aia_df.empty:
        return result

    t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0

    for offset_h, cols in [(0, AIA_131_T0), (1, AIA_131_1H), (2, AIA_131_2H)]:
        t_target = t0_utc - pd.Timedelta(hours=offset_h)
        row = _find_nearest(
            aia_df, harpnum, t_target, max_gap_hours=2.0, value_column="CR131Area",
        )
        if row is None:
            continue
        base_cols = ["CR131Area", "CR131All", "CR131Max"]
        for base, out in zip(base_cols, cols):
            v = row.get(base, 0.0)
            result[out] = float(v) if pd.notna(v) else 0.0

    return result


class _GOESIndex:
    """Pre-sorted GOES data with O(log N) time-range lookups."""

    def __init__(self, goes_df: pd.DataFrame):
        if goes_df.empty:
            self._times = np.array([], dtype="datetime64[ns]")
            self._flux = np.array([], dtype=np.float64)
            self.empty = True
            return
        times = goes_df["time"].values.astype("datetime64[ns]")
        flux = goes_df["xrsb_flux"].values.astype(np.float64)
        order = np.argsort(times)
        self._times = times[order]
        self._flux = flux[order]
        self.empty = False

    def _range(self, t_start: pd.Timestamp, t_end: pd.Timestamp) -> np.ndarray:
        ts = np.datetime64(t_start.tz_convert("UTC").tz_localize(None), "ns")
        te = np.datetime64(t_end.tz_convert("UTC").tz_localize(None), "ns")
        i0 = int(np.searchsorted(self._times, ts, side="left"))
        i1 = int(np.searchsorted(self._times, te, side="right"))
        return self._flux[i0:i1]

    def mean_flux(self, t0_utc: pd.Timestamp, hours: int) -> float:
        if self.empty:
            return 0.0
        t_start = t0_utc - pd.Timedelta(hours=hours)
        vals = self._range(t_start, t0_utc)
        return float(vals.mean()) if len(vals) > 0 else 0.0

    def max_flux(self, t0_utc: pd.Timestamp, hours: int) -> float:
        if self.empty:
            return 0.0
        t_start = t0_utc - pd.Timedelta(hours=hours)
        vals = self._range(t_start, t0_utc)
        return float(vals.max()) if len(vals) > 0 else 0.0

    def point_flux(self, t_utc: pd.Timestamp) -> float:
        if self.empty:
            return 0.0
        ts = np.datetime64(t_utc.tz_convert("UTC").tz_localize(None), "ns")
        idx = int(np.searchsorted(self._times, ts, side="left"))
        idx = min(idx, len(self._times) - 1)
        if abs(self._times[idx] - ts) > np.timedelta64(5, "m"):
            return 0.0
        return float(self._flux[idx])


_goes_index: _GOESIndex | None = None


def compute_goes_features(
    t0: pd.Timestamp,
) -> dict[str, float]:
    """
    DeFN GOES features at t0:
      Xflux1h:  Average X-ray flux over past 1h
      Xflux4h:  Average X-ray flux over past 4h
      Xmax1d:   Maximum X-ray intensity one day before
    """
    global _goes_index
    result = {"Xflux1h": 0.0, "Xflux4h": 0.0, "Xmax1d": 0.0}
    if _goes_index is None or _goes_index.empty:
        return result
    t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    result["Xflux1h"] = _goes_index.mean_flux(t0_utc, 1)
    result["Xflux4h"] = _goes_index.mean_flux(t0_utc, 4)
    result["Xmax1d"] = _goes_index.max_flux(t0_utc, 24)
    return result


def compute_goes_offsets(
    t0: pd.Timestamp,
) -> dict[str, float]:
    """
    2018 additions: X-ray flux at t0−1h and t0−2h (point values, not averages).
    """
    global _goes_index
    result = {"Xflux_1hbef": 0.0, "Xflux_2hbef": 0.0}
    if _goes_index is None or _goes_index.empty:
        return result

    t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0

    for offset_h, key in [(1, "Xflux_1hbef"), (2, "Xflux_2hbef")]:
        t_target = t0_utc - pd.Timedelta(hours=offset_h)
        result[key] = _goes_index.point_flux(t_target)

    return result


def compute_flare_history(
    flares_df: pd.DataFrame, noaa_ars: list[int], t0: pd.Timestamp,
    harp_flare_index: dict | None = None, harpnum: int = 0,
) -> dict[str, float]:
    """
    DeFN flare history features:
      Xhis:   Total number of X-class flares ever in this AR
      Mhis:   Total number of M-class flares ever in this AR (≥M)
      Xhis1d: X-class flares in past 24h
      Mhis1d: M-class flares in past 24h

    Uses harp_flare_index (precomputed from window labels) if the flare
    catalog has poor NOAA AR coverage.
    """
    result = {"Xhis": 0.0, "Mhis": 0.0, "Xhis1d": 0.0, "Mhis1d": 0.0}
    t0_utc = t0.tz_localize("UTC") if t0.tzinfo is None else t0

    # Try flare catalog first
    if not flares_df.empty and noaa_ars:
        ar_flares = flares_df[
            flares_df["noaa_ar"].isin(noaa_ars) & (flares_df["start"] < t0_utc)
        ]
        if not ar_flares.empty:
            result["Xhis"] = float((ar_flares["letter"] == "X").sum())
            result["Mhis"] = float((ar_flares["letter"] == "M").sum())

            t_24h = t0_utc - pd.Timedelta(hours=24)
            recent = ar_flares[ar_flares["start"] >= t_24h]
            if not recent.empty:
                result["Xhis1d"] = float((recent["letter"] == "X").sum())
                result["Mhis1d"] = float((recent["letter"] == "M").sum())
            return result

    # Fallback: use precomputed flare index from window labels
    if harp_flare_index is not None and harpnum in harp_flare_index:
        flare_times = harp_flare_index[harpnum]
        past_flares = [t for t in flare_times if t < t0_utc]
        result["Mhis"] = float(len(past_flares))
        t_24h = t0_utc - pd.Timedelta(hours=24)
        result["Mhis1d"] = float(sum(1 for t in past_flares if t >= t_24h))

    return result


def compute_time_derivatives(
    vals_now: dict[str, float], vals_past: dict[str, float],
    features: list[str], hours: float,
) -> dict[str, float]:
    """Compute (val_now - val_past) / hours for given features."""
    prefix = f"dt{int(hours):02d}"
    result: dict[str, float] = {}
    for f in features:
        v_now = vals_now.get(f, 0.0)
        v_past = vals_past.get(f, 0.0)
        if hours > 0 and v_now != 0.0 and v_past != 0.0:
            result[f"{prefix}_{f}"] = (v_now - v_past) / hours
        else:
            result[f"{prefix}_{f}"] = 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

@click.command()
@click.option("--output", default="data/defn/defn_features.parquet")
@click.option("--sharp-path", default="data/defn/sharp_keywords.parquet")
@click.option("--goes-path", default="data/defn/goes_xrs.parquet")
@click.option("--aia-path", default="data/defn/aia_features.parquet",
              help="AIA features parquet (1600 Å + 131 Å). Optional.")
@click.option("--windows-path", default="data/scalar_features.parquet")
@click.option("--mapping-path", default="data/harp_noaa_mapping.parquet")
@click.option(
    "--consolidated-dir",
    default=None,
    help=(
        "Per-HARP consolidated NPZ directory (H<harp>.npz) for Bz-derived and "
        "neutral-line features. If omitted, uses env DEFN_CONSOLIDATED_DIR, "
        "then ~/flare_data/consolidated when present."
    ),
)
def main(
    output: str,
    sharp_path: str,
    goes_path: str,
    aia_path: str,
    windows_path: str,
    mapping_path: str,
    consolidated_dir: str | None,
):
    out_path = project_root / output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cons_dir = _resolve_consolidated_dir(consolidated_dir)

    global _npz_cache
    _npz_cache = _NPZCache(cons_dir, max_size=200)

    print("[build_defn_features] Loading data sources...")

    windows = pd.read_parquet(project_root / windows_path)
    print(f"  Windows: {len(windows)} rows")

    mapping = pd.read_parquet(project_root / mapping_path)
    harp_to_noaa: dict[int, list[int]] = (
        mapping.groupby("harpnum")["noaa_ar"].apply(list).to_dict()
    )
    print(f"  HARP-NOAA mapping: {len(harp_to_noaa)} HARPs")

    sharp_df = _safe_load(project_root / sharp_path, "SHARP keywords")
    goes_path_full = project_root / goes_path
    aia_df = _safe_load(project_root / aia_path, "AIA features")

    global _goes_index
    if goes_path_full.exists():
        print(f"  GOES XRS: loading and indexing {goes_path_full}...")
        gdf = pd.read_parquet(goes_path_full, columns=["time", "xrsb_flux"])
        gdf["time"] = pd.to_datetime(gdf["time"], utc=True)
        print(f"  GOES: {len(gdf)} raw records, building index...")
        _goes_index = _GOESIndex(gdf)
        del gdf
        print(f"  GOES index: {len(_goes_index._flux)} records")
    else:
        print(f"  GOES XRS: NOT FOUND at {goes_path_full}")
        _goes_index = _GOESIndex(pd.DataFrame())
    flares_df = _load_flare_catalog()
    print(f"  Flares: {len(flares_df)} events")

    # Build flare index from window labels as fallback
    harp_flare_index = _build_harp_flare_index(windows)
    n_harps_flaring = sum(1 for v in harp_flare_index.values() if v)
    print(f"  Flare index (from labels): {n_harps_flaring} HARPs with flares")

    if cons_dir:
        print(f"  Consolidated NPZ: {cons_dir}")
    else:
        print("  Consolidated NPZ: not provided (magneto/neutral-line features = 0)")

    # ── Build feature matrix ─────────────────────────────────────────────
    print(f"\n[build_defn_features] Computing {len(ALL_FEATURE_COLS)} features "
          f"for {len(windows)} windows...")
    print(f"  Instantaneous: {len(INSTANTANEOUS_COLS)}")
    print(f"  Derivatives:   {len(DERIVATIVE_COLS)}")
    print(f"  2018 additions:{len(ADDITIONS_2018)}")

    feature_rows: list[dict] = []
    total = len(windows)

    windows = windows.sort_values("harpnum")

    for i, (_, row) in enumerate(windows.iterrows()):
        if i % 2000 == 0:
            print(f"  {i}/{total} ({100 * i / total:.1f}%)", flush=True)

        harpnum = int(row["harpnum"])
        t0 = pd.Timestamp(row["t0"])
        noaa_ars = harp_to_noaa.get(harpnum, [])

        feat: dict = {"harpnum": harpnum, "t0": t0}

        for col in ["y_geq_M_6h", "y_geq_M_12h", "y_geq_M_24h",
                     "has_noaa_mapping", "window_uid"]:
            if col in row.index:
                feat[col] = row[col]
        if "is_val" in row.index:
            feat["is_val"] = row["is_val"]

        # ── 1. SHARP keywords (16 features) ──────────────────────────────
        sharp_now = extract_sharp_features(sharp_df, harpnum, t0)
        feat.update(sharp_now)

        # ── 2. Magnetogram-derived (5 features + 2 derivative-only) ──────
        magneto = compute_magneto_features(cons_dir, harpnum, t0)
        for f in MAGNETO_FEATURES:
            feat[f] = magneto[f]

        # ── 3. Neutral line features (3) ─────────────────────────────────
        nl = compute_neutral_line_features(cons_dir, harpnum, t0)
        feat.update(nl)

        # ── 4. AIA 1600 Å features (3) ──────────────────────────────────
        aia_1600 = compute_aia_1600_features(aia_df, harpnum, t0)
        feat.update(aia_1600)

        # ── 5. GOES X-ray (3) ───────────────────────────────────────────
        goes = compute_goes_features(t0)
        feat.update(goes)

        # ── 6. Flare history (4) ────────────────────────────────────────
        hist = compute_flare_history(
            flares_df, noaa_ars, t0,
            harp_flare_index=harp_flare_index, harpnum=harpnum,
        )
        feat.update(hist)

        # ── TIME DERIVATIVES (31) ───────────────────────────────────────

        # Merge all current values for derivative computation
        all_now = {
            **sharp_now,
            **{f: magneto[f] for f in MAGNETO_FEATURES},
            "MaxGraB": magneto["MaxGraB"],
            "MaxdzBy": magneto["MaxdzBy"],
            **nl, **aia_1600,
        }

        # 24h derivatives (26)
        sharp_24h = extract_sharp_at_offset(sharp_df, harpnum, t0, 24.0)
        magneto_24h = compute_magneto_features(cons_dir, harpnum,
                      t0 - pd.Timedelta(hours=24)) if cons_dir else {f: 0.0 for f in MAGNETO_FEATURES + ["MaxGraB", "MaxdzBy"]}
        nl_24h = compute_neutral_line_features(cons_dir, harpnum,
                 t0 - pd.Timedelta(hours=24)) if cons_dir else {f: 0.0 for f in NEUTRAL_LINE_FEATURES}
        aia_1600_24h = compute_aia_1600_features(aia_df, harpnum,
                       t0 - pd.Timedelta(hours=24))

        all_24h = {
            **sharp_24h,
            **{f: magneto_24h[f] for f in MAGNETO_FEATURES},
            "MaxGraB": magneto_24h.get("MaxGraB", 0.0),
            "MaxdzBy": magneto_24h.get("MaxdzBy", 0.0),
            **nl_24h, **aia_1600_24h,
        }
        feat.update(compute_time_derivatives(all_now, all_24h, DT24_BASE, 24.0))

        # 12h derivatives (3)
        sharp_12h = extract_sharp_at_offset(sharp_df, harpnum, t0, 12.0)
        magneto_12h = compute_magneto_features(cons_dir, harpnum,
                      t0 - pd.Timedelta(hours=12)) if cons_dir else {f: 0.0 for f in MAGNETO_FEATURES + ["MaxGraB", "MaxdzBy"]}
        now_12 = {**sharp_now, "Bmax": magneto["Bmax"]}
        past_12 = {**sharp_12h, "Bmax": magneto_12h.get("Bmax", 0.0)}
        feat.update(compute_time_derivatives(now_12, past_12, DT12_BASE, 12.0))

        # 2h derivatives (2)
        sharp_2h = extract_sharp_at_offset(sharp_df, harpnum, t0, 2.0)
        magneto_2h = compute_magneto_features(cons_dir, harpnum,
                     t0 - pd.Timedelta(hours=2)) if cons_dir else {f: 0.0 for f in MAGNETO_FEATURES + ["MaxGraB", "MaxdzBy"]}
        now_2 = {**sharp_now, "Bmax": magneto["Bmax"]}
        past_2 = {**sharp_2h, "Bmax": magneto_2h.get("Bmax", 0.0)}
        feat.update(compute_time_derivatives(now_2, past_2, DT02_BASE, 2.0))

        # ── 2018 ADDITIONS (14) ─────────────────────────────────────────

        # AIA 131 Å at t0, t0-1h, t0-2h (9)
        feat.update(compute_aia_131_features(aia_df, harpnum, t0))

        # GOES at t0-1h, t0-2h (2)
        feat.update(compute_goes_offsets(t0))

        # 24h derivatives of 131 Å (3)
        aia_131_now = {k: feat.get(k, 0.0) for k in AIA_131_T0}
        aia_131_24h = compute_aia_131_features(aia_df, harpnum,
                      t0 - pd.Timedelta(hours=24))
        for base, deriv in zip(AIA_131_T0, DT24_131):
            v_now = aia_131_now.get(base, 0.0)
            v_past = aia_131_24h.get(base, 0.0)
            if v_now != 0.0 and v_past != 0.0:
                feat[deriv] = (v_now - v_past) / 24.0
            else:
                feat[deriv] = 0.0

        feature_rows.append(feat)

    # ── Save ─────────────────────────────────────────────────────────────
    result = pd.DataFrame(feature_rows)
    result.to_parquet(out_path, index=False)

    print(f"\n[done] Saved {out_path}")
    print(f"  {len(result)} windows x {len(ALL_FEATURE_COLS)} features")
    _print_coverage(result)


def _print_coverage(result: pd.DataFrame):
    """Print feature coverage statistics."""
    groups = [
        ("SHARP keywords (16)", SHARP_FEATURES),
        ("Magnetogram-derived (5)", MAGNETO_FEATURES),
        ("Neutral line (3)", NEUTRAL_LINE_FEATURES),
        ("AIA 1600 A (3)", AIA_1600_FEATURES),
        ("GOES X-ray (3)", GOES_FEATURES),
        ("Flare history (4)", FLARE_HISTORY_FEATURES),
        ("24h derivatives (26)", DT24_COLS),
        ("12h derivatives (3)", DT12_COLS),
        ("2h derivatives (2)", DT02_COLS),
        ("AIA 131 A t0 (3)", AIA_131_T0),
        ("AIA 131 A t0-1h (3)", AIA_131_1H),
        ("AIA 131 A t0-2h (3)", AIA_131_2H),
        ("GOES offsets (2)", GOES_OFFSET),
        ("24h deriv 131 A (3)", DT24_131),
    ]
    for name, cols in groups:
        in_result = [c for c in cols if c in result.columns]
        if in_result:
            nonzero = (result[in_result] != 0).any(axis=1).sum()
            pct = 100 * nonzero / len(result) if len(result) > 0 else 0
            print(f"  {name}: {nonzero}/{len(result)} windows non-zero ({pct:.1f}%)")
        else:
            print(f"  {name}: NOT PRESENT in output")


def _build_harp_flare_index(
    windows: pd.DataFrame,
) -> dict[int, list[pd.Timestamp]]:
    """
    Build a per-HARP index of approximate flare times from window labels.
    For each window where y_geq_M_24h is True, treat t0 as an approximate
    flare time. This captures the signal that a >=M-class flare occurred
    within 24h of that issuance, providing flare history even when the
    raw flare catalog has wrong AR associations.
    """
    index: dict[int, list[pd.Timestamp]] = {}
    label_col = "y_geq_M_24h"
    if label_col not in windows.columns:
        return index

    flaring = windows[windows[label_col] == True]  # noqa: E712
    for _, row in flaring.iterrows():
        h = int(row["harpnum"])
        t = pd.Timestamp(row["t0"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        index.setdefault(h, []).append(t)

    for h in index:
        index[h] = sorted(set(index[h]))
    return index


def _safe_load(path: Path, name: str) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  {name}: {len(df)} records")
        return df
    print(f"  {name}: NOT FOUND at {path} (will use zeros)")
    return pd.DataFrame()


def _load_flare_catalog() -> pd.DataFrame:
    """Load flare events for history computation."""
    for path in [
        project_root / "data/defn/flares_cmx.parquet",
        project_root / "data/flares_hek.parquet",
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            if "letter" not in df.columns and "class" in df.columns:
                df["letter"] = df["class"].str[0].str.upper()
            if "start" in df.columns:
                df["start"] = pd.to_datetime(df["start"], utc=True)
            return df

    chunks_dir = project_root / "data/interim/hek_chunks"
    if chunks_dir.exists():
        dfs = []
        for f in sorted(chunks_dir.glob("*.parquet")):
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                continue
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if "letter" not in df.columns and "class" in df.columns:
                df["letter"] = df["class"].str[0].str.upper()
            return df

    print("  WARNING: No flare catalog found.")
    return pd.DataFrame()


if __name__ == "__main__":
    main()
