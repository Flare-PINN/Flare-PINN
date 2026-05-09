import math
import yaml
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

RSUN_Mm = 696.34  # solar radius in megameters


class PathConfig(BaseModel):
    """Configuration for data paths."""
    raw_dir: str = Field(description="Raw data directory")
    interim_dir: str = Field(description="Interim data directory")
    processed_dir: str = Field(description="Processed data directory")
    cache_dir: str = Field(description="Cache directory")
    logs_dir: str = Field(description="Logs directory")


class SpanConfig(BaseModel):
    """Configuration for time span."""
    start: str = Field(description="Start date (YYYY-MM-DD)")
    end: str = Field(description="End date (YYYY-MM-DD)")


class CensoringConfig(BaseModel):
    """Configuration for data censoring."""
    cmd_deg_threshold: float = Field(default=70.0, description="CMD threshold in degrees")


class ResizeConfig(BaseModel):
    """Configuration for image resizing."""
    target_px: int = Field(default=256, description="Target pixel size")


class DataConfig(BaseModel):
    """Main data configuration."""
    span: SpanConfig
    cadence_minutes: int = Field(default=60, description="Data cadence in minutes")
    window_hours: int = Field(default=48, description="Window size in hours")
    window_stride_hours: int = Field(default=6, description="Window stride in hours")
    targets_hours: list[int] = Field(default=[6, 12, 24], description="Target horizons in hours")
    censoring: CensoringConfig
    resize: ResizeConfig
    paths: PathConfig


def load_cfg(path: Path) -> DataConfig:
    """Load YAML configuration file and parse into typed config model."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DataConfig(**data)

def ensure_dirs(cfg: DataConfig) -> None:
    """Create necessary directories from config."""
    for k in ["raw_dir", "interim_dir", "processed_dir", "cache_dir", "logs_dir"]:
        Path(getattr(cfg.paths, k)).mkdir(parents=True, exist_ok=True)

def pixel_scale_Mm_per_pix_from_header(hdr: dict) -> float:
    """
    SHARP CEA: CDELT1 (deg/pix). Convert deg->Mm at photosphere.
    Returns Mm/pixel.
    """
    cdelt1 = float(hdr.get("CDELT1", "nan"))
    if math.isnan(cdelt1):
        return float("nan")
    return (cdelt1 * math.pi / 180.0) * RSUN_Mm

def cmd_abs_deg_from_header(hdr: dict) -> float:
    """
    Approx |CMD| (deg) for patch center using CRLN_OBS and a center-long key.
    """
    cln_obs = hdr.get("CRLN_OBS")
    lon_center = None
    for k in ["LON_FWT", "CRLN", "LON_CEA", "CRLN_CNTR", "HGLOBS"]:
        if k in hdr: 
            lon_center = hdr[k]
            break
    if cln_obs is None or lon_center is None:
        return float("nan")
    cmd = (float(lon_center) - float(cln_obs)) % 360.0
    if cmd > 180.0: 
        cmd -= 360.0
    return abs(cmd)

def finite_diff_divergence(Bx: np.ndarray, By: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    """Compute divergence using finite differences."""
    dBx_dx = np.gradient(Bx, axis=1) / dx
    dBy_dy = np.gradient(By, axis=0) / dy
    return dBx_dx + dBy_dy

def to_local_fields(Br: np.ndarray, Bt: np.ndarray, Bp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert SHARP CEA coordinates to local field components.
    
    Plan's convention:
      Bz = Br
      Bx = -Bp  
      By = -Bt
    """
    Bz = Br
    Bx = -Bp
    By = -Bt
    return Bx, By, Bz

def validate_divergence(Bx: np.ndarray, By: np.ndarray, threshold: float = 1e-3) -> bool:
    """
    Validate that magnetic field is approximately divergence-free.
    
    Args:
        Bx, By: Magnetic field components
        threshold: Maximum allowed divergence magnitude
        
    Returns:
        True if divergence is below threshold
    """
    div = finite_diff_divergence(Bx, By)
    max_div = np.nanmax(np.abs(div))
    return max_div < threshold

class NormalizationStats(BaseModel):
    """Statistics from robust normalization."""
    median: float
    q25: float
    q75: float
    iqr: float


def robust_normalize(data: np.ndarray, clip_range: Tuple[float, float] = (-5.0, 5.0)) -> Tuple[np.ndarray, NormalizationStats]:
    """
    Robust normalization using median and IQR.
    
    Args:
        data: Input array
        clip_range: Range to clip normalized data
        
    Returns:
        Normalized data and statistics dictionary
    """
    median = np.nanmedian(data)
    q75, q25 = np.nanpercentile(data, [75, 25])
    iqr = q75 - q25
    
    if iqr == 0:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - median) / iqr
    
    # Clip outliers
    normalized = np.clip(normalized, clip_range[0], clip_range[1])
    
    stats = NormalizationStats(
        median=float(median),
        q25=float(q25),
        q75=float(q75),
        iqr=float(iqr)
    )
    
    return normalized, stats
