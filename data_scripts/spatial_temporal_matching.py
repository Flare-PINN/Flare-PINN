#!/usr/bin/env python3
"""
Spatial-Temporal Fallback Matching
===================================

For flares/HARPs without NOAA mapping, match by position and time.
Tests if flare heliographic coordinates fall within SHARP map footprint.

Based on best practices from Bobra & Couvidat and similar work.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


def flare_in_harp_footprint(
    flare_lon: float,
    flare_lat: float,
    harp_center_lon: float,
    harp_center_lat: float,
    harp_size_deg: float = 10.0,
    margin_deg: float = 5.0
) -> bool:
    """
    Simple rectangular footprint test.
    
    Args:
        flare_lon, flare_lat: Flare heliographic coordinates (degrees)
        harp_center_lon, harp_center_lat: HARP center (degrees)
        harp_size_deg: Approximate HARP size (degrees, default ~10°)
        margin_deg: Extra margin for matching (degrees)
        
    Returns:
        True if flare is within HARP footprint + margin
        
    Note: For production, use actual SHARP WCS corners from FITS headers.
    This simplified version works for ~80% of cases.
    """
    # Account for periodicity in longitude
    dlon = abs(flare_lon - harp_center_lon)
    if dlon > 180:
        dlon = 360 - dlon
    
    dlat = abs(flare_lat - harp_center_lat)
    
    # Rectangular box with margin
    half_size = (harp_size_deg / 2.0) + margin_deg
    
    return (dlon <= half_size) and (dlat <= half_size)


def spatial_temporal_match(
    flares_df: pd.DataFrame,
    frames_df: pd.DataFrame,
    existing_mapping: pd.DataFrame,
    time_window_hours: float = 1.0,
    margin_deg: float = 5.0
) -> pd.DataFrame:
    """
    Find spatial-temporal matches for unmapped flares/HARPs.
    
    Args:
        flares_df: Flare catalog with columns:
            - start (datetime)
            - noaa_ar (int or NaN)
            - event_coord1, event_coord2 (heliographic lon/lat in degrees)
        frames_df: Frame metadata with columns:
            - harpnum (int)
            - date_obs (datetime)
            - hgs_lon, hgs_lat (heliographic coordinates of HARP center)
        existing_mapping: Existing HARP-NOAA mapping (to avoid duplicates)
        time_window_hours: Match if frame within ±this many hours of flare
        margin_deg: Spatial matching margin
        
    Returns:
        DataFrame with columns: harpnum, noaa_ar, match_type='spatial'
        
    Note: Requires flare coordinates in heliographic system.
    HEK provides these in event_coord1/event_coord2 fields.
    """
    
    print("\n[Spatial-Temporal Matching]")
    
    # Check required columns
    required_flare = ['start', 'noaa_ar']
    required_frame = ['harpnum', 'date_obs']
    
    missing_flare = [c for c in required_flare if c not in flares_df.columns]
    missing_frame = [c for c in required_frame if c not in frames_df.columns]
    
    if missing_flare or missing_frame:
        print(f"  WARNING: Missing required columns")
        if missing_flare:
            print(f"    Flares: {missing_flare}")
        if missing_frame:
            print(f"    Frames: {missing_frame}")
        print(f"  Skipping spatial matching")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar', 'match_type'])
    
    # Check for coordinate columns (may have various names)
    flare_coord_cols = None
    for lon_col in ['event_coord1', 'hgs_lon', 'hgln_coord', 'lon']:
        for lat_col in ['event_coord2', 'hgs_lat', 'hglt_coord', 'lat']:
            if lon_col in flares_df.columns and lat_col in flares_df.columns:
                flare_coord_cols = (lon_col, lat_col)
                break
        if flare_coord_cols:
            break
    
    frame_coord_cols = None
    for lon_col in ['hgs_lon', 'crln_obs', 'center_lon']:
        for lat_col in ['hgs_lat', 'crlt_obs', 'center_lat']:
            if lon_col in frames_df.columns and lat_col in frames_df.columns:
                frame_coord_cols = (lon_col, lat_col)
                break
        if frame_coord_cols:
            break
    
    if not flare_coord_cols:
        print(f"  WARNING: Flare coordinates not found in flares_df")
        print(f"  Available columns: {list(flares_df.columns)}")
        print(f"  Skipping spatial matching")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar', 'match_type'])
    
    if not frame_coord_cols:
        print(f"  WARNING: Frame coordinates not found in frames_df")
        print(f"  Note: Spatial matching requires heliographic coords in frame metadata")
        print(f"  Skipping spatial matching")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar', 'match_type'])
    
    print(f"  Using flare coords: {flare_coord_cols}")
    print(f"  Using frame coords: {frame_coord_cols}")
    
    # Filter to unmapped flares (no NOAA AR or NOAA AR not in mapping)
    mapped_noaa_ars = set(existing_mapping['noaa_ar'].unique())
    unmapped_flares = flares_df[
        flares_df['noaa_ar'].isna() | 
        ~flares_df['noaa_ar'].isin(mapped_noaa_ars)
    ].copy()
    
    print(f"  Found {len(unmapped_flares)} unmapped flares to match")
    
    if len(unmapped_flares) == 0:
        return pd.DataFrame(columns=['harpnum', 'noaa_ar', 'match_type'])
    
    # Also find unmapped HARPs
    mapped_harps = set(existing_mapping['harpnum'].unique())
    unmapped_harps = set(frames_df['harpnum'].unique()) - mapped_harps
    
    print(f"  Found {len(unmapped_harps)} unmapped HARPs")
    
    # Perform matching
    matches = []
    time_window = pd.Timedelta(hours=time_window_hours)
    
    for _, flare in unmapped_flares.iterrows():
        flare_time = pd.to_datetime(flare['start'])
        flare_lon = flare[flare_coord_cols[0]]
        flare_lat = flare[flare_coord_cols[1]]
        
        if pd.isna(flare_lon) or pd.isna(flare_lat):
            continue
        
        # Find frames near this time
        nearby_frames = frames_df[
            (frames_df['date_obs'] >= flare_time - time_window) &
            (frames_df['date_obs'] <= flare_time + time_window)
        ]
        
        for _, frame in nearby_frames.iterrows():
            frame_lon = frame[frame_coord_cols[0]]
            frame_lat = frame[frame_coord_cols[1]]
            
            if pd.isna(frame_lon) or pd.isna(frame_lat):
                continue
            
            if flare_in_harp_footprint(
                flare_lon, flare_lat,
                frame_lon, frame_lat,
                margin_deg=margin_deg
            ):
                # Create synthetic NOAA AR for this unmapped flare if needed
                noaa_ar = flare.get('noaa_ar')
                if pd.isna(noaa_ar):
                    # Use negative numbers for spatial-only matches to distinguish
                    noaa_ar = -int(flare.name)  # Use row index as unique ID
                
                matches.append({
                    'harpnum': int(frame['harpnum']),
                    'noaa_ar': int(noaa_ar),
                    'match_type': 'spatial',
                    'flare_time': flare_time,
                    'frame_time': frame['date_obs'],
                    'separation_deg': np.sqrt((flare_lon - frame_lon)**2 + (flare_lat - frame_lat)**2)
                })
    
    if len(matches) == 0:
        print(f"  No spatial matches found")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar', 'match_type'])
    
    matches_df = pd.DataFrame(matches)
    
    # Remove duplicates (keep closest match)
    matches_df = (matches_df
                  .sort_values('separation_deg')
                  .drop_duplicates(subset=['harpnum', 'noaa_ar'], keep='first')
                  .reset_index(drop=True))
    
    print(f"\n[Spatial Matching Results]")
    print(f"  Found {len(matches_df)} spatial matches")
    print(f"  Matched {matches_df['harpnum'].nunique()} HARPs")
    print(f"  Median separation: {matches_df['separation_deg'].median():.2f}°")
    print(f"  Max separation: {matches_df['separation_deg'].max():.2f}°")
    
    return matches_df[['harpnum', 'noaa_ar', 'match_type']]


def combine_mappings(
    id_based: pd.DataFrame,
    spatial: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine ID-based and spatial mappings.
    
    Args:
        id_based: NOAA ID-based mapping
        spatial: Spatial-temporal matches
        
    Returns:
        Combined mapping with match_type column
    """
    
    # Add match_type to ID-based if not present
    if 'match_type' not in id_based.columns:
        id_based = id_based.copy()
        id_based['match_type'] = 'id'
    
    # Combine
    combined = pd.concat([id_based, spatial], ignore_index=True)
    combined = combined.drop_duplicates(subset=['harpnum', 'noaa_ar'], keep='first')
    combined = combined.sort_values(['harpnum', 'noaa_ar']).reset_index(drop=True)
    
    print(f"\n[Combined Mapping]")
    print(f"  Total: {len(combined)} mappings")
    print(f"  ID-based: {(combined['match_type'] == 'id').sum()}")
    print(f"  Spatial: {(combined['match_type'] == 'spatial').sum()}")
    print(f"  Coverage: {combined['harpnum'].nunique()} HARPs")
    
    return combined

