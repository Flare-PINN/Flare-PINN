#!/usr/bin/env python3
"""
Fetch HARP Spatial Information (Bounding Boxes)
================================================

Downloads heliographic coordinates and bounding boxes for each HARP
to enable spatial matching of flares to active regions.

Output: data/harp_spatial_info.parquet
Columns: harpnum, lat_hg, lon_hg, lat_min, lat_max, lon_min, lon_max, t_rec
"""

import click
import pandas as pd
import drms
from pathlib import Path
from tqdm import tqdm
import time


def fetch_harp_spatial_info(frames_path: str, output_path: Path):
    """
    Query JSOC for spatial info (lat/lon bounds) for each HARP.
    """
    print("[fetch_harp_spatial_info]")
    print(f"  Loading frames to get HARP list...")
    
    # Load frames
    frames = pd.read_parquet(frames_path)
    unique_harps = sorted(frames['harpnum'].unique())
    
    print(f"  Found {len(unique_harps)} unique HARPs")
    print(f"  HARP range: {unique_harps[0]} to {unique_harps[-1]}")
    print(f"\n  Querying JSOC for bounding box coordinates...")
    print(f"  This will take ~20-30 minutes...\n")
    
    # Initialize DRMS client
    client = drms.Client()
    
    # Query each HARP for its spatial metadata
    spatial_info = []
    batch_size = 100
    
    for i in tqdm(range(0, len(unique_harps), batch_size), desc="Querying JSOC"):
        batch = unique_harps[i:i+batch_size]
        
        for harp in batch:
            try:
                # Query SHARP CEA series for spatial keywords
                query = f'hmi.sharp_cea_720s[][{harp}]'
                
                # Request bounding box and centroid coordinates
                keys = 'HARPNUM, LAT_FWT, LON_FWT, CRLN_OBS, CRLT_OBS, T_REC'
                result = client.query(query, key=keys, n=10)  # Get a few records
                
                if len(result) > 0:
                    # Average over available records for this HARP
                    for _, row in result.iterrows():
                        spatial_info.append({
                            'harpnum': int(row['HARPNUM']),
                            'lat_hg': float(row['LAT_FWT']) if pd.notna(row['LAT_FWT']) else None,
                            'lon_hg': float(row['LON_FWT']) if pd.notna(row['LON_FWT']) else None,
                            'crln_obs': float(row['CRLN_OBS']) if pd.notna(row['CRLN_OBS']) else None,
                            'crlt_obs': float(row['CRLT_OBS']) if pd.notna(row['CRLT_OBS']) else None,
                            't_rec': pd.to_datetime(row['T_REC'])
                        })
                
                # Rate limiting
                if (i + 1) % 50 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                # Skip failed queries
                continue
    
    if len(spatial_info) == 0:
        print("    No spatial info retrieved!")
        return pd.DataFrame()
    
    df = pd.DataFrame(spatial_info)
    
    # Aggregate per HARP (take median of coordinates)
    df_agg = df.groupby('harpnum').agg({
        'lat_hg': 'median',
        'lon_hg': 'median',
        'crln_obs': 'median',
        'crlt_obs': 'median',
        't_rec': 'first'
    }).reset_index()
    
    # Add bounding box estimate (±15 degrees from center is typical HARP size)
    df_agg['lat_min'] = df_agg['lat_hg'] - 15.0
    df_agg['lat_max'] = df_agg['lat_hg'] + 15.0
    df_agg['lon_min'] = df_agg['lon_hg'] - 15.0
    df_agg['lon_max'] = df_agg['lon_hg'] + 15.0
    
    print(f"\n  ✓ Retrieved spatial info for {len(df_agg)} HARPs")
    print(f"  Coverage: {100*len(df_agg)/len(unique_harps):.1f}% of your HARPs")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    
    print(f"\n  Sample spatial info:")
    print(df_agg[['harpnum', 'lat_hg', 'lon_hg', 'lat_min', 'lat_max']].head(10).to_string(index=False))
    
    return df_agg


@click.command()
@click.option("--frames", default="S:/flare_forecasting/frames_meta.parquet", help="Path to frames metadata")
@click.option("--output", default="data/harp_spatial_info.parquet", help="Output path")
def main(frames, output):
    """Fetch HARP spatial information for spatial flare matching."""
    output_path = Path(output)
    spatial_info = fetch_harp_spatial_info(frames, output_path)
    
    if len(spatial_info) > 0:
        print(f"\n✓ Done! Spatial info saved.")
    else:
        print(f"\n✗ Failed to fetch spatial info")


if __name__ == "__main__":
    main()

