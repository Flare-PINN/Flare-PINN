#!/usr/bin/env python3
"""
Fetch HARP to NOAA AR Mapping from JSOC
========================================

Downloads the mapping between HARP numbers (SDO/HMI) and NOAA AR numbers
from the JSOC SHARP metadata.

Output: data/harp_noaa_mapping.parquet
"""

import click
import pandas as pd
import drms
from pathlib import Path
from tqdm import tqdm

from src.utils.common import load_cfg


def fetch_harp_noaa_mapping_from_frames(frames_path: str, output_path: Path):
    """
    Query JSOC for HARP-NOAA mapping by getting metadata for each HARP in our dataset.
    
    More reliable than date-based queries.
    """
    print(f"[fetch_harp_noaa_mapping]")
    print(f"  Loading frame metadata to get HARP list...")
    
    # Load frames to get list of HARPs
    frames = pd.read_parquet(frames_path)
    unique_harps = sorted(frames['harpnum'].unique())
    
    print(f"  Found {len(unique_harps)} unique HARPs in dataset")
    print(f"  HARP range: {unique_harps[0]} to {unique_harps[-1]}")
    print(f"\n  Querying JSOC for NOAA AR mappings...")
    print(f"  This will take a few minutes...\n")
    
    # Initialize DRMS client
    client = drms.Client()
    
    # Query each HARP's metadata (in batches to avoid timeout)
    all_mappings = []
    batch_size = 50
    
    for i in tqdm(range(0, len(unique_harps), batch_size), desc="Querying JSOC"):
        batch = unique_harps[i:i+batch_size]
        
        for harp in batch:
            try:
                # Query for this specific HARP
                # Use hmi.sharp_cea_720s series which has NOAA_AR keyword
                query = f'hmi.sharp_cea_720s[][{harp}]'
                keys = client.query(query, key='HARPNUM, NOAA_AR, T_REC', n=1)
                
                if len(keys) > 0:
                    # Get the NOAA_AR value
                    noaa_ar = keys['NOAA_AR'].iloc[0] if 'NOAA_AR' in keys.columns else 0
                    if noaa_ar > 0:
                        all_mappings.append({'harpnum': harp, 'noaa_ar': int(noaa_ar)})
            except Exception as e:
                # Skip HARPs that don't have NOAA AR or fail to query
                continue
    
    if len(all_mappings) == 0:
        print("    No HARP-NOAA mappings found!")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar'])
    
    df_mapping = pd.DataFrame(all_mappings)
    
    print(f"\n  ✓ Found {len(df_mapping)} HARP-NOAA mappings")
    
    return df_mapping


def fetch_harp_noaa_mapping(start_date: str, end_date: str, output_path: Path):
    """
    Wrapper that calls the frames-based approach.
    """
    print(f"[fetch_harp_noaa_mapping]")
    print(f"  Date range: {start_date} to {end_date} (will query by HARP numbers instead)\n")
    
    # Use frames metadata to get HARP list
    frames_path = 'S:/flare_forecasting/frames_meta.parquet'
    
    try:
        mapping = fetch_harp_noaa_mapping_from_frames(frames_path, output_path)
        
        if len(mapping) == 0:
            print("    No HARP-NOAA mappings found!")
            return pd.DataFrame(columns=['harpnum', 'noaa_ar'])
        
        # Save mapping
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mapping.to_parquet(output_path, index=False)
        print(f"  ✓ Saved mapping: {output_path}")
        
        # Show some examples
        print(f"\n  Sample mappings:")
        print(mapping.head(20).to_string(index=False))
        
        return mapping
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['harpnum', 'noaa_ar'])


@click.command()
@click.option("--frames", required=True, help="Path to frames_meta.parquet")
@click.option("--output", default="data/harp_noaa_mapping_sc25.parquet", help="Output parquet file")
@click.option("--merge-existing", default=None, help="Path to existing mapping to merge with")
def main(frames, output, merge_existing):
    """Fetch HARP to NOAA AR mapping from JSOC."""
    
    output_path = Path(output)
    
    # Fetch new mapping from JSOC
    mapping = fetch_harp_noaa_mapping_from_frames(frames, output_path)
    
    if len(mapping) == 0:
        print(f"\n✗ Failed to fetch mapping")
        return
    
    # Merge with existing mapping if provided
    if merge_existing and Path(merge_existing).exists():
        existing = pd.read_parquet(merge_existing)
        print(f"\n  Merging with existing mapping: {len(existing)} entries")
        mapping = pd.concat([existing, mapping], ignore_index=True)
        mapping = mapping.drop_duplicates(subset=['harpnum'], keep='last')
        print(f"  After merge: {len(mapping)} entries")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)
    
    print(f"\n✓ Done! Mapping saved to {output_path}")
    print(f"  {len(mapping)} HARPs mapped to NOAA AR numbers")


if __name__ == "__main__":
    main()

