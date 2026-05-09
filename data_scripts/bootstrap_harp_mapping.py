#!/usr/bin/env python3
"""
Bootstrap HARP-NOAA Mapping with JSOC Public List
==================================================

Downloads JSOC's canonical all_harps_with_noaa_ars.txt and unions it
with your existing mapping for maximum coverage.

Based on: https://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/
"""

import re
import pandas as pd
import urllib.request
from pathlib import Path
import click


JSOC_PUBLIC_LIST = "https://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"


def download_jsoc_mapping() -> pd.DataFrame:
    """Download and parse JSOC's public HARP-NOAA mapping list."""
    
    print("[bootstrap_harp_mapping]")
    print(f"  Downloading JSOC public mapping from:")
    print(f"  {JSOC_PUBLIC_LIST}")
    
    try:
        # Try with SSL context that's more permissive
        import ssl
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(JSOC_PUBLIC_LIST, timeout=30, context=context) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"  ERROR: Could not download: {e}")
        print(f"\n  You can manually download from:")
        print(f"  {JSOC_PUBLIC_LIST}")
        print(f"  and save as: data/all_harps_with_noaa_ars.txt")
        return pd.DataFrame(columns=['harpnum', 'noaa_ar'])
    
    # Parse the file
    # Format: HARPNUM   NOAA_ARS (comma-separated)
    rows = []
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 2:
            continue
            
        try:
            harpnum = int(parts[0])
            # NOAA_ARS can be comma-separated like "11234,11235"
            noaa_ars_str = parts[1]
            
            # Extract all numbers from the NOAA_ARS field
            noaa_ars = [int(n) for n in re.findall(r'\d+', noaa_ars_str) if int(n) > 0]
            
            for noaa_ar in noaa_ars:
                rows.append({'harpnum': harpnum, 'noaa_ar': noaa_ar})
                
        except (ValueError, IndexError):
            continue
    
    df = pd.DataFrame(rows)
    
    if len(df) > 0:
        df = df.drop_duplicates().sort_values(['harpnum', 'noaa_ar']).reset_index(drop=True)
    
    print(f"  Parsed {len(df)} mappings from JSOC public list")
    print(f"  Covers {df['harpnum'].nunique()} HARPs, {df['noaa_ar'].nunique()} NOAA ARs")
    
    return df


def merge_mappings(existing_path: Path, jsoc_df: pd.DataFrame) -> pd.DataFrame:
    """Merge existing mapping with JSOC public list."""
    
    print(f"\n  Loading existing mapping from: {existing_path}")
    
    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        print(f"  Existing: {len(existing_df)} mappings ({existing_df['harpnum'].nunique()} HARPs)")
        
        # Union: keep all unique (harpnum, noaa_ar) pairs
        combined = pd.concat([existing_df, jsoc_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['harpnum', 'noaa_ar'])
        combined = combined.sort_values(['harpnum', 'noaa_ar']).reset_index(drop=True)
        
        # Stats
        new_harps = set(jsoc_df['harpnum']) - set(existing_df['harpnum'])
        new_mappings = len(combined) - len(existing_df)
        
        print(f"\n[Results]")
        print(f"  Combined: {len(combined)} mappings ({combined['harpnum'].nunique()} HARPs)")
        print(f"  Added {new_mappings} new mappings")
        print(f"  Added {len(new_harps)} new HARPs from JSOC list")
        
        return combined
        
    else:
        print(f"  No existing mapping found - using JSOC list only")
        return jsoc_df


@click.command()
@click.option('--existing', default='data/harp_noaa_mapping.parquet',
              help='Path to existing mapping (will be merged)')
@click.option('--output', default='data/harp_noaa_mapping_merged.parquet',
              help='Path for merged output')
@click.option('--replace', is_flag=True,
              help='Replace existing file instead of creating new one')
def main(existing, output, replace):
    """Bootstrap HARP-NOAA mapping with JSOC public list."""
    
    existing_path = Path(existing)
    output_path = Path(output)
    
    # Download JSOC public list
    jsoc_df = download_jsoc_mapping()
    
    if jsoc_df.empty:
        print("\nERROR: Could not download JSOC mapping")
        return
    
    # Merge with existing
    merged_df = merge_mappings(existing_path, jsoc_df)
    
    # Save
    if replace:
        output_path = existing_path
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    
    print(f"\n✓ Saved merged mapping: {output_path}")
    print(f"  {len(merged_df)} total mappings")
    print(f"  {merged_df['harpnum'].nunique()} unique HARPs")
    print(f"  {merged_df['noaa_ar'].nunique()} unique NOAA ARs")
    
    # Show some examples
    print(f"\n  Sample mappings (first 10):")
    print(merged_df.head(10).to_string(index=False))
    
    # HARPs with multiple NOAA ARs (normal - ARs evolve)
    multi = merged_df.groupby('harpnum').size()
    multi = multi[multi > 1]
    if len(multi) > 0:
        print(f"\n  {len(multi)} HARPs have multiple NOAA ARs (expected - regions evolve)")
        print(f"  Example: HARP {multi.index[0]} has {multi.iloc[0]} NOAA ARs")


if __name__ == '__main__':
    main()

