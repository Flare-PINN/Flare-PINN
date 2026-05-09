#!/usr/bin/env python3
"""
Event-Based Evaluation for Flare Forecasting

Computes operational metrics from window-level predictions:
1. Event detection rate (per-flare)
2. Median warning lead time 
3. False-alert rate (alerts per day)

Usage:
    python compute_event_metrics.py --npz outputs/.../test.npz --data data/windows_test_15.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta


def load_data(npz_path: Path, windows_path: Path, horizon_idx: int):
    """Load predictions and metadata."""
    npz = np.load(npz_path)
    windows = pd.read_parquet(windows_path)
    
    probs = npz['probs'][:, horizon_idx]
    labels = npz['labels'][:, horizon_idx]
    threshold = npz['thresholds'][horizon_idx]
    
    # Add predictions to windows
    windows = windows.copy()
    windows['prob'] = probs
    windows['pred'] = (probs >= threshold).astype(int)
    windows['label'] = labels
    windows['t0'] = pd.to_datetime(windows['t0'])
    
    return windows, threshold


def compute_event_detection_rate(windows: pd.DataFrame, horizon_hours: int):
    """
    For each flare event, check if it was predicted by ANY window in the forecast period.
    
    Returns:
        detection_rate: fraction of events detected
        n_events: total number of events
        n_detected: number detected
    """
    # Get actual flare events (unique t_peak times with positive labels)
    # A window is positive if a flare occurs within horizon_hours after t0
    # So flare at t_peak is labeled in windows with t0 in [t_peak - horizon_hours, t_peak]
    
    positive_windows = windows[windows['label'] == 1].copy()
    
    if len(positive_windows) == 0:
        return 0.0, 0, 0, []
    
    # Group by HARP to find flare events
    # We need to reconstruct flare times from positive windows
    # If a window at t0 is positive for horizon h, a flare occurs in [t0, t0+h]
    
    events = []
    for harp, harp_windows in positive_windows.groupby('harpnum'):
        harp_windows = harp_windows.sort_values('t0')
        
        # Find gaps > horizon_hours (indicates separate events)
        times = harp_windows['t0'].values
        time_diffs = np.diff(times.astype('datetime64[h]').astype(int))
        
        # Start new event if gap > horizon
        event_groups = np.concatenate([[0], np.cumsum(time_diffs > horizon_hours)])
        
        for event_id in np.unique(event_groups):
            event_windows = harp_windows.iloc[event_groups == event_id]
            
            # Event time = earliest possible flare time = min(t0) + small buffer
            # But we want to see if we PREDICTED before the event
            # Let's use the last positive window's t0 as proxy for event time
            event_time = event_windows['t0'].max()
            
            # Was it predicted? Check windows in forecast period [event_time - horizon, event_time]
            forecast_start = event_time - pd.Timedelta(hours=horizon_hours)
            forecast_windows = windows[
                (windows['harpnum'] == harp) &
                (windows['t0'] >= forecast_start) &
                (windows['t0'] <= event_time)
            ]
            
            detected = (forecast_windows['pred'] == 1).any()
            
            # Lead time: time from first alert to event
            if detected:
                first_alert = forecast_windows[forecast_windows['pred'] == 1]['t0'].min()
                lead_time_hours = (event_time - first_alert).total_seconds() / 3600
            else:
                lead_time_hours = None
            
            events.append({
                'harpnum': harp,
                'event_time': event_time,
                'detected': detected,
                'lead_time_hours': lead_time_hours,
            })
    
    events_df = pd.DataFrame(events)
    n_events = len(events_df)
    n_detected = events_df['detected'].sum()
    detection_rate = n_detected / n_events if n_events > 0 else 0.0
    
    lead_times = events_df[events_df['detected']]['lead_time_hours'].values
    
    return detection_rate, n_events, n_detected, lead_times


def compute_false_alert_rate(windows: pd.DataFrame):
    """
    Compute false alerts per day.
    
    An alert day = any day with ≥1 predicted positive window.
    A false-alert day = alert day with no actual flare event.
    """
    # Add date column
    windows = windows.copy()
    windows['date'] = windows['t0'].dt.date
    
    # Group by date
    daily = windows.groupby('date').agg({
        'pred': 'max',  # 1 if any alert that day
        'label': 'max',  # 1 if any flare that day
    })
    
    n_alert_days = (daily['pred'] == 1).sum()
    n_correct_alert_days = ((daily['pred'] == 1) & (daily['label'] == 1)).sum()
    n_false_alert_days = ((daily['pred'] == 1) & (daily['label'] == 0)).sum()
    
    n_days = len(daily)
    false_alert_rate = n_false_alert_days / n_days
    
    return {
        'total_days': n_days,
        'alert_days': n_alert_days,
        'correct_alert_days': n_correct_alert_days,
        'false_alert_days': n_false_alert_days,
        'false_alert_rate': false_alert_rate,
        'alerts_per_day': n_alert_days / n_days,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute event-based metrics')
    parser.add_argument('--npz', type=str, required=True, help='Path to test predictions NPZ')
    parser.add_argument('--data', type=str, required=True, help='Path to test windows parquet')
    parser.add_argument('--output', type=str, help='Output CSV path')
    args = parser.parse_args()
    
    npz_path = Path(args.npz)
    windows_path = Path(args.data)
    
    print("="*80)
    print("EVENT-BASED EVALUATION")
    print("="*80)
    print(f"Predictions: {npz_path.name}")
    print()
    
    results = []
    
    for horizon_idx, horizon_name, horizon_hours in [(0, '6h', 6), (1, '12h', 12), (2, '24h', 24)]:
        print(f"\n{'='*80}")
        print(f"{horizon_name} HORIZON")
        print('='*80)
        
        windows, threshold = load_data(npz_path, windows_path, horizon_idx)
        
        print(f"Threshold: {threshold:.4f}")
        print(f"Total windows: {len(windows)}")
        print(f"Positive windows: {(windows['label'] == 1).sum()}")
        print()
        
        # Event detection
        detection_rate, n_events, n_detected, lead_times = compute_event_detection_rate(
            windows, horizon_hours
        )
        
        print(f" EVENT DETECTION")
        print(f"  Total flare events: {n_events}")
        print(f"  Events detected: {n_detected}")
        print(f"  Detection rate: {100*detection_rate:.1f}%")
        
        if len(lead_times) > 0:
            print(f"\n⏱️  WARNING LEAD TIMES (detected events only)")
            print(f"  Median: {np.median(lead_times):.1f} hours")
            print(f"  Mean: {np.mean(lead_times):.1f} hours")
            print(f"  Min: {np.min(lead_times):.1f} hours")
            print(f"  Max: {np.max(lead_times):.1f} hours")
        
        # False alerts
        fa_stats = compute_false_alert_rate(windows)
        
        print(f"\n🚨 ALERT STATISTICS")
        print(f"  Total days: {fa_stats['total_days']}")
        print(f"  Alert days: {fa_stats['alert_days']} ({100*fa_stats['alerts_per_day']:.1f}%)")
        print(f"  Correct alert days: {fa_stats['correct_alert_days']}")
        print(f"  False-alert days: {fa_stats['false_alert_days']}")
        print(f"  False-alert rate: {100*fa_stats['false_alert_rate']:.2f}% of days")
        
        results.append({
            'horizon': horizon_name,
            'threshold': threshold,
            'n_events': n_events,
            'n_detected': n_detected,
            'detection_rate': detection_rate,
            'median_lead_time_hours': np.median(lead_times) if len(lead_times) > 0 else None,
            'mean_lead_time_hours': np.mean(lead_times) if len(lead_times) > 0 else None,
            'total_days': fa_stats['total_days'],
            'alert_days': fa_stats['alert_days'],
            'alerts_per_day': fa_stats['alerts_per_day'],
            'false_alert_days': fa_stats['false_alert_days'],
            'false_alert_rate': fa_stats['false_alert_rate'],
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n Results saved to {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()

