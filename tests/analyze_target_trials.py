#!/usr/bin/env python3
"""Analyze RS1 duplicate-detection trial-recorder CSV; standard library only."""
import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--trial', required=True, choices=['empty', 'ascent', 'spaced', 'far_single', 'far_pair'])
    args = parser.parse_args()
    with args.csv_file.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'File: {args.csv_file}')
    print(f'Trial: {args.trial}; confirmed rescue alerts: {len(rows)}')
    for i, r in enumerate(rows, 1):
        print(f"  {i}: ID={r['target_id']} t={r['alert_sim_time_sec']} "
              f"position=({r['target_x_m']},{r['target_y_m']}) "
              f"drone_z={r['drone_z_m'] or '?'} "
              f"({r['altitude_observation']})")
    if args.trial == 'empty':
        print('PASS: no alerts in no-human world' if not rows
              else 'FAIL: alerts with NO humans. Investigate false positives.')
    elif args.trial in ('ascent', 'far_single'):
        if len(rows) == 0:
            print('INCONCLUSIVE: no detection; confirm the human entered the camera view.')
        elif len(rows) == 1:
            print('PASS for this run: exactly one alert for the single human.')
        else:
            print('FAIL: >1 alert in single-human world; repeated localisation or false positives.')
    else:
        print('PASS count only: exactly two alerts' if len(rows) == 2
              else 'CHECK: expected two confirmed alerts; investigate coverage/false positives.')
    if len(rows) > 1:
        print('Alert separations in reported frame (XY):')
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                try:
                    a, b = rows[i], rows[j]
                    d = math.hypot(float(a['target_x_m']) - float(b['target_x_m']),
                                   float(a['target_y_m']) - float(b['target_y_m']))
                except (ValueError, KeyError):
                    continue
                print(f"  #{a['target_id']} vs #{b['target_id']}: {d:.2f} m"
                      + (' (within original dedup radius)' if d < 1.9 else ''))
    print('Important: counts alone do not verify identity or location accuracy.')
    print('Targets use parrot1_odom; world SDF positions need frame alignment before numerical comparison.')


if __name__ == '__main__':
    main()
