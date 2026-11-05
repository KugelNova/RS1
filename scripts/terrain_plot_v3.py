#!/usr/bin/env python3
"""Plot Mapping V3 observed surfaces and qualified surface candidates."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(csv_path):
    csv_path = Path(csv_path).expanduser().resolve()
    with csv_path.open(newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError('Empty mapping CSV')
    x = np.array([float(r['x']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    z = np.array([float(r['observed_z']) for r in rows])
    good = np.array([r['status'] == 'surface_candidate' for r in rows])
    statuses = {}
    for row in rows:
        statuses[row['status']] = statuses.get(row['status'], 0) + 1
    vmin, vmax = float(z.min()), float(z.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax in axes:
        ax.set_facecolor('0.92')
        ax.set(xlabel='X (m)', ylabel='Y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.12)
    axes[0].set_title('All observed surfaces (not necessarily ground)')
    axes[0].scatter(x, y, c=z, cmap='terrain', marker='s', s=17,
                    vmin=vmin, vmax=vmax, linewidths=0)
    axes[1].set_title('Surface candidates only; other cells unknown')
    if good.any():
        axes[1].scatter(x[good], y[good], c=z[good], cmap='terrain',
                        marker='s', s=17, vmin=vmin, vmax=vmax, linewidths=0)
    else:
        axes[1].text(0.5, 0.5, 'No candidate cells yet', transform=axes[1].transAxes,
                     ha='center', va='center')
    scalar = plt.cm.ScalarMappable(cmap='terrain',
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(scalar, ax=axes, shrink=0.8, label='Observed surface height (m)')
    fig.suptitle('RS1 V3 experimental 2.5D surface map — not verified bare earth')
    img2d = csv_path.with_name(csv_path.stem + '_2d.png')
    fig.savefig(img2d, dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    if good.any():
        sc = ax.scatter(x[good], y[good], z[good], c=z[good], cmap='terrain', s=7)
        fig.colorbar(sc, ax=ax, shrink=0.6, label='Candidate elevation (m)')
    else:
        ax.text2D(0.3, 0.5, 'No candidate cells yet', transform=ax.transAxes)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Observed Z (m)',
           title='RS1 V3 candidate surfaces (unverified)')
    img3d = csv_path.with_name(csv_path.stem + '_3d.png')
    fig.savefig(img3d, dpi=150)
    plt.close(fig)
    print('Observed cells:', len(rows), 'Candidate cells:', int(good.sum()))
    print('Status breakdown:', statuses)
    print('2D:', img2d)
    print('3D:', img3d)
    return img2d, img3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    args = parser.parse_args()
    plot(args.csv)
