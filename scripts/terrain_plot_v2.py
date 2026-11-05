#!/usr/bin/env python3
"""Generate headless 2D and 3D preliminary surface maps from terrain_scanner_v2 CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(csv_path, output_dir=None, min_frames=3):
    csv_path = Path(csv_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with csv_path.open(newline='', encoding='utf-8') as fp:
        for row in csv.DictReader(fp):
            rows.append((float(row['x']), float(row['y']),
                         float(row['terrain_z']), int(row['sample_count']),
                         int(row.get('frame_count', 1))))
    if not rows:
        raise ValueError(f'No mapped cells in {csv_path}')
    data = np.asarray(rows, dtype=float)
    x, y, z, points, frames = data.T
    reliable = frames >= min_frames

    # Leave gaps blank. Never invent heights where no surface was observed.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmin, vmax = float(np.min(z)), float(np.max(z))
    if abs(vmax-vmin) < 1e-6:
        vmax = vmin + 1e-6
    for ax, mask, title in (
        (axes[0], np.ones(len(z), dtype=bool), 'All observed cells'),
        (axes[1], reliable, f'Cells observed in {min_frames}+ distinct clouds')):
        ax.set_facecolor('0.94')
        if np.any(mask):
            ax.scatter(x[mask], y[mask], c=z[mask], cmap='terrain',
                       marker='s', s=17, vmin=vmin, vmax=vmax, linewidths=0)
        else:
            ax.text(0.5, 0.5, 'No cells meet threshold',
                    transform=ax.transAxes, ha='center', va='center')
        ax.set(xlabel='X (m)', ylabel='Y (m)', title=title)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.18)
    scalar = plt.cm.ScalarMappable(cmap='terrain',
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(scalar, ax=axes, shrink=0.8, label='Lowest observed Z (m)')
    fig.suptitle('RS1 preliminary observed-surface map (not verified terrain)')
    png2d = output_dir / f'{csv_path.stem}_2d.png'
    fig.savefig(png2d, dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig3d = plt.figure(figsize=(11, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    scatter = ax3d.scatter(x, y, z, c=z, cmap='terrain', s=7,
                           vmin=vmin, vmax=vmax)
    ax3d.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Observed Z (m)',
             title='RS1 3D observed-surface samples')
    fig3d.colorbar(scatter, ax=ax3d, shrink=0.65, label='Observed Z (m)')
    png3d = output_dir / f'{csv_path.stem}_3d.png'
    fig3d.savefig(png3d, dpi=160, bbox_inches='tight')
    plt.close(fig3d)
    print(f'Input CSV: {csv_path}')
    print(f'Mapped cells: {len(z)}; observed in {min_frames}+ clouds: {int(reliable.sum())}')
    print(f'Observed Z: {z.min():.3f} ... {z.max():.3f} m')
    print(f'2D plot: {png2d}\n3D plot: {png3d}')
    return png2d, png3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='terrain_profile_*.csv from terrain_scanner_v2')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--min-frames', type=int, default=3)
    args = parser.parse_args()
    if args.min_frames < 1:
        parser.error('--min-frames must be >= 1')
    plot(args.csv, args.output_dir, args.min_frames)


if __name__ == '__main__':
    main()
