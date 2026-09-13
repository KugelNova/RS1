
#!/usr/bin/env python3

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


csv_path = os.path.expanduser(
    '~/41068_ws/terrain_profile.csv'
)

output_path = os.path.expanduser(
    '~/41068_ws/terrain_elevation_map.png'
)

minimum_samples = 20

x_values = []
y_values = []
z_values = []
sample_counts = []

with open(csv_path, newline='') as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        x_values.append(float(row['x']))
        y_values.append(float(row['y']))
        z_values.append(float(row['terrain_z']))
        sample_counts.append(int(row['sample_count']))

x_values = np.asarray(x_values)
y_values = np.asarray(y_values)
z_values = np.asarray(z_values)
sample_counts = np.asarray(sample_counts)

reliable = sample_counts >= minimum_samples

figure, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    constrained_layout=True
)

colour_min = np.min(z_values)
colour_max = np.max(z_values)

all_map = axes[0].scatter(
    x_values,
    y_values,
    c=z_values,
    cmap='terrain',
    marker='s',
    s=80,
    vmin=colour_min,
    vmax=colour_max
)

axes[0].set_title('All measured terrain cells')
axes[0].set_xlabel('World x (m)')
axes[0].set_ylabel('World y (m)')
axes[0].set_aspect('equal')
axes[0].grid(alpha=0.25)

reliable_map = axes[1].scatter(
    x_values[reliable],
    y_values[reliable],
    c=z_values[reliable],
    cmap='terrain',
    marker='s',
    s=80,
    vmin=colour_min,
    vmax=colour_max
)

axes[1].set_title(
    f'Reliable cells ({minimum_samples}+ samples)'
)
axes[1].set_xlabel('World x (m)')
axes[1].set_ylabel('World y (m)')
axes[1].set_aspect('equal')
axes[1].grid(alpha=0.25)

colour_bar = figure.colorbar(
    all_map,
    ax=axes,
    shrink=0.85
)

colour_bar.set_label('Measured elevation z (m)')

figure.suptitle(
    'Parrot Drone Terrain Elevation Map',
    fontsize=15
)

figure.savefig(
    output_path,
    dpi=200,
    bbox_inches='tight'
)

print(f'Total cells: {len(z_values)}')
print(f'Reliable cells: {np.sum(reliable)}')
print(f'Minimum elevation: {np.min(z_values):.3f} m')
print(f'Maximum elevation: {np.max(z_values):.3f} m')
print(f'Saved map to: {output_path}')

plt.show()