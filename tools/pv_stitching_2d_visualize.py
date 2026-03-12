#!/usr/bin/env python3
"""Visualize 2D PV stitching results from exact_pv_stitching_2d CSV output."""

import sys
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def read_csv(fname):
    """Read curves CSV file."""
    curves = defaultdict(list)
    with open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row['curve_id'])
            curves[cid].append({
                'point_idx': int(row['point_idx']),
                'x': float(row['x']),
                'y': float(row['y']),
                'lambda': float(row['lambda']),
            })
    return curves


def plot_case(csv_file, title, desc, N, output_pdf):
    """Plot one test case."""
    curves = read_csv(csv_file)
    if not curves:
        print(f"  No curves in {csv_file}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(curves), 3)))

    # Left panel: spatial (x,y)
    ax1.set_xlim(-0.5, N - 0.5)
    ax1.set_ylim(-0.5, N - 0.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title}\n{desc}')

    # Draw mesh grid (light)
    for i in range(N):
        ax1.axhline(i, color='#e0e0e0', linewidth=0.3)
        ax1.axvline(i, color='#e0e0e0', linewidth=0.3)

    n_closed = 0
    n_open = 0
    for cid in sorted(curves.keys()):
        pts = curves[cid]
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        # Check if closed (first and last point ~same)
        closed = (len(pts) > 2 and
                  abs(xs[0] - xs[-1]) < 0.01 and
                  abs(ys[0] - ys[-1]) < 0.01)
        if closed:
            n_closed += 1
        else:
            n_open += 1

        color = colors[cid % len(colors)]
        ax1.plot(xs, ys, '-', color=color, linewidth=1.5,
                 label=f'C{cid} ({len(pts)}pts, {"closed" if closed else "open"})')
        # Mark endpoints
        if not closed:
            ax1.plot(xs[0], ys[0], 'o', color=color, markersize=5)
            ax1.plot(xs[-1], ys[-1], 's', color=color, markersize=5)

    ax1.legend(fontsize=8, loc='upper right')

    # Right panel: lambda profile along curve arc length
    ax2.set_xlabel('Arc-length parameter')
    ax2.set_ylabel('λ')
    ax2.set_title(f'λ along curves ({n_open} open, {n_closed} closed)')

    for cid in sorted(curves.keys()):
        pts = curves[cid]
        xs = [p['x'] for p in pts]
        ys = [p['y'] for p in pts]
        lams = [p['lambda'] for p in pts]

        # Compute cumulative arc length
        arc = [0.0]
        for i in range(1, len(xs)):
            ds = np.sqrt((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2)
            arc.append(arc[-1] + ds)

        color = colors[cid % len(colors)]
        ax2.plot(arc, lams, '-o', color=color, markersize=2,
                 linewidth=1.0, label=f'C{cid}')

    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_pdf, dpi=150)
    plt.close(fig)
    print(f"  Wrote {output_pdf}")


def main():
    # Default: look for CSV files in current directory
    csv_dir = '.'
    output_dir = '.'
    N = 32

    if len(sys.argv) > 1:
        csv_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        N = int(sys.argv[3])

    cases = [
        ('field1_diagonal_line', 'F1: Diagonal Line', 'det = x-y-0.2'),
        ('field2_circle', 'F2: Circle', 'det = R²-r² (R=10.3)'),
        ('field3_two_lines', 'F3: Two Vertical Lines', 'det = (x-cx)²-R² (R=5.3)'),
        ('field4_horizontal_line', 'F4: Horizontal Line', 'det = -(y-15.8)'),
    ]

    for name, title, desc in cases:
        csv_file = os.path.join(csv_dir, f'{name}_curves.csv')
        if not os.path.exists(csv_file):
            print(f"  Skipping {name}: {csv_file} not found")
            continue
        output_pdf = os.path.join(output_dir, f'stitching_2d_{name}.pdf')
        plot_case(csv_file, title, desc, N, output_pdf)


if __name__ == '__main__':
    main()
