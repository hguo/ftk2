#!/usr/bin/env python3
"""
Visualize single-tet PV cases from JSON output of pv_tet_case_finder.

Three-panel figure per case:
  (a) 3D tet wireframe with PV curve segments (colored per segment)
  (b) Lambda-ring diagram with segment bands and puncture labels
  (c) Info panel: V, W field values and Q, P polynomials

Usage:
  python3 pv_tet_visualize.py cases.json --output-dir figures/
  python3 pv_tet_visualize.py cases.json --seeds 15587 414
"""

import json
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ─── Publication styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 150,
})

# Colors for segments (ColorBrewer Set1, up to 6)
SEGMENT_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#ff7f00',  # orange
    '#984ea3',  # purple
    '#a65628',  # brown
]

# Tet vertices: regular tetrahedron in 3D
TET_VERTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [0.5, np.sqrt(3)/6, np.sqrt(6)/3],
])

TET_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

FACE_VERTS = [
    [1, 3, 2],  # face 0
    [0, 2, 3],  # face 1
    [0, 3, 1],  # face 2
    [0, 1, 2],  # face 3
]


# ─── Math helpers ────────────────────────────────────────────────────────────

def poly_eval(coeffs, x):
    """Evaluate polynomial coeffs[0] + coeffs[1]*x + ... at x (Horner)."""
    val = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        val = val * x + coeffs[i]
    return val


def bary_to_3d(bary, face_idx):
    """Convert face barycentric coords to 3D position on regular tet."""
    fv = FACE_VERTS[face_idx]
    return sum(bary[i] * TET_VERTS[fv[i]] for i in range(3))


def lambda_to_bary_tet(lam, Q_coeffs, P_coeffs):
    """Convert lambda to tet barycentric coords: mu_i = P_i(lam) / Q(lam)."""
    Q_val = poly_eval(Q_coeffs, lam)
    if abs(Q_val) < 1e-30:
        return None
    mu = np.array([poly_eval(P_coeffs[i], lam) / Q_val for i in range(4)])
    return mu


def bary_tet_to_3d(mu):
    """Convert tet barycentric coords (4 values) to 3D."""
    return sum(mu[i] * TET_VERTS[i] for i in range(4))


# ─── Lambda-ring mapping ────────────────────────────────────────────────────

def lambda_to_angle(lam, scale=1.0):
    """Map lambda to angle on ring. lambda=0 -> -pi/2 (bottom), ±inf -> pi/2 (top)."""
    theta = 2.0 * np.arctan(lam / scale)
    return theta - np.pi / 2


def angle_to_xy(angle, radius=1.0):
    return radius * np.cos(angle), radius * np.sin(angle)


# ─── Curve sampling ─────────────────────────────────────────────────────────

def sample_pv_curve(Q_coeffs, P_coeffs, lam_lo, lam_hi, is_infinity,
                    n_samples=200, puncture_lambdas=None):
    """Sample PV curve in an interval.
    Returns list of (pts_3d_array, lam_entry, lam_exit) tuples."""
    # Build lambda sample array
    if is_infinity:
        lo_inf = (lam_lo is None)
        hi_inf = (lam_hi is None)
        if lo_inf and hi_inf:
            t = np.linspace(-0.499 * np.pi, 0.499 * np.pi, n_samples)
            lam_vals = np.tan(t)
        elif lo_inf:
            t = np.linspace(-0.499 * np.pi, 0, n_samples)
            s = max(abs(lam_hi), 1.0) * 10.0
            lam_vals = lam_hi + s * np.tan(t)
        else:
            t = np.linspace(0, 0.499 * np.pi, n_samples)
            s = max(abs(lam_lo), 1.0) * 10.0
            lam_vals = lam_lo + s * np.tan(t)
    else:
        mid = np.linspace(lam_lo, lam_hi, n_samples)
        eps = (lam_hi - lam_lo) * 0.01
        near_lo = np.linspace(lam_lo, lam_lo + eps, 10)
        near_hi = np.linspace(lam_hi - eps, lam_hi, 10)
        lam_vals = np.sort(np.unique(np.concatenate([mid, near_lo, near_hi])))

    # Densify near puncture lambdas
    if puncture_lambdas:
        extra = []
        for pl in puncture_lambdas:
            if pl is None:
                continue
            lo_ok = (lam_lo is None) or (pl >= lam_lo)
            hi_ok = (lam_hi is None) or (pl <= lam_hi)
            if lo_ok and hi_ok:
                spread = max(abs(pl), 1.0) * 0.1
                extra.append(np.linspace(pl - spread, pl + spread, 50))
        if extra:
            lam_vals = np.sort(np.unique(np.concatenate([lam_vals] + extra)))

    def is_inside(mu):
        return mu is not None and np.all(mu > -1e-6) and np.all(mu < 1 + 1e-6)

    def bisect_boundary(lam_in, lam_out, n_bisect=30):
        """Returns (3d_point, lambda_at_boundary)."""
        for _ in range(n_bisect):
            lam_mid = 0.5 * (lam_in + lam_out)
            mu_mid = lambda_to_bary_tet(lam_mid, Q_coeffs, P_coeffs)
            if is_inside(mu_mid):
                lam_in = lam_mid
            else:
                lam_out = lam_mid
        mu = lambda_to_bary_tet(lam_in, Q_coeffs, P_coeffs)
        return bary_tet_to_3d(np.clip(mu, 0, 1)), lam_in

    segments = []
    current = []
    seg_lam_entry = None
    seg_lam_exit = None
    prev_lam = None
    prev_inside = False

    for lam in lam_vals:
        mu = lambda_to_bary_tet(lam, Q_coeffs, P_coeffs)
        inside = is_inside(mu)
        if inside:
            if not prev_inside and prev_lam is not None:
                pt, bl = bisect_boundary(lam, prev_lam)
                current.append(pt)
                seg_lam_entry = bl
            elif not prev_inside:
                seg_lam_entry = lam
            current.append(bary_tet_to_3d(mu))
            seg_lam_exit = lam
        else:
            if prev_inside and prev_lam is not None:
                pt, bl = bisect_boundary(prev_lam, lam)
                current.append(pt)
                seg_lam_exit = bl
            if len(current) > 1:
                segments.append((np.array(current), seg_lam_entry, seg_lam_exit))
            current = []
        prev_lam = lam
        prev_inside = inside

    if len(current) > 1:
        segments.append((np.array(current), seg_lam_entry, seg_lam_exit))

    return segments


# ─── Segment collection ─────────────────────────────────────────────────────

def _lam_dist(a, b):
    """Distance between two lambda values, handling None (=infinity)."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1e15  # large but finite, so finite matches win
    return abs(a - b)


def _match_to_pair(lam_e, lam_x, pairs, punctures):
    """Match a sub-segment to the pair whose punctures are nearest."""
    best_pair = 0
    best_dist = float('inf')
    for idx, (pi1, pi2, _) in enumerate(pairs):
        l1 = punctures[pi1]['lambda']
        l2 = punctures[pi2]['lambda']
        d = min(_lam_dist(lam_e, l1), _lam_dist(lam_e, l2),
                _lam_dist(lam_x, l1), _lam_dist(lam_x, l2))
        if d < best_dist:
            best_dist = d
            best_pair = idx
    return best_pair


def collect_segments(case_data):
    """Build segments from pre-computed puncture pairing (from C++ JSON)."""
    Q = case_data['Q_coeffs']
    P = case_data['P_coeffs']
    intervals = case_data['intervals']
    punctures = case_data['punctures']
    puncture_lambdas = [p['lambda'] for p in punctures]

    # Read pairs from JSON (computed by C++ build_pairs)
    pairs = [(p['pi_a'], p['pi_b'], p.get('is_cross', False))
             for p in case_data.get('pairs', [])]

    # Sample ALL intervals (including n_pv=0 infinity intervals for Cw curves)
    all_subsegs = []
    for iv in intervals:
        segs = sample_pv_curve(Q, P, iv['lb'], iv['ub'], iv['is_infinity'],
                               puncture_lambdas=puncture_lambdas)
        all_subsegs.extend(segs)

    # Match each sub-segment to the nearest pair by lambda proximity
    pair_subsegs = defaultdict(list)
    for pts, lam_e, lam_x in all_subsegs:
        best = _match_to_pair(lam_e, lam_x, pairs, punctures)
        pair_subsegs[best].append(pts)

    # Build segment list
    segments = []
    for idx, (pi1, pi2, inf_span) in enumerate(pairs):
        color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
        segments.append({
            'pts_list': pair_subsegs.get(idx, []),
            'color': color,
            'pi_entry': pi1,
            'pi_exit': pi2,
            'lam_entry': punctures[pi1]['lambda'],
            'lam_exit': punctures[pi2]['lambda'],
            'infinity_spanning': inf_span,
        })

    return segments


# ─── Formatting helpers ─────────────────────────────────────────────────────

def poly_to_latex(coeffs, name='Q'):
    """Format polynomial as matplotlib mathtext string."""
    sup = {2: '2', 3: '3', 4: '4'}
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        c = int(round(coeffs[i]))
        if c == 0:
            continue
        if i == 0:
            var = ''
        elif i == 1:
            var = '\\lambda'
        else:
            var = '\\lambda^{' + sup.get(i, str(i)) + '}'

        if not terms:  # first term
            if abs(c) == 1 and i > 0:
                term = ('-' + var) if c < 0 else var
            else:
                term = str(c) + var
        else:
            if abs(c) == 1 and i > 0:
                term = (' - ' + var) if c < 0 else (' + ' + var)
            else:
                term = f' - {abs(c)}{var}' if c < 0 else f' + {c}{var}'
        terms.append(term)

    expr = ''.join(terms) if terms else '0'
    return f'${name}(\\lambda) = {expr}$'


# ─── 3D Tet panel ───────────────────────────────────────────────────────────

def find_d00_vertices(case_data):
    """Find tet vertices where V×W = 0 (D00 degeneracy)."""
    V = np.array(case_data['V'], dtype=float)
    W = np.array(case_data['W'], dtype=float)
    d00 = []
    for i in range(4):
        cross = np.cross(V[i], W[i])
        if np.linalg.norm(cross) < 1e-10 * max(np.linalg.norm(V[i]),
                                                 np.linalg.norm(W[i]), 1e-30):
            d00.append(i)
    return d00


def compute_cv_position(case_data):
    """Compute Cv position (lambda=0): mu_i = P_i(0)/Q(0)."""
    Q = case_data['Q_coeffs']
    P = case_data['P_coeffs']
    Q0 = Q[0]
    if abs(Q0) < 1e-30:
        return None
    mu = np.array([P[i][0] / Q0 for i in range(4)])
    if np.all(mu > -1e-6) and np.all(mu < 1 + 1e-6):
        return bary_tet_to_3d(np.clip(mu, 0, 1))
    return None


def compute_cw_position(case_data):
    """Compute Cw position (lambda->inf): mu_i = P_i[3]/Q[3]."""
    Q = case_data['Q_coeffs']
    P = case_data['P_coeffs']
    Q3 = Q[3] if len(Q) > 3 else 0
    if abs(Q3) < 1e-30:
        return None
    mu = np.array([P[i][3] / Q3 if len(P[i]) > 3 else 0 for i in range(4)])
    if np.all(mu > -1e-6) and np.all(mu < 1 + 1e-6):
        return bary_tet_to_3d(np.clip(mu, 0, 1))
    return None


def draw_tet_wireframe(ax):
    """Draw tet wireframe with semi-transparent faces."""
    for i, j in TET_EDGES:
        ax.plot3D(*zip(TET_VERTS[i], TET_VERTS[j]),
                  color='#555555', linewidth=1.0, alpha=0.6)

    face_polys = [[TET_VERTS[v] for v in tri] for tri in FACE_VERTS]
    collection = Poly3DCollection(face_polys,
                                  facecolors='#e0e0e0', edgecolors='none',
                                  alpha=0.08)
    ax.add_collection3d(collection)

    offsets = [
        np.array([-0.06, -0.04, -0.04]),
        np.array([0.04, -0.04, -0.04]),
        np.array([0.0, 0.06, -0.04]),
        np.array([0.0, 0.0, 0.06]),
    ]
    for i in range(4):
        p = TET_VERTS[i] + offsets[i]
        ax.text(p[0], p[1], p[2], f'$v_{i}$',
                fontsize=9, color='#333333', ha='center', va='center')


def _find_segment_for_lambda(lam, segments):
    """Find the segment whose lambda range contains lam. Return its color."""
    for seg in segments:
        l1 = seg['lam_entry']
        l2 = seg['lam_exit']
        if seg.get('infinity_spanning', False):
            # Infinity-spanning: covers (l1, +inf) U (-inf, l2) or similar
            if l1 is not None and l2 is not None:
                hi, lo = max(l1, l2), min(l1, l2)
                if lam >= hi or lam <= lo:
                    return seg['color']
            else:
                return seg['color']
        else:
            lo = min(l1, l2) if l1 is not None and l2 is not None else None
            hi = max(l1, l2) if l1 is not None and l2 is not None else None
            if lo is not None and hi is not None and lo <= lam <= hi:
                return seg['color']
    return '#333333'


def draw_special_points(ax, case_data, segments=None):
    """Draw D00, D01, Cv, Cw, SR markers."""
    category = case_data['category']
    if segments is None:
        segments = []

    # D01: puncture on a tet edge (one bary coord ≈ 0)
    # Deduplicate: two adjacent faces can report the same edge point
    if 'D01' in category:
        punctures = case_data['punctures']
        d01_offsets = [
            np.array([0.10, 0.08, 0.10]),
            np.array([-0.10, -0.06, 0.12]),
            np.array([0.08, -0.10, 0.10]),
            np.array([-0.08, 0.10, 0.12]),
        ]
        d01_positions = []  # (pos, lam) for dedup
        for i, pi in enumerate(punctures):
            bary = np.array(pi['bary'])
            n_small = np.sum(np.abs(bary) < 1e-3)
            if n_small >= 1:
                pos = bary_to_3d(bary, pi['face'])
                # Skip if too close to an already-labeled D01 point
                is_dup = False
                for prev_pos, _ in d01_positions:
                    if np.linalg.norm(pos - prev_pos) < 0.05:
                        is_dup = True
                        break
                if is_dup:
                    continue
                lam = pi['lambda']
                d01_positions.append((pos, lam))
                lam_str = f'$\\lambda={lam:.2f}$' if lam is not None else r'$\lambda=\infty$'
                off = d01_offsets[len(d01_positions) - 1 % len(d01_offsets)]
                ax.plot3D([pos[0], pos[0]+off[0]], [pos[1], pos[1]+off[1]],
                          [pos[2], pos[2]+off[2]],
                          color='#dd5500', linewidth=1.2, linestyle='-')
                ax.text(pos[0]+off[0], pos[1]+off[1], pos[2]+off[2]+0.01,
                        f'D01  {lam_str}',
                        fontsize=8, ha='left' if off[0] > 0 else 'right',
                        va='bottom',
                        color='#aa4400', fontweight='bold',
                        bbox=dict(facecolor='#fff4ee', alpha=0.9,
                                  edgecolor='#dd5500', linewidth=1.2,
                                  boxstyle='round,pad=0.3'))

    # D00: vertices where V x W = 0
    if 'D00' in category:
        d00_verts = find_d00_vertices(case_data)
        for vi in d00_verts:
            p = TET_VERTS[vi]
            # Large magenta star
            ax.scatter(p[0], p[1], p[2], c='#ff00ff', s=250, marker='*',
                       zorder=10, edgecolors='black', linewidth=1.0)
            # Leader line from offset annotation to point
            off = np.array([0.08, 0.08, 0.12])
            ax.plot3D([p[0], p[0]+off[0]], [p[1], p[1]+off[1]],
                      [p[2], p[2]+off[2]],
                      color='#cc00cc', linewidth=1.2, linestyle='-')
            # D00 is at a vertex; lambda depends on which field vanishes
            # V[i]×W[i]=0 means V∥W, so lambda = -|V|/|W| or |V|/|W|
            # More precisely: v(λ)=V+λW=0 at vertex i → λ = -V[i]/W[i]
            Vi = np.array(case_data['V'][vi], dtype=float)
            Wi = np.array(case_data['W'][vi], dtype=float)
            # Find λ from any nonzero component
            d00_lam = None
            for comp in range(3):
                if abs(Wi[comp]) > 1e-10:
                    d00_lam = -Vi[comp] / Wi[comp]
                    break
            if d00_lam is not None:
                lam_str = f'$\\lambda={d00_lam:.2f}$'
            else:
                lam_str = r'$\lambda=?$'
            ax.text(p[0]+off[0], p[1]+off[1], p[2]+off[2]+0.01,
                    lam_str,
                    fontsize=8, ha='left', va='bottom',
                    color='#cc00cc', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9,
                              edgecolor='#cc00cc', linewidth=1.2,
                              boxstyle='round,pad=0.3'))

    # Cv: critical point of v at lambda=0
    if 'Cv' in category:
        pos = compute_cv_position(case_data)
        if pos is not None:
            cv_color = _find_segment_for_lambda(0.0, segments)
            ax.scatter(pos[0], pos[1], pos[2], c=cv_color, s=200,
                       marker='*', zorder=10, edgecolors='black',
                       linewidth=1.0)
            off = np.array([-0.08, 0.08, 0.10])
            ax.plot3D([pos[0], pos[0]+off[0]], [pos[1], pos[1]+off[1]],
                      [pos[2], pos[2]+off[2]],
                      color=cv_color, linewidth=1.2, linestyle='-')
            ax.text(pos[0]+off[0], pos[1]+off[1], pos[2]+off[2]+0.01,
                    r'Cv ($\lambda=0$)',
                    fontsize=8, ha='right', va='bottom',
                    color=cv_color, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9,
                              edgecolor=cv_color, linewidth=1.2,
                              boxstyle='round,pad=0.3'))

    # Cw: critical point of w at lambda->inf
    if 'Cw' in category:
        pos = compute_cw_position(case_data)
        if pos is not None:
            # Cw belongs to the infinity-spanning segment
            cw_color = '#333333'
            for seg in segments:
                if seg.get('infinity_spanning', False):
                    cw_color = seg['color']
                    break
            ax.scatter(pos[0], pos[1], pos[2], c=cw_color, s=200,
                       marker='*', zorder=10, edgecolors='black',
                       linewidth=1.0)
            off = np.array([0.10, -0.06, 0.10])
            ax.plot3D([pos[0], pos[0]+off[0]], [pos[1], pos[1]+off[1]],
                      [pos[2], pos[2]+off[2]],
                      color=cw_color, linewidth=1.2, linestyle='-')
            ax.text(pos[0]+off[0], pos[1]+off[1], pos[2]+off[2]+0.01,
                    r'Cw ($\lambda \to \infty$)',
                    fontsize=8, ha='left', va='bottom',
                    color=cw_color, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.9,
                              edgecolor=cw_color, linewidth=1.2,
                              boxstyle='round,pad=0.3'))


    # SR: shared root — Q-root where some P_k also vanishes
    if 'SR' in category:
        Q = case_data['Q_coeffs']
        P = case_data['P_coeffs']
        Q_roots = case_data.get('Q_roots', [])
        # Find true SR roots: Q-roots where P_k(qr) ≈ 0 for some k
        sr_roots = []
        for qr in Q_roots:
            q_scale = max(abs(poly_eval(Q, qr + 0.01)), 1e-10)
            for k in range(4):
                pk_val = abs(poly_eval(P[k], qr))
                pk_scale = max(abs(poly_eval(P[k], qr + 0.1)),
                               abs(poly_eval(P[k], qr - 0.1)), 1e-10)
                if pk_val / pk_scale < 1e-6:
                    sr_roots.append(qr)
                    break
        sr_offsets = [
            np.array([0.06, 0.10, 0.08]),
            np.array([-0.08, 0.06, 0.12]),
            np.array([0.10, -0.06, 0.10]),
        ]
        for si, qr in enumerate(sr_roots):
            # Compute 3D position via L'Hopital
            Q_prime = [Q[j] * j for j in range(1, len(Q))]
            Qp_val = poly_eval(Q_prime, qr)
            if abs(Qp_val) < 1e-20:
                continue
            mu = np.array([
                poly_eval([P[i][j] * j for j in range(1, len(P[i]))], qr) / Qp_val
                for i in range(4)
            ])
            mu_clip = np.clip(mu, 0, 1)
            s = mu_clip.sum()
            if s > 1e-10:
                mu_clip /= s
            pos = bary_tet_to_3d(mu_clip)
            ax.scatter(pos[0], pos[1], pos[2], c='#ff8800', s=180,
                       marker='D', zorder=10, edgecolors='black',
                       linewidth=1.0)
            off = sr_offsets[si % len(sr_offsets)]
            ax.plot3D([pos[0], pos[0]+off[0]], [pos[1], pos[1]+off[1]],
                      [pos[2], pos[2]+off[2]],
                      color='#ff8800', linewidth=1.2, linestyle='-')
            ax.text(pos[0]+off[0], pos[1]+off[1], pos[2]+off[2]+0.01,
                    f'SR ($\\lambda={qr:.2f}$)',
                    fontsize=8, ha='left', va='bottom',
                    color='#cc6600', fontweight='bold',
                    bbox=dict(facecolor='#fff8ee', alpha=0.9,
                              edgecolor='#ff8800', linewidth=1.2,
                              boxstyle='round,pad=0.3'))


def draw_vector_arrows(ax, case_data, arrow_scale=0.10):
    """Draw V (red) and W (blue) arrows at tet vertices."""
    V = np.array(case_data['V'], dtype=float)
    W = np.array(case_data['W'], dtype=float)
    v_max = max(np.max(np.abs(V)), 1e-10)
    w_max = max(np.max(np.abs(W)), 1e-10)
    for i in range(4):
        origin = TET_VERTS[i]
        v_dir = V[i] / v_max * arrow_scale
        w_dir = W[i] / w_max * arrow_scale
        ax.quiver(*origin, *v_dir, color='#cc0000', alpha=0.8,
                  arrow_length_ratio=0.25, linewidth=1.0)
        ax.quiver(*origin, *w_dir, color='#0044cc', alpha=0.8,
                  arrow_length_ratio=0.25, linewidth=1.0)


def draw_pv_curves(ax, segments):
    """Draw PV curve segments, each with its own color."""
    for seg in segments:
        for pts in seg['pts_list']:
            if len(pts) > 1:
                ax.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                          color=seg['color'], linewidth=2.0, alpha=0.9)


def _find_special_punctures(case_data):
    """Return set of puncture indices that are D01/D00 (labeled separately)."""
    category = case_data['category']
    special = set()
    if 'D01' in category:
        for i, pi in enumerate(case_data['punctures']):
            bary = np.array(pi['bary'])
            if np.sum(np.abs(bary) < 1e-3) >= 1:
                special.add(i)
    if 'D00' in category:
        for i, pi in enumerate(case_data['punctures']):
            bary = np.array(pi['bary'])
            if np.sum(np.abs(bary) < 1e-3) >= 2:
                special.add(i)
    return special


def draw_puncture_markers(ax, case_data, segments):
    """Draw puncture points colored by segment, with lambda labels."""
    punctures = case_data['punctures']
    special = _find_special_punctures(case_data)

    # Build puncture → color map
    punc_color = {}
    for seg in segments:
        punc_color.setdefault(seg['pi_entry'], seg['color'])
        punc_color.setdefault(seg['pi_exit'], seg['color'])

    # Alternate label offset direction to reduce overlap
    label_offsets = [
        np.array([0.0, 0.0, 0.07]),
        np.array([0.0, 0.06, 0.04]),
        np.array([0.06, 0.0, 0.04]),
        np.array([-0.06, 0.0, 0.04]),
    ]

    for i, pi in enumerate(punctures):
        pos = bary_to_3d(pi['bary'], pi['face'])
        color = punc_color.get(i, '#666666')
        bary = np.array(pi['bary'])
        n_small = np.sum(np.abs(bary) < 1e-4)

        if n_small >= 2:
            marker, size = 'D', 60
        elif n_small == 1:
            marker, size = 's', 50
        else:
            marker, size = 'o', 40

        ax.scatter(*pos, c=color, s=size, marker=marker, zorder=6,
                   edgecolors='black', linewidth=0.7)

        # Skip lambda label for D01/D00 punctures (labeled by draw_special_points)
        if i in special:
            continue

        # Lambda label with leader line
        lam = pi['lambda']
        lam_str = f'$\\lambda$={lam:.2f}' if lam is not None else r'$\lambda=\infty$'
        off = label_offsets[i % len(label_offsets)]
        lp = pos + off
        ax.plot3D([pos[0], lp[0]], [pos[1], lp[1]], [pos[2], lp[2]],
                  color=color, linewidth=0.8, alpha=0.6)
        ax.text(lp[0], lp[1], lp[2], lam_str,
                fontsize=8, ha='center', va='bottom', color=color,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85,
                          edgecolor=color, linewidth=0.6,
                          boxstyle='round,pad=0.2'))


# ─── Lambda-ring panel ──────────────────────────────────────────────────────

def draw_lambda_ring(ax, case_data, segments):
    """Draw lambda-ring with per-segment colored bands and puncture labels."""
    Q_roots = case_data['Q_roots']
    punctures = case_data['punctures']

    # Choose scale to spread Q roots around the ring
    all_lams = [p['lambda'] for p in punctures if p['lambda'] is not None]
    all_lams += [r for r in Q_roots]
    abs_vals = [abs(v) for v in all_lams if abs(v) > 1e-15]
    scale = np.median(abs_vals) * 1.2 if abs_vals else 1.0
    scale = max(scale, 0.5)

    R_ring = 1.0

    # Draw main ring circle
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_ring * np.cos(th), R_ring * np.sin(th),
            color='#333333', linewidth=1.5, zorder=2)

    # Mark infinity at top (no X, just label)
    ax.text(0, R_ring + 0.10, r'$\infty$', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333')

    # Mark lambda=0 at bottom
    a0 = lambda_to_angle(0.0, scale)
    x0, y0 = angle_to_xy(a0, R_ring)
    ax.plot(x0, y0, '|', color='#666666', markersize=6, markeredgewidth=1.5,
            zorder=5)
    ax.text(x0, y0 - 0.12, r'$0$', ha='center', va='top',
            fontsize=9, color='#666666')

    # ── Draw colored band for each segment (NOT per Q-interval!) ──
    # REQUIREMENT: bands must be per-segment, matching 3D curve colors.
    # REQUIREMENT: treat infinity-spanning interval as ONE single interval,
    #   not two separate halves.  Bands must NEVER overlap.
    r_inner = 0.85
    r_outer = 0.98

    def _draw_band(a_start, a_end, color):
        """Draw a colored annular band from a_start to a_end (counterclockwise)."""
        arc_th = np.linspace(a_start, a_end, 80)
        inner_x = r_inner * np.cos(arc_th)
        inner_y = r_inner * np.sin(arc_th)
        outer_x = r_outer * np.cos(arc_th)
        outer_y = r_outer * np.sin(arc_th)
        verts_x = np.concatenate([outer_x, inner_x[::-1]])
        verts_y = np.concatenate([outer_y, inner_y[::-1]])
        verts = np.column_stack([verts_x, verts_y])
        ax.add_patch(Polygon(verts, closed=True,
                             facecolor=color, alpha=0.45,
                             edgecolor=color,
                             linewidth=0.5, zorder=1))

    for seg in segments:
        lam1 = seg['lam_entry']
        lam2 = seg['lam_exit']

        # Map lambdas to angles (None = infinity = top of ring)
        a1 = lambda_to_angle(lam1, scale) if lam1 is not None else np.pi / 2
        a2 = lambda_to_angle(lam2, scale) if lam2 is not None else np.pi / 2

        if seg.get('infinity_spanning', False):
            # Draw arc through infinity (top of ring, angle=pi/2)
            a_hi = max(a1, a2)  # angle closer to +inf side
            a_lo = min(a1, a2)  # angle closer to -inf side
            _draw_band(a_hi, a_lo + 2 * np.pi, seg['color'])
        else:
            # Draw short arc
            if a1 > a2:
                a1, a2 = a2, a1
            _draw_band(a1, a2, seg['color'])

    # ── Mark Q roots (hollow circles INSIDE the ring, with labels) ──
    q_label_r = 0.60  # inside ring
    q_label_radii = [q_label_r] * len(Q_roots)
    q_angles = [lambda_to_angle(r, scale) for r in Q_roots]
    # Stagger Q-root labels when close
    q_sorted = sorted(range(len(Q_roots)), key=lambda i: q_angles[i])
    for j in range(1, len(q_sorted)):
        ip, ic = q_sorted[j - 1], q_sorted[j]
        if abs(q_angles[ic] - q_angles[ip]) < 0.30:
            q_label_radii[ic] = 0.42 if q_label_radii[ip] >= 0.55 else 0.60

    for i, r in enumerate(Q_roots):
        a = q_angles[i]
        rx, ry = angle_to_xy(a, R_ring)
        ax.plot(rx, ry, 'o', color='white', markersize=7, zorder=8,
                markeredgecolor='black', markeredgewidth=1.5)

        # Label inside ring
        lr = q_label_radii[i]
        lx, ly = angle_to_xy(a, lr)
        mx, my = angle_to_xy(a, R_ring - 0.06)
        ax.plot([mx, lx], [my, ly], '-', color='#888888', linewidth=0.6)
        ax.text(lx, ly, f'{r:.2f}',
                ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.5))

    # ── Mark SR points on ring ──
    # SR: Q-root where some P_k also vanishes (true shared root)
    category = case_data.get('category', '')
    if 'SR' in category:
        Q_coeffs = case_data['Q_coeffs']
        P_coeffs = case_data['P_coeffs']
        sr_roots = set()
        for qi, qr in enumerate(Q_roots):
            for k in range(4):
                pk_val = abs(sum(P_coeffs[k][j] * qr**j for j in range(len(P_coeffs[k]))))
                pk_scale = max(
                    abs(sum(P_coeffs[k][j] * (qr + 0.1)**j for j in range(len(P_coeffs[k])))),
                    abs(sum(P_coeffs[k][j] * (qr - 0.1)**j for j in range(len(P_coeffs[k])))),
                    1e-10)
                if pk_val / pk_scale < 1e-6:
                    sr_roots.add(qi)
                    break
        for qi in sr_roots:
            qr = Q_roots[qi]
            a = lambda_to_angle(qr, scale)
            sx, sy = angle_to_xy(a, R_ring)
            ax.plot(sx, sy, 'D', color='#ff8800', markersize=9,
                    zorder=9, markeredgecolor='black', markeredgewidth=1.0)
            # SR label outside ring
            lr = R_ring + 0.35
            lx, ly = angle_to_xy(a, lr)
            mx, my = angle_to_xy(a, R_ring + 0.06)
            ax.plot([mx, lx], [my, ly], '-', color='#ff8800', linewidth=0.8)
            ax.text(lx, ly, 'SR',
                    ha='center', va='center', fontsize=8,
                    color='#cc6600', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='#fff4ee',
                              edgecolor='#ff8800', linewidth=0.8))

    # ── Mark D00/D01 points on ring ──
    # D00: vertex where V∥W (detected from field data)
    # D01: puncture on tet edge (one bary coord ≈ 0, deduplicated)
    if 'D00' in category or 'D01' in category:
        d_markers = []  # (lam, label) deduplicated
        # D00: check vertices for V×W = 0
        V = np.array(case_data['V'])
        W = np.array(case_data['W'])
        for vi in range(4):
            cross = np.cross(V[vi], W[vi])
            if np.all(cross == 0):
                v_zero = np.all(V[vi] == 0)
                w_zero = np.all(W[vi] == 0)
                if w_zero:
                    lam_d = None
                elif v_zero:
                    lam_d = 0.0
                else:
                    lam_d = None
                    for k in range(3):
                        if W[vi][k] != 0:
                            lam_d = -V[vi][k] / W[vi][k]
                            break
                d_markers.append((lam_d, 'D00'))
        # D01: punctures with one near-zero bary coord (on tet edge)
        for pi in punctures:
            bary = pi.get('bary')
            if not bary:
                continue
            n_small = sum(1 for b in bary if abs(b) < 1e-3)
            if n_small >= 2:
                continue  # D00 vertex, already handled above
            if n_small >= 1:
                lam_d = pi['lambda']
                # Deduplicate by λ proximity
                is_dup = False
                for prev_lam, _ in d_markers:
                    if prev_lam is None and lam_d is None:
                        is_dup = True; break
                    if prev_lam is not None and lam_d is not None:
                        if abs(prev_lam - lam_d) < 0.01:
                            is_dup = True; break
                if not is_dup:
                    d_markers.append((lam_d, 'D01'))

        d_color = '#9900cc'
        d_label_offsets = [0.35, 0.55, 0.75, 0.45, 0.65]
        for di, (lam_d, label) in enumerate(d_markers):
            a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
            sx, sy = angle_to_xy(a, R_ring)
            marker = 's' if label == 'D00' else 'v'
            ax.plot(sx, sy, marker, color=d_color, markersize=9,
                    zorder=9, markeredgecolor='black', markeredgewidth=1.0)
            lr = R_ring + d_label_offsets[di % len(d_label_offsets)]
            lx, ly = angle_to_xy(a, lr)
            mx, my = angle_to_xy(a, R_ring + 0.06)
            ax.plot([mx, lx], [my, ly], '-', color=d_color, linewidth=0.8)
            lam_str = f'{lam_d:.2f}' if lam_d is not None else r'$\infty$'
            ax.text(lx, ly, f'{label} ($\\lambda$={lam_str})',
                    ha='center', va='center', fontsize=7,
                    color='#660099', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='#f4eeff',
                              edgecolor=d_color, linewidth=0.8))

    # ── Puncture ticks with lambda labels OUTSIDE the ring ──
    punc_color = {}
    for seg in segments:
        punc_color.setdefault(seg['pi_entry'], seg['color'])
        punc_color.setdefault(seg['pi_exit'], seg['color'])

    # Compute angles for all punctures
    punc_angles = []
    for i, pi in enumerate(punctures):
        lam = pi['lambda']
        if lam is None:
            punc_angles.append(np.pi / 2)
        else:
            punc_angles.append(lambda_to_angle(lam, scale))

    # Stagger label radii outside ring to avoid overlap
    label_radii = [R_ring + 0.35] * len(punctures)
    sorted_idx = sorted(range(len(punctures)), key=lambda i: punc_angles[i])
    for j in range(1, len(sorted_idx)):
        i_prev = sorted_idx[j - 1]
        i_curr = sorted_idx[j]
        if abs(punc_angles[i_curr] - punc_angles[i_prev]) < 0.25:
            label_radii[i_curr] = (R_ring + 0.55
                                   if label_radii[i_prev] < R_ring + 0.50
                                   else R_ring + 0.35)

    for i, pi in enumerate(punctures):
        lam = pi['lambda']
        color = punc_color.get(i, 'black')
        a = punc_angles[i]
        lam_str = f'{lam:.2f}' if lam is not None else r'$\infty$'

        # Tick mark on ring
        x1, y1 = angle_to_xy(a, 0.88)
        x2, y2 = angle_to_xy(a, 1.07)
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2.0, zorder=7)

        # Lambda label outside ring
        lr = label_radii[i]
        lx, ly = angle_to_xy(a, lr)
        mx, my = angle_to_xy(a, R_ring + 0.08)
        ax.plot([mx, lx], [my, ly], '-', color=color, linewidth=0.6, alpha=0.5)
        ax.text(lx, ly, lam_str, ha='center', va='center', fontsize=7,
                color=color,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='#dddddd', linewidth=0.3))

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.6, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')


# ─── Info panel ──────────────────────────────────────────────────────────────

def draw_info_panel(ax, case_data, segments):
    """Display V/W matrices, Q/P polynomials, and PV segment intervals."""
    ax.axis('off')

    V = np.array(case_data['V'], dtype=int)
    W = np.array(case_data['W'], dtype=int)
    Q = case_data['Q_coeffs']
    P = case_data['P_coeffs']

    # V/W matrix text (monospace)
    mat_lines = []
    for i in range(4):
        vr = '[' + ', '.join(f'{v:4d}' for v in V[i]) + ']'
        wr = '[' + ', '.join(f'{v:4d}' for v in W[i]) + ']'
        pv = 'V = ' if i == 0 else '    '
        pw = '    W = ' if i == 0 else '        '
        mat_lines.append(f'{pv}{vr}{pw}{wr}')

    ax.text(0.01, 0.95, '\n'.join(mat_lines), fontsize=8,
            fontfamily='monospace', transform=ax.transAxes, va='top')

    # PV segment intervals (monospace, below matrices)
    category = case_data['category']
    seg_lines = ['Segments:']
    for i, seg in enumerate(segments):
        l1 = seg['lam_entry']
        l2 = seg['lam_exit']
        l1_s = f'{l1:.3f}' if l1 is not None else 'inf'
        l2_s = f'{l2:.3f}' if l2 is not None else 'inf'
        if seg.get('infinity_spanning', False):
            seg_lines.append(f'  S{i+1}: ({l1_s}, +inf) U (-inf, {l2_s})')
        elif l1 is None or l2 is None:
            # One endpoint at infinity
            fin = l1 if l2 is None else l2
            fin_s = f'{fin:.3f}' if fin is not None else 'inf'
            seg_lines.append(f'  S{i+1}: ({fin_s}, inf)')
        else:
            lo_s = f'{min(l1, l2):.3f}'
            hi_s = f'{max(l1, l2):.3f}'
            seg_lines.append(f'  S{i+1}: ({lo_s}, {hi_s})')

    # D00: vertex where V×W=0
    if 'D00' in category:
        d00_verts = find_d00_vertices(case_data)
        for vi in d00_verts:
            Wi = np.array(case_data['W'][vi], dtype=float)
            Vi = np.array(case_data['V'][vi], dtype=float)
            d00_lam = None
            for comp in range(3):
                if abs(Wi[comp]) > 1e-10:
                    d00_lam = -Vi[comp] / Wi[comp]
                    break
            lam_s = f'{d00_lam:.3f}' if d00_lam is not None else '?'
            seg_lines.append(f'  D00: vertex v{vi}, lambda={lam_s}')

    ax.text(0.01, 0.22, '\n'.join(seg_lines), fontsize=7.5,
            fontfamily='monospace', transform=ax.transAxes, va='top')

    # Polynomial text (mathtext for lambda symbols)
    poly_strs = [poly_to_latex(Q, 'Q')]
    for k in range(4):
        poly_strs.append(poly_to_latex(P[k], f'P_{k}'))

    y = 0.95
    for j, ps in enumerate(poly_strs):
        ax.text(0.50, y - j * 0.16, ps, fontsize=8,
                transform=ax.transAxes, va='top')


# ─── Figure assembly ─────────────────────────────────────────────────────────

def visualize_case(case_data, output_path=None):
    """Generate three-panel figure for a single tet case."""
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, height_ratios=[3, 1.2], figure=fig,
                  hspace=0.25, wspace=0.3)

    # Category title
    cat = case_data['category']
    cat_display = cat.replace('_', ' ')
    seed = case_data['seed']
    fig.suptitle(f'{cat_display}  (seed={seed})', fontsize=12,
                 fontweight='bold', y=0.98, fontfamily='monospace')

    # Compute segments
    segments = collect_segments(case_data)

    # Panel (a): 3D tet
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    draw_tet_wireframe(ax3d)
    draw_vector_arrows(ax3d, case_data)
    draw_pv_curves(ax3d, segments)
    draw_puncture_markers(ax3d, case_data, segments)
    draw_special_points(ax3d, case_data, segments)

    # Bubble: closed PV curve inside tet (0 punctures, 0 segments)
    if case_data.get('has_B') and case_data['n_punctures'] == 0:
        Q = case_data['Q_coeffs']
        P = case_data['P_coeffs']
        t = np.linspace(-0.499 * np.pi, 0.499 * np.pi, 400)
        lam_vals = np.tan(t)
        pts = []
        for lam in lam_vals:
            mu = lambda_to_bary_tet(lam, Q, P)
            if all(m >= -0.01 for m in mu):
                mu_clip = np.clip(mu, 0, 1)
                s = mu_clip.sum()
                if s > 1e-10:
                    mu_clip /= s
                pts.append(bary_tet_to_3d(mu_clip))
        if len(pts) > 2:
            pts.append(pts[0])  # close the loop
            pts = np.array(pts)
            ax3d.plot3D(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=SEGMENT_COLORS[0], linewidth=2.5, zorder=5)
        segments = [{'color': SEGMENT_COLORS[0], 'pts_list': [np.array(pts)],
                     'pi_entry': -1, 'pi_exit': -1,
                     'lam_entry': None, 'lam_exit': None,
                     'infinity_spanning': True}]

    ax3d.set_title(f'{case_data["n_punctures"]} punctures, '
                   f'{len(segments)} segment{"s" if len(segments) != 1 else ""}',
                   fontsize=10, pad=-5)
    ax3d.view_init(elev=20, azim=140)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('none')
    ax3d.yaxis.pane.set_edgecolor('none')
    ax3d.zaxis.pane.set_edgecolor('none')
    ax3d.grid(False)
    ax3d.set_xlabel('')
    ax3d.set_ylabel('')
    ax3d.set_zlabel('')

    # Panel (b): Lambda ring
    ax_ring = fig.add_subplot(gs[0, 1])
    draw_lambda_ring(ax_ring, case_data, segments)

    disc_labels = {1: r'$\Delta_Q > 0$ (3 roots)',
                   -1: r'$\Delta_Q < 0$ (1 root)',
                   0: r'$\Delta_Q = 0$'}
    disc_str = disc_labels.get(case_data['Q_disc_sign'],
                               f'$\\Delta_Q$ sign={case_data["Q_disc_sign"]}')
    ax_ring.set_title(f'$\\lambda$-ring: {disc_str}', fontsize=10)

    # Panel (c): Info
    ax_info = fig.add_subplot(gs[1, :])
    draw_info_panel(ax_info, case_data, segments)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize PV tet cases')
    parser.add_argument('input', help='JSON file from pv_tet_case_finder')
    parser.add_argument('--output-dir', '-o', default='figures',
                        help='Output directory (default: figures/)')
    parser.add_argument('--categories', '-c', nargs='+',
                        help='Only visualize these categories')
    parser.add_argument('--first-per-category', '-f', action='store_true',
                        help='Only visualize first case per category')
    parser.add_argument('--max-cases', '-m', type=int, default=50,
                        help='Maximum number of figures (default: 50)')
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf',
                        help='Output format (default: pdf)')
    parser.add_argument('--seeds', nargs='+', type=int,
                        help='Only visualize cases with these seeds')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load cases
    cases = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line: {e}",
                      file=sys.stderr)

    print(f"Loaded {len(cases)} cases")

    # Category histogram
    cat_counts = defaultdict(int)
    for c in cases:
        cat_counts[c['category']] += 1
    print("\nCategory distribution:")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:40s} {cat_counts[cat]}")

    # Filter
    if args.categories:
        cases = [c for c in cases if c['category'] in args.categories]
        print(f"\nFiltered to {len(cases)} cases")

    if args.seeds:
        seed_set = set(args.seeds)
        cases = [c for c in cases if c['seed'] in seed_set]
        print(f"\nFiltered to {len(cases)} cases by seed")

    if args.first_per_category:
        seen = set()
        filtered = []
        for c in cases:
            if c['category'] not in seen:
                seen.add(c['category'])
                filtered.append(c)
        cases = filtered
        print(f"\nFirst per category: {len(cases)} cases")

    if len(cases) > args.max_cases:
        cases = cases[:args.max_cases]
        print(f"Truncated to {args.max_cases} cases")

    # Generate
    print(f"\nGenerating {len(cases)} figures...")
    for i, case_data in enumerate(cases):
        cat = case_data['category'].replace('/', '_')
        seed = case_data['seed']
        filename = f"pvtet_{cat}_seed{seed}.{args.format}"
        output_path = os.path.join(args.output_dir, filename)
        print(f"[{i+1}/{len(cases)}] {case_data['category']} (seed={seed})")
        visualize_case(case_data, output_path)

    print(f"\nDone. {len(cases)} figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
