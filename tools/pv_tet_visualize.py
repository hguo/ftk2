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
import re
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
    """Read Cv position from C++ JSON (tet barycentric coords)."""
    mu = case_data.get('Cv_mu')
    if mu is None:
        return None
    mu = np.clip(mu, 0, 1)
    return bary_tet_to_3d(mu)


def compute_cw_position(case_data):
    """Read Cw position from C++ JSON (tet barycentric coords)."""
    mu = case_data.get('Cw_mu')
    if mu is None:
        return None
    mu = np.clip(mu, 0, 1)
    return bary_tet_to_3d(mu)


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
    """Draw D00, D01, Cv, Cw, SR, TN markers — merging co-located tags."""
    category = case_data['category']
    if segments is None:
        segments = []
    cv_cw_default = '#008800'  # green fallback

    # ── Collect annotations: (pos, tag, lam_str, color, bg_color,
    #    marker, marker_size, marker_color) ──
    annots = []

    # D01: punctures on tet edge
    if 'D01' in category:
        for pi in case_data['punctures']:
            if not pi.get('is_edge', False):
                continue
            pos = bary_to_3d(pi['bary'], pi['face'])
            lam = pi['lambda']
            lam_str = (f'$\\lambda={lam:.2f}$' if lam is not None
                       else r'$\lambda\!\to\!\infty$')
            annots.append(dict(pos=pos, tag='D01', lam_str=lam_str,
                               color='#dd5500', bg='#fff4ee',
                               marker='v', ms=80, mc='#dd5500'))

    # D00: vertices where V x W = 0
    if 'D00' in category:
        for vi in find_d00_vertices(case_data):
            p = TET_VERTS[vi]
            Vi = np.array(case_data['V'][vi], dtype=float)
            Wi = np.array(case_data['W'][vi], dtype=float)
            d00_lam = None
            for comp in range(3):
                if Wi[comp] != 0:
                    d00_lam = -Vi[comp] / Wi[comp]
                    break
            lam_str = (f'$\\lambda={d00_lam:.2f}$' if d00_lam is not None
                       else r'$\lambda=?$')
            annots.append(dict(pos=p, tag='D00', lam_str=lam_str,
                               color='#cc00cc', bg='white',
                               marker='s', ms=100, mc='#9900cc'))

    # Cv/Cv2/Cv1/Cv0
    cv_tag = None
    cv_m = re.search(r'(?:^|_)Cv(\d?)(?=_|$)', category)
    if cv_m:
        cv_tag = 'Cv' + cv_m.group(1)
    if cv_tag:
        pos = compute_cv_position(case_data)
        if pos is not None:
            seg_color = _find_segment_for_lambda(0.0, segments)
            cv_color = seg_color if seg_color != '#333333' else cv_cw_default
            annots.append(dict(pos=pos, tag=cv_tag,
                               lam_str=r'$\lambda=0$',
                               color=cv_color, bg='#eeffee',
                               marker='*', ms=200, mc=cv_color))

    # Cw/Cw2/Cw1/Cw0
    cw_tag = None
    cw_m = re.search(r'(?:^|_)Cw(\d?)(?=_|$)', category)
    if cw_m:
        cw_tag = 'Cw' + cw_m.group(1)
    if cw_tag:
        pos = compute_cw_position(case_data)
        if pos is not None:
            cw_color = cv_cw_default
            for seg in segments:
                if (seg.get('infinity_spanning', False) or
                    seg['lam_entry'] is None or seg['lam_exit'] is None):
                    cw_color = seg['color']
                    break
            annots.append(dict(pos=pos, tag=cw_tag,
                               lam_str=r'$\lambda\!\to\!\infty$',
                               color=cw_color, bg='#eeffee',
                               marker='*', ms=200, mc=cw_color))

    # SR/ISR
    sr_label_prefix = 'ISR' if 'ISR' in category else 'SR'
    sr_q_indices = case_data.get('sr_q_root_indices', [])
    if 'SR' in category and sr_q_indices:
        Q = case_data['Q_coeffs']
        P = case_data['P_coeffs']
        Q_roots = case_data.get('Q_roots', [])
        for si, qi in enumerate(sr_q_indices):
            if qi >= len(Q_roots):
                continue
            qr = Q_roots[qi]
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
            annots.append(dict(pos=pos, tag=sr_label_prefix,
                               lam_str=f'$\\lambda={qr:.2f}$',
                               color='#ff8800', bg='#fff8ee',
                               marker='D', ms=180, mc='#ff8800'))

    # TN: tangency
    tn_points = case_data.get('tn_points', [])
    if 'TN' in category and tn_points:
        tn_color = '#9933cc'
        for tn in tn_points:
            pos = bary_to_3d(tn['bary'], tn['face'])
            seg_color = _find_segment_for_lambda(tn['lambda'], segments)
            mc = seg_color if seg_color != '#333333' else tn_color
            annots.append(dict(pos=pos, tag='TN',
                               lam_str=f'$\\lambda={tn["lambda"]:.2f}$',
                               color=tn_color, bg='#f4eeff',
                               marker='^', ms=120, mc=mc))

    if not annots:
        return

    # ── Merge co-located annotations (distance < 0.03) ──
    groups = []
    used = [False] * len(annots)
    for i, a in enumerate(annots):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j in range(i + 1, len(annots)):
            if used[j]:
                continue
            if np.linalg.norm(np.array(a['pos']) - np.array(annots[j]['pos'])) < 0.03:
                group.append(annots[j])
                used[j] = True
        groups.append(group)

    # ── Draw each group ──
    offsets = [
        np.array([0.10, 0.08, 0.10]),
        np.array([-0.10, -0.06, 0.12]),
        np.array([0.08, -0.10, 0.10]),
        np.array([-0.08, 0.10, 0.12]),
        np.array([0.06, 0.10, 0.08]),
        np.array([-0.08, 0.06, 0.12]),
    ]
    for gi, group in enumerate(groups):
        pos = group[0]['pos']
        # Draw all markers (stacked at same pos)
        for g in group:
            ax.scatter(pos[0], pos[1], pos[2], c=g['mc'], s=g['ms'],
                       marker=g['marker'], zorder=10, edgecolors='black',
                       linewidth=1.0)
        # Build merged label: "D01+Cw1 (λ→∞)" or "SR (λ=1.23)"
        tags = '+'.join(g['tag'] for g in group)
        # Deduplicate lambda strings
        lam_strs = list(dict.fromkeys(g['lam_str'] for g in group))
        lam_combined = ', '.join(lam_strs)
        label = f'{tags}  {lam_combined}'
        color = group[0]['color']
        bg = group[0]['bg']
        off = offsets[gi % len(offsets)]
        ax.plot3D([pos[0], pos[0]+off[0]], [pos[1], pos[1]+off[1]],
                  [pos[2], pos[2]+off[2]],
                  color=color, linewidth=1.2, linestyle='-')
        ax.text(pos[0]+off[0], pos[1]+off[1], pos[2]+off[2]+0.01,
                label, fontsize=8,
                ha='left' if off[0] > 0 else 'right', va='bottom',
                color=color, fontweight='bold',
                bbox=dict(facecolor=bg, alpha=0.9,
                          edgecolor=color, linewidth=1.2,
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
    """Return (skip_marker, skip_label) sets for punctures with dedicated markers.

    skip_marker: D00/D01 punctures — drawn by draw_special_points with own markers.
    skip_label:  Cv/Cw waypoints — keep puncture marker, suppress λ-label (Cv label shown instead).
    """
    category = case_data['category']
    skip_marker = set()
    skip_label = set()
    for i, pi in enumerate(case_data['punctures']):
        if 'D01' in category and pi.get('is_edge', False):
            skip_marker.add(i)
        if 'D00' in category and pi.get('is_vertex', False):
            skip_marker.add(i)
        # Cv/Cw waypoint: keep marker (it's a real boundary crossing), skip label only
        if 'Cv' in category and pi.get('lambda') is not None and pi['lambda'] == 0.0:
            skip_label.add(i)
    return skip_marker, skip_label


def draw_puncture_markers(ax, case_data, segments):
    """Draw puncture points colored by segment, with lambda labels."""
    category = case_data['category']
    punctures = case_data['punctures']
    skip_marker, skip_label = _find_special_punctures(case_data)

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
        # Skip marker entirely for D00/D01 (drawn by draw_special_points)
        if i in skip_marker:
            continue

        pos = bary_to_3d(pi['bary'], pi['face'])
        color = punc_color.get(i, '#666666')
        if pi.get('is_vertex', False):
            marker, size = 'D', 60
        elif pi.get('is_edge', False):
            marker, size = 's', 50
        else:
            marker, size = 'o', 40

        ax.scatter(*pos, c=color, s=size, marker=marker, zorder=6,
                   edgecolors='black', linewidth=0.7)

        # Skip lambda label for Cv/Cw waypoints (Cv/Cw label shown instead)
        if i in skip_label:
            continue
        lam = pi['lambda']
        if 'Cw' in category and (lam is None or (lam is not None and abs(lam) > 1e30)):
            continue
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

    # Choose scale so all finite λ values get reasonable angular separation.
    # arctan(λ/scale) maps [-∞,+∞] to [-π/2, π/2].  Using max(|λ|) as
    # scale ensures the most extreme puncture maps to arctan(1) = 45°,
    # leaving 45° of ring between it and ∞.
    all_lams = [p['lambda'] for p in punctures if p['lambda'] is not None]
    all_lams += [r for r in Q_roots]
    abs_vals = [abs(v) for v in all_lams if abs(v) > 1e-15]
    scale = max(abs_vals) if abs_vals else 1.0
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

    # ── Collect ring annotations, merge co-located, then draw ──
    category = case_data.get('category', '')
    cv_cw_default = '#008800'
    ring_annots = []  # list of dict(angle, tag, lam_str, color, bg, marker, ms)

    # SR/ISR
    sr_ring_label = 'ISR' if 'ISR' in category else 'SR'
    sr_q_indices_ring = case_data.get('sr_q_root_indices', [])
    if 'SR' in category and sr_q_indices_ring:
        for qi in sr_q_indices_ring:
            if qi >= len(Q_roots):
                continue
            qr = Q_roots[qi]
            ring_annots.append(dict(angle=lambda_to_angle(qr, scale),
                tag=sr_ring_label, lam_str=f'$\\lambda$={qr:.2f}',
                color='#ff8800', bg='#fff4ee', marker='D', ms=9))

    # D00: vertices where V x W = 0
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
            a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
            lam_str = f'$\\lambda$={lam_d:.2f}' if lam_d is not None else r'$\lambda\!=\!\infty$'
            ring_annots.append(dict(angle=a, tag='D00', lam_str=lam_str,
                color='#9900cc', bg='#f4eeff', marker='s', ms=9))

    # D01: edge punctures
    for pi in punctures:
        if pi.get('is_vertex', False):
            continue
        if pi.get('is_edge', False):
            lam_d = pi['lambda']
            a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
            lam_str = f'$\\lambda$={lam_d:.2f}' if lam_d is not None else r'$\lambda\!\to\!\infty$'
            ring_annots.append(dict(angle=a, tag='D01', lam_str=lam_str,
                color='#9900cc', bg='#f4eeff', marker='v', ms=9))

    # Cv
    cv_tag_ring = None
    cv_mr = re.search(r'(?:^|_)Cv(\d?)(?=_|$)', category)
    if cv_mr:
        cv_tag_ring = 'Cv' + cv_mr.group(1)
    if cv_tag_ring:
        seg_color = _find_segment_for_lambda(0.0, segments)
        cv_color = seg_color if seg_color != '#333333' else cv_cw_default
        ring_annots.append(dict(angle=lambda_to_angle(0.0, scale),
            tag=cv_tag_ring, lam_str=r'$\lambda\!=\!0$',
            color=cv_color, bg='#eeffee', marker='*', ms=10))

    # Cw
    cw_tag_ring = None
    cw_mr = re.search(r'(?:^|_)Cw(\d?)(?=_|$)', category)
    if cw_mr:
        cw_tag_ring = 'Cw' + cw_mr.group(1)
    if cw_tag_ring:
        cw_color = cv_cw_default
        for seg in segments:
            if (seg.get('infinity_spanning', False) or
                seg['lam_entry'] is None or seg['lam_exit'] is None):
                cw_color = seg['color']
                break
        ring_annots.append(dict(angle=np.pi / 2,
            tag=cw_tag_ring, lam_str=r'$\lambda\!\to\!\infty$',
            color=cw_color, bg='#eeffee', marker='*', ms=10))

    # TN
    tn_ring_points = case_data.get('tn_points', [])
    if 'TN' in category and tn_ring_points:
        tn_color = '#9933cc'
        for tn in tn_ring_points:
            if any(b < -0.05 for b in tn['bary']):
                continue
            ring_annots.append(dict(angle=lambda_to_angle(tn['lambda'], scale),
                tag='TN', lam_str=f'$\\lambda$={tn["lambda"]:.2f}',
                color=tn_color, bg='#f4eeff', marker='^', ms=8))

    # ── Merge co-located ring annotations (angular distance < 0.08 rad) ──
    ring_groups = []
    r_used = [False] * len(ring_annots)
    for i, a in enumerate(ring_annots):
        if r_used[i]:
            continue
        group = [a]
        r_used[i] = True
        for j in range(i + 1, len(ring_annots)):
            if r_used[j]:
                continue
            da = abs(a['angle'] - ring_annots[j]['angle'])
            if da > np.pi:
                da = 2 * np.pi - da
            if da < 0.08:
                group.append(ring_annots[j])
                r_used[j] = True
        ring_groups.append(group)

    # ── Draw merged ring annotations ──
    label_offsets = [0.35, 0.55, 0.75, 0.45, 0.65]
    for gi, group in enumerate(ring_groups):
        a = group[0]['angle']
        # Draw all markers
        for g in group:
            sx, sy = angle_to_xy(g['angle'], R_ring)
            ax.plot(sx, sy, g['marker'], color=g['color'], markersize=g['ms'],
                    zorder=9, markeredgecolor='black', markeredgewidth=0.8)
        # Build merged label
        tags = '+'.join(g['tag'] for g in group)
        lam_strs = list(dict.fromkeys(g['lam_str'] for g in group))
        lam_combined = ', '.join(lam_strs)
        label = f'{tags} ({lam_combined})'
        color = group[0]['color']
        bg = group[0]['bg']
        lr = R_ring + label_offsets[gi % len(label_offsets)]
        lx, ly = angle_to_xy(a, lr)
        mx, my = angle_to_xy(a, R_ring + 0.06)
        ax.plot([mx, lx], [my, ly], '-', color=color, linewidth=0.8)
        ax.text(lx, ly, label,
                ha='center', va='center', fontsize=7,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor=bg,
                          edgecolor=color, linewidth=0.8))

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

    skip_marker_ring, skip_label_ring = _find_special_punctures(case_data)
    for i, pi in enumerate(punctures):
        # Skip ring marker for D00/D01 (drawn separately)
        if i in skip_marker_ring:
            continue

        lam = pi['lambda']
        color = punc_color.get(i, 'black')
        a = punc_angles[i]
        lam_str = f'{lam:.2f}' if lam is not None else r'$\infty$'

        # Tick mark on ring
        x1, y1 = angle_to_xy(a, 0.88)
        x2, y2 = angle_to_xy(a, 1.07)
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2.0, zorder=7)

        # Skip lambda label for Cv/Cw waypoints (Cv/Cw label shown instead)
        if i in skip_label_ring:
            continue
        if 'Cw' in category and (lam is None or (lam is not None and abs(lam) > 1e30)):
            continue

        if True:
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
                if Wi[comp] != 0:
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

    # Bubble: closed PV curve inside tet (0 punctures, 0 segments)
    # Must be drawn BEFORE draw_special_points so Cv/Cw picks up segment color.
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

    draw_puncture_markers(ax3d, case_data, segments)
    draw_special_points(ax3d, case_data, segments)

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

    # Determine Q degree from coefficients
    Q_coeffs = case_data.get('Q_coeffs', [0,0,0,0])
    degQ = 3
    while degQ > 0 and Q_coeffs[degQ] == 0:
        degQ -= 1
    n_qr = len(case_data.get('Q_roots', []))
    if degQ < 3:
        disc_str = f'Q degree {degQ} ({n_qr} root{"s" if n_qr != 1 else ""})'
    else:
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
