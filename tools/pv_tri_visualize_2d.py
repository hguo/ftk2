#!/usr/bin/env python3
"""
Visualize single-triangle PV cases from JSON output of pv_tri_case_finder_2d.

Three-panel figure per case:
  (a) 2D triangle with PV curve segments (colored per segment)
  (b) Lambda-ring diagram with segment bands and puncture labels
  (c) Info panel: V, W field values and Q, P polynomials

Usage:
  python3 pv_tri_visualize_2d.py cases.jsonl --output-dir figures/
  python3 pv_tri_visualize_2d.py cases.jsonl --seeds 15587 414
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

# Triangle vertices: equilateral triangle in 2D
TRI_VERTS = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2],
])

TRI_EDGES = [(0, 1), (0, 2), (1, 2)]

# Edge k is opposite vertex k:
#   Edge 0: v1-v2,  Edge 1: v2-v0,  Edge 2: v0-v1
EDGE_VERTS = [
    [1, 2],  # edge 0 (opposite v0)
    [2, 0],  # edge 1 (opposite v1)
    [0, 1],  # edge 2 (opposite v2)
]


# ─── Math helpers ────────────────────────────────────────────────────────────

def poly_eval(coeffs, x):
    """Evaluate polynomial coeffs[0] + coeffs[1]*x + ... at x (Horner)."""
    val = 0.0
    for i in range(len(coeffs) - 1, -1, -1):
        val = val * x + coeffs[i]
    return val


def bary_to_2d(bary):
    """Convert triangle barycentric coords (3 values) to 2D position."""
    return sum(bary[i] * TRI_VERTS[i] for i in range(3))


def lambda_to_bary_tri(lam, Q_coeffs, P_coeffs):
    """Convert lambda to triangle barycentric coords: mu_k = P_k(lam) / Q(lam)."""
    Q_val = poly_eval(Q_coeffs, lam)
    if abs(Q_val) < 1e-30:
        return None
    mu = np.array([poly_eval(P_coeffs[k], lam) / Q_val for k in range(3)])
    return mu


def edge_bary_to_2d(bary, edge_idx):
    """Convert edge barycentric coords (2 values) to 2D position on triangle edge."""
    ev = EDGE_VERTS[edge_idx]
    return bary[0] * TRI_VERTS[ev[0]] + bary[1] * TRI_VERTS[ev[1]]


# ─── Integer-to-float reconstruction ──────────────────────────────────────

def _solve_poly_roots(coeffs_float, degree, n_expected):
    """Compute sorted distinct real roots of quadratic/linear polynomial."""
    if degree <= 0 or n_expected == 0:
        return []

    trimmed = list(coeffs_float[:degree + 1])
    zero_roots = []
    while len(trimmed) > 1 and abs(trimmed[0]) < 0.5:
        zero_roots.append(0.0)
        trimmed = trimmed[1:]

    remaining_deg = len(trimmed) - 1
    if remaining_deg <= 0:
        return sorted(zero_roots[:n_expected])

    poly_np = [trimmed[i] for i in range(remaining_deg, -1, -1)]
    all_roots = np.roots(poly_np)

    real_roots = [r.real for r in all_roots
                  if abs(r.imag) < 1e-6 * max(1.0, abs(r.real))]
    real_roots = sorted(zero_roots + real_roots)

    if len(real_roots) <= n_expected:
        return real_roots

    deduped = [real_roots[0]]
    for r in real_roots[1:]:
        if abs(r - deduped[-1]) > 1e-8 * max(1.0, abs(deduped[-1])):
            deduped.append(r)
        else:
            deduped[-1] = (deduped[-1] + r) / 2
    return deduped[:n_expected]


def ensure_float_fields(case_data):
    """Reconstruct float fields from integer-only JSON for visualization."""
    if 'Q_coeffs' in case_data:
        return

    # ── 1. Q_coeffs and P_coeffs from i128 strings ──
    Q_coeffs = [float(int(c)) for c in case_data['Q']]
    P_coeffs = [[float(int(c)) for c in row] for row in case_data['P']]
    case_data['Q_coeffs'] = Q_coeffs
    case_data['P_coeffs'] = P_coeffs

    degQ = 2
    while degQ > 0 and abs(Q_coeffs[degQ]) < 0.5:
        degQ -= 1
    case_data['degQ'] = degQ

    # ── 2. Q_roots ──
    expected_n = case_data.get('n_Q_roots', 0)
    if degQ >= 1:
        poly_np = [Q_coeffs[i] for i in range(degQ, -1, -1)]
        all_roots = np.roots(poly_np)
        Q_roots = sorted([r.real for r in all_roots
                          if abs(r.imag) < 1e-6 * max(1.0, abs(r.real))])
        if len(Q_roots) > expected_n:
            deduped = [Q_roots[0]]
            for r in Q_roots[1:]:
                if abs(r - deduped[-1]) > 1e-8 * max(1.0, abs(deduped[-1])):
                    deduped.append(r)
                else:
                    deduped[-1] = (deduped[-1] + r) / 2
            Q_roots = deduped[:expected_n]
        elif len(Q_roots) < expected_n:
            Q_roots = sorted([r.real for r in all_roots
                              if abs(r.imag) < 0.1 * max(1.0, abs(r.real))])
            Q_roots = Q_roots[:expected_n]
    else:
        Q_roots = []
    case_data['Q_roots'] = Q_roots

    # ── 3. Compute P[k] roots (face roots) ──
    face_roots = {}
    for k in range(3):
        pk = P_coeffs[k]
        deg_pk = 2
        while deg_pk > 0 and abs(pk[deg_pk]) < 0.5:
            deg_pk -= 1
        if deg_pk == 2:
            disc = pk[1]**2 - 4 * pk[2] * pk[0]
            if disc > 0:
                n_exp = 2
            elif disc == 0:
                n_exp = 1
            else:
                n_exp = 0
        elif deg_pk == 1:
            n_exp = 1
        else:
            n_exp = 0
        face_roots[k] = _solve_poly_roots(pk, deg_pk, n_exp)
    case_data['face_roots'] = face_roots

    # ── 4. Puncture lambda and bary ──
    n_qr = len(Q_roots)
    for pi in case_data.get('punctures', []):
        f = pi['face']
        ri = pi['root_idx']

        if ri < 0:
            # Infinity puncture
            pi['lambda'] = None
            if degQ >= 1 and abs(Q_coeffs[degQ]) > 0.5:
                ev = EDGE_VERTS[f]
                pi['bary_tri'] = [P_coeffs[ev[j]][degQ] / Q_coeffs[degQ]
                                  for j in range(2)]
                # Full triangle bary
                mu = [P_coeffs[k][degQ] / Q_coeffs[degQ] for k in range(3)]
                pi['bary'] = mu
            else:
                pi['bary_tri'] = [0.5, 0.5]
                pi['bary'] = [0.33, 0.33, 0.34]
        elif ri < len(face_roots.get(f, [])):
            lam = face_roots[f][ri]
            pi['lambda'] = lam
            Q_val = poly_eval(Q_coeffs, lam)
            if abs(Q_val) > 1e-30:
                mu = [poly_eval(P_coeffs[k], lam) / Q_val for k in range(3)]
                pi['bary'] = mu
            else:
                # At Q root (SR): L'Hopital
                Q_prime = [Q_coeffs[j] * j for j in range(1, len(Q_coeffs))]
                Qp_val = poly_eval(Q_prime, lam)
                if abs(Qp_val) > 1e-30:
                    mu = [poly_eval([P_coeffs[k][j] * j
                                     for j in range(1, len(P_coeffs[k]))],
                                    lam) / Qp_val
                          for k in range(3)]
                    pi['bary'] = mu
                else:
                    pi['bary'] = [0.33, 0.33, 0.34]
        else:
            pi['lambda'] = 0.0
            pi['bary'] = [0.33, 0.33, 0.34]

    # ── 5. Build intervals from Q roots ──
    intervals = []
    if n_qr == 0:
        intervals.append({'lb': None, 'ub': None, 'is_infinity': True})
    elif n_qr == 1:
        intervals.append({'lb': None, 'ub': Q_roots[0], 'is_infinity': False})
        intervals.append({'lb': Q_roots[0], 'ub': None, 'is_infinity': False})
    else:  # n_qr == 2
        intervals.append({'lb': None, 'ub': Q_roots[0], 'is_infinity': False})
        intervals.append({'lb': Q_roots[0], 'ub': Q_roots[1], 'is_infinity': False})
        intervals.append({'lb': Q_roots[1], 'ub': None, 'is_infinity': False})
    # Mark infinity interval
    if case_data.get('merge_infinity', False) and n_qr >= 2:
        intervals[0]['is_infinity'] = True
        intervals[-1]['is_infinity'] = True
    elif n_qr == 0:
        intervals[0]['is_infinity'] = True
    elif n_qr == 1:
        # Both intervals span to infinity
        intervals[0]['is_infinity'] = True
        intervals[1]['is_infinity'] = True
    else:
        intervals[0]['is_infinity'] = True
        intervals[-1]['is_infinity'] = True
    case_data['intervals'] = intervals


# ─── Lambda-ring mapping ────────────────────────────────────────────────────

def lambda_to_angle(lam, scale=1.0):
    """Map lambda to angle on ring. lambda=0 -> -pi/2 (bottom), inf -> pi/2 (top)."""
    theta = 2.0 * np.arctan(lam / scale)
    return theta - np.pi / 2


def angle_to_xy(angle, radius=1.0):
    return radius * np.cos(angle), radius * np.sin(angle)


# ─── Curve sampling ─────────────────────────────────────────────────────────

def sample_pv_curve(Q_coeffs, P_coeffs, lam_lo, lam_hi, is_infinity,
                    n_samples=300, puncture_lambdas=None):
    """Sample PV curve in an interval.
    Returns list of (pts_2d_array, lam_entry, lam_exit) tuples."""
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
        if lam_lo is not None and lam_hi is not None:
            mid = np.linspace(lam_lo, lam_hi, n_samples)
            eps = (lam_hi - lam_lo) * 0.01
            near_lo = np.linspace(lam_lo, lam_lo + eps, 10)
            near_hi = np.linspace(lam_hi - eps, lam_hi, 10)
            lam_vals = np.sort(np.unique(np.concatenate([mid, near_lo, near_hi])))
        else:
            t = np.linspace(-0.499 * np.pi, 0.499 * np.pi, n_samples)
            lam_vals = np.tan(t)

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
        for _ in range(n_bisect):
            lam_mid = 0.5 * (lam_in + lam_out)
            mu_mid = lambda_to_bary_tri(lam_mid, Q_coeffs, P_coeffs)
            if is_inside(mu_mid):
                lam_in = lam_mid
            else:
                lam_out = lam_mid
        mu = lambda_to_bary_tri(lam_in, Q_coeffs, P_coeffs)
        return bary_to_2d(np.clip(mu, 0, 1)), lam_in

    segments = []
    current = []
    seg_lam_entry = None
    seg_lam_exit = None
    prev_lam = None
    prev_inside = False

    for lam in lam_vals:
        mu = lambda_to_bary_tri(lam, Q_coeffs, P_coeffs)
        inside = is_inside(mu)
        if inside:
            if not prev_inside and prev_lam is not None:
                pt, bl = bisect_boundary(lam, prev_lam)
                current.append(pt)
                seg_lam_entry = bl
            elif not prev_inside:
                seg_lam_entry = lam
            current.append(bary_to_2d(mu))
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
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1e15
    return abs(a - b)


def _match_to_pair(lam_e, lam_x, pairs, punctures):
    best_pair = 0
    best_dist = float('inf')
    for idx, (pi1, pi2) in enumerate(pairs):
        l1 = punctures[pi1].get('lambda')
        l2 = punctures[pi2].get('lambda')
        d = min(_lam_dist(lam_e, l1), _lam_dist(lam_e, l2),
                _lam_dist(lam_x, l1), _lam_dist(lam_x, l2))
        if d < best_dist:
            best_dist = d
            best_pair = idx
    return best_pair


def collect_segments(case_data):
    """Build segments from pre-computed puncture pairing."""
    Q = case_data['Q_coeffs']
    P = case_data['P_coeffs']
    intervals = case_data['intervals']
    punctures = case_data['punctures']
    puncture_lambdas = [p.get('lambda') for p in punctures]

    pairs = case_data.get('pairs', [])

    all_subsegs = []
    for iv in intervals:
        segs = sample_pv_curve(Q, P, iv['lb'], iv['ub'], iv['is_infinity'],
                               puncture_lambdas=puncture_lambdas)
        all_subsegs.extend(segs)

    pair_subsegs = defaultdict(list)
    for pts, lam_e, lam_x in all_subsegs:
        if pairs:
            best = _match_to_pair(lam_e, lam_x, pairs, punctures)
            pair_subsegs[best].append(pts)
        else:
            pair_subsegs[0].append(pts)

    segments = []
    for idx, (pi1, pi2) in enumerate(pairs):
        color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
        l1 = punctures[pi1].get('lambda')
        l2 = punctures[pi2].get('lambda')
        inf_span = case_data.get('merge_infinity', False) and (l1 is None or l2 is None)
        segments.append({
            'pts_list': pair_subsegs.get(idx, []),
            'color': color,
            'pi_entry': pi1,
            'pi_exit': pi2,
            'lam_entry': l1,
            'lam_exit': l2,
            'infinity_spanning': inf_span,
        })

    # Unpaired sub-segments: only create catch-all for bubble (T0 with curve)
    if not pairs and all_subsegs:
        n_punc = len(punctures)
        if n_punc == 0:
            # True bubble: closed curve inside triangle, no edge crossings
            segments = [{
                'pts_list': [pts for pts, _, _ in all_subsegs],
                'color': SEGMENT_COLORS[0],
                'pi_entry': -1,
                'pi_exit': -1,
                'lam_entry': None,
                'lam_exit': None,
                'infinity_spanning': True,
            }]
        # else: unpaired punctures (T1/T3 waypoint cases) — no segments

    return segments


# ─── Formatting helpers ─────────────────────────────────────────────────────

def poly_to_latex(coeffs, name='Q'):
    """Format polynomial as matplotlib mathtext string."""
    sup = {2: '2', 3: '3'}
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

        if not terms:
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


# ─── 2D Triangle panel ───────────────────────────────────────────────────────

def find_d00_vertices(case_data):
    """Find triangle vertices where det(V, W) = 0 (D00 degeneracy)."""
    V = np.array(case_data['V'], dtype=float)
    W = np.array(case_data['W'], dtype=float)
    d00 = []
    for i in range(3):
        det = V[i][0] * W[i][1] - V[i][1] * W[i][0]
        if abs(det) < 1e-10 * max(np.linalg.norm(V[i]) * np.linalg.norm(W[i]), 1e-30):
            d00.append(i)
    return d00


def find_d11_edges(case_data):
    """Find triangle edges where both endpoints have det(V,W)=0 at the same lambda."""
    V = np.array(case_data['V'], dtype=float)
    W = np.array(case_data['W'], dtype=float)

    pv_info = {}
    for i in range(3):
        det = V[i][0] * W[i][1] - V[i][1] * W[i][0]
        if abs(det) > 1e-10 * max(np.linalg.norm(V[i]) * np.linalg.norm(W[i]), 1e-30):
            continue
        v_zero = np.allclose(V[i], 0, atol=1e-15)
        w_zero = np.allclose(W[i], 0, atol=1e-15)
        if v_zero and w_zero:
            pv_info[i] = (0, 0, True)
        elif v_zero:
            pv_info[i] = (0, 1, False)
        elif w_zero:
            pv_info[i] = (1, 0, False)
        else:
            for k in range(2):
                if abs(W[i][k]) > 1e-15:
                    pv_info[i] = (-V[i][k], W[i][k], False)
                    break

    d11_edges = []
    for edge_idx, (vi, vj) in enumerate(EDGE_VERTS):
        if vi not in pv_info or vj not in pv_info:
            continue
        n1, d1, any1 = pv_info[vi]
        n2, d2, any2 = pv_info[vj]
        if any1 or any2:
            lam = n2 / d2 if d2 != 0 and not any2 else (n1 / d1 if d1 != 0 else 0.0)
            d11_edges.append((vi, vj, edge_idx, lam))
        elif d1 != 0 and d2 != 0 and abs(n1 * d2 - n2 * d1) < 1e-10:
            d11_edges.append((vi, vj, edge_idx, n1 / d1))
        elif d1 == 0 and d2 == 0:
            d11_edges.append((vi, vj, edge_idx, float('inf')))
    return d11_edges


def _find_segment_for_lambda(lam, segments):
    """Find the segment whose lambda range contains lam. Return its color."""
    for seg in segments:
        l1 = seg['lam_entry']
        l2 = seg['lam_exit']
        if seg.get('infinity_spanning', False):
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


def draw_tri_wireframe(ax):
    """Draw triangle wireframe with semi-transparent fill."""
    tri = plt.Polygon(TRI_VERTS, fill=True, facecolor='#f0f0f0',
                      edgecolor='#555555', linewidth=1.5, alpha=0.4)
    ax.add_patch(tri)

    offsets = [
        np.array([-0.06, -0.06]),
        np.array([0.06, -0.06]),
        np.array([0.0, 0.06]),
    ]
    for i in range(3):
        p = TRI_VERTS[i] + offsets[i]
        ax.text(p[0], p[1], f'$v_{i}$',
                fontsize=10, color='#333333', ha='center', va='center')


def _find_field_zero_2d(F):
    """Find where F(x) = mu_0*F_0 + mu_1*F_1 + mu_2*F_2 = 0 in triangle.
    Handles degenerate cases (collinear/parallel vectors).
    Returns 2D position or None."""
    F = np.array(F, dtype=float)

    # Check vertex zeros first
    for i in range(3):
        if np.allclose(F[i], 0, atol=0.5):
            return bary_to_2d(np.eye(3)[i])

    # Cramer's rule: mu_0(F_0-F_2) + mu_1(F_1-F_2) = -F_2
    A = np.column_stack([F[0] - F[2], F[1] - F[2]])
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) > 1e-14:
        rhs = -F[2]
        mu0 = (rhs[0] * A[1, 1] - rhs[1] * A[0, 1]) / det
        mu1 = (A[0, 0] * rhs[1] - A[1, 0] * rhs[0]) / det
        mu2 = 1 - mu0 - mu1
        mu = np.array([mu0, mu1, mu2])
        if np.all(mu >= -1e-6):
            return bary_to_2d(np.clip(mu, 0, 1))
        return None

    # Degenerate: all F vectors are parallel (det=0).
    # Project onto the direction of any nonzero F vector.
    d = None
    for i in range(3):
        if np.linalg.norm(F[i]) > 1e-10:
            d = F[i] / np.linalg.norm(F[i])
            break
    if d is None:
        # All F vectors are zero
        return bary_to_2d(np.array([1/3, 1/3, 1/3]))

    projs = [np.dot(F[i], d) for i in range(3)]
    # Check each edge for sign change (antiparallel endpoints)
    for edge_k, (vi, vj) in enumerate(EDGE_VERTS):
        if projs[vi] * projs[vj] < 0:
            # Zero crossing on this edge
            t = projs[vi] / (projs[vi] - projs[vj])
            mu = np.zeros(3)
            mu[vi] = 1 - t
            mu[vj] = t
            return bary_to_2d(mu)
    return None


def compute_cv_position(case_data):
    """Compute Cv position (V(x)=0) in triangle."""
    return _find_field_zero_2d(case_data['V'])


def compute_cw_position(case_data):
    """Compute Cw position (W(x)=0) in triangle."""
    return _find_field_zero_2d(case_data['W'])


def draw_vector_arrows(ax, case_data, arrow_scale=0.08):
    """Draw V (red) and W (blue) arrows at triangle vertices."""
    V = np.array(case_data['V'], dtype=float)
    W = np.array(case_data['W'], dtype=float)
    v_max = max(np.max(np.abs(V)), 1e-10)
    w_max = max(np.max(np.abs(W)), 1e-10)
    for i in range(3):
        origin = TRI_VERTS[i]
        v_dir = V[i] / v_max * arrow_scale
        w_dir = W[i] / w_max * arrow_scale
        ax.annotate('', xy=origin + v_dir, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='#cc0000',
                                    lw=1.5, mutation_scale=10))
        ax.annotate('', xy=origin + w_dir, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='#0044cc',
                                    lw=1.5, mutation_scale=10))


def draw_pv_curves(ax, segments):
    """Draw PV curve segments, each with its own color."""
    for seg in segments:
        for pts in seg['pts_list']:
            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1],
                        color=seg['color'], linewidth=2.5, alpha=0.9, zorder=5)


def draw_special_points(ax, case_data, segments=None):
    """Draw D00, D01, D11, Cv, Cw, SR, TN markers — merging co-located tags."""
    category = case_data['category']
    if segments is None:
        segments = []
    cv_cw_default = '#008800'

    annots = []

    # D01: punctures on triangle edge interior (is_edge=True but NOT is_D00)
    # D00 punctures are handled by the separate D00 block below
    for pi in case_data['punctures']:
        if not pi.get('is_edge', False) or pi.get('is_D00', False):
            continue
        bary = pi.get('bary', [0.33, 0.33, 0.34])
        pos = bary_to_2d(bary)
        lam = pi.get('lambda')
        lam_str = (f'$\\lambda={lam:.2f}$' if lam is not None
                   else r'$\lambda\!\to\!\infty$')
        annots.append(dict(pos=pos, tag='D01', lam_str=lam_str,
                           color='#dd5500', bg='#fff4ee',
                           marker='v', ms=80, mc='#dd5500'))

    # D00: vertices where det(V,W) = 0
    # Skip vertices already tagged as Cv0 or Cw0 to avoid duplicate markers
    cv_type = case_data.get('Cv', 0)
    cw_type = case_data.get('Cw', 0)
    if 'D00' in category:
        for vi in find_d00_vertices(case_data):
            # Skip if this vertex is a Cv0 or Cw0 (already drawn with its own marker)
            Vi = np.array(case_data['V'][vi], dtype=float)
            Wi = np.array(case_data['W'][vi], dtype=float)
            v_zero = np.allclose(Vi, 0, atol=0.5)
            w_zero = np.allclose(Wi, 0, atol=0.5)
            if v_zero and cv_type == 3:
                continue  # Skip, Cv0 marker will be drawn
            if w_zero and cw_type == 3:
                continue  # Skip, Cw0 marker will be drawn

            p = TRI_VERTS[vi]
            d00_lam = None
            if w_zero:
                lam_str = r'$\lambda\!\to\!\infty$'
            elif v_zero:
                d00_lam = 0.0
                lam_str = r'$\lambda=0$'
            else:
                for comp in range(2):
                    if Wi[comp] != 0:
                        d00_lam = -Vi[comp] / Wi[comp]
                        break
                lam_str = (f'$\\lambda={d00_lam:.2f}$' if d00_lam is not None
                           else r'$\lambda\!\to\!\infty$')
            annots.append(dict(pos=p, tag='D00', lam_str=lam_str,
                               color='#cc00cc', bg='white',
                               marker='s', ms=100, mc='#9900cc'))

    # Cv/Cv1/Cv0
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

    # Cw/Cw1/Cw0
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
    sr_label = 'ISR' if 'ISR' in category else 'SR'
    if 'SR' in category:
        Q = case_data['Q_coeffs']
        P = case_data['P_coeffs']
        Q_roots = case_data.get('Q_roots', [])
        for qi, qr in enumerate(Q_roots):
            # Check if this is a shared root
            for k in range(3):
                pk_val = poly_eval(P[k], qr)
                if abs(pk_val) < 1e-6 * max(1.0, abs(qr)):
                    # Shared root — compute position via L'Hopital
                    Q_prime = [Q[j] * j for j in range(1, len(Q))]
                    Qp_val = poly_eval(Q_prime, qr)
                    if abs(Qp_val) < 1e-20:
                        break
                    mu = np.array([
                        poly_eval([P[k2][j] * j for j in range(1, len(P[k2]))],
                                  qr) / Qp_val
                        for k2 in range(3)
                    ])
                    mu_clip = np.clip(mu, 0, 1)
                    s = mu_clip.sum()
                    if s > 1e-10:
                        mu_clip /= s
                    pos = bary_to_2d(mu_clip)
                    annots.append(dict(pos=pos, tag=sr_label,
                                       lam_str=f'$\\lambda={qr:.2f}$',
                                       color='#ff8800', bg='#fff8ee',
                                       marker='D', ms=120, mc='#ff8800'))
                    break

    # D11: curve on edge
    if 'D11' in category:
        d11_edges = find_d11_edges(case_data)
        d11_color = '#cc6600'
        for vi, vj, eidx, lam in d11_edges:
            p1, p2 = TRI_VERTS[vi], TRI_VERTS[vj]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=d11_color,
                    linewidth=4.0, alpha=0.9, zorder=10)
            mid = (p1 + p2) / 2
            lam_str = (f'$\\lambda={lam:.2f}$' if not np.isinf(lam)
                       else r'$\lambda\!\to\!\infty$')
            annots.append(dict(pos=mid, tag='D11', lam_str=lam_str,
                               color=d11_color, bg='#fff4ee',
                               marker='h', ms=100, mc=d11_color))

    # TN: tangency — compute from P[k] discriminant
    if 'TN' in category:
        P = case_data.get('P_coeffs', case_data.get('P'))
        Q = case_data.get('Q_coeffs', case_data.get('Q'))
        if P and Q:
            P_f = [[float(int(c)) for c in pk] if isinstance(pk[0], str) else pk
                   for pk in P]
            Q_f = [float(int(c)) for c in Q] if isinstance(Q[0], str) else Q
            tn_color = '#9933cc'
            for k in range(3):
                pk = P_f[k]
                if len(pk) < 3 or abs(pk[2]) < 0.5:
                    continue
                disc = pk[1]**2 - 4 * pk[2] * pk[0]
                if abs(disc) > 1e-6 * max(abs(pk[1]**2), abs(4*pk[2]*pk[0]), 1):
                    continue  # Not a double root
                lam = -pk[1] / (2 * pk[2])
                Q_val = poly_eval(Q_f, lam)
                Ppp = 2 * pk[2]  # P''[k] is constant
                if Ppp * Q_val <= 0:
                    continue  # Isolated tangency, skip
                # Non-isolated tangency: compute position
                mu = [poly_eval(P_f[j], lam) / Q_val if abs(Q_val) > 1e-30
                      else 0.33 for j in range(3)]
                if all(m >= -0.05 for m in mu):
                    pos = bary_to_2d(np.clip(mu, 0, 1))
                    seg_color = _find_segment_for_lambda(lam, segments)
                    mc = seg_color if seg_color != '#333333' else tn_color
                    annots.append(dict(pos=pos, tag='TN',
                                       lam_str=f'$\\lambda={lam:.2f}$',
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

    offsets = [
        np.array([0.10, 0.10]),
        np.array([-0.10, 0.10]),
        np.array([0.10, -0.08]),
        np.array([-0.10, -0.08]),
        np.array([0.0, 0.12]),
        np.array([0.0, -0.10]),
    ]
    for gi, group in enumerate(groups):
        pos = np.array(group[0]['pos'])
        for g in group:
            ax.scatter(pos[0], pos[1], c=g['mc'], s=g['ms'],
                       marker=g['marker'], zorder=10, edgecolors='black',
                       linewidth=1.0)
        tags = '+'.join(g['tag'] for g in group)
        lam_strs = list(dict.fromkeys(g['lam_str'] for g in group))
        lam_combined = ', '.join(lam_strs)
        label = f'{tags}  {lam_combined}'
        color = group[0]['color']
        bg = group[0]['bg']
        off = offsets[gi % len(offsets)]
        lp = pos + off
        ax.plot([pos[0], lp[0]], [pos[1], lp[1]],
                color=color, linewidth=1.0, linestyle='-')
        ax.text(lp[0], lp[1] + 0.01, label, fontsize=8,
                ha='left' if off[0] > 0 else 'right', va='bottom',
                color=color, fontweight='bold',
                bbox=dict(facecolor=bg, alpha=0.9,
                          edgecolor=color, linewidth=1.2,
                          boxstyle='round,pad=0.3'))


def draw_puncture_markers(ax, case_data, segments):
    """Draw puncture points colored by segment, with lambda labels."""
    category = case_data['category']
    punctures = case_data['punctures']

    skip_marker = set()
    skip_label = set()
    for i, pi in enumerate(punctures):
        if pi.get('is_edge', False) or pi.get('is_D00', False):
            skip_marker.add(i)
        if 'Cv' in category and pi.get('lambda') is not None and pi['lambda'] == 0.0:
            skip_label.add(i)

    punc_color = {}
    for seg in segments:
        if seg['pi_entry'] >= 0:
            punc_color.setdefault(seg['pi_entry'], seg['color'])
        if seg['pi_exit'] >= 0:
            punc_color.setdefault(seg['pi_exit'], seg['color'])

    label_offsets = [
        np.array([0.0, 0.06]),
        np.array([0.06, 0.03]),
        np.array([-0.06, 0.03]),
        np.array([0.0, -0.06]),
    ]

    for i, pi in enumerate(punctures):
        if i in skip_marker:
            continue
        bary = pi.get('bary', [0.33, 0.33, 0.34])
        pos = bary_to_2d(bary)
        color = punc_color.get(i, '#666666')
        marker, size = 'o', 50

        ax.scatter(pos[0], pos[1], c=color, s=size, marker=marker, zorder=6,
                   edgecolors='black', linewidth=0.7)

        if i in skip_label:
            continue
        lam = pi.get('lambda')
        if 'Cw' in category and (lam is None or (lam is not None and abs(lam) > 1e30)):
            continue
        lam_str = f'$\\lambda$={lam:.2f}' if lam is not None else r'$\lambda=\infty$'
        off = label_offsets[i % len(label_offsets)]
        lp = pos + off
        ax.plot([pos[0], lp[0]], [pos[1], lp[1]],
                color=color, linewidth=0.8, alpha=0.6)
        ax.text(lp[0], lp[1], lam_str,
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

    all_lams = [p.get('lambda') for p in punctures if p.get('lambda') is not None]
    all_lams += [r for r in Q_roots]
    abs_vals = [abs(v) for v in all_lams if abs(v) > 1e-15]
    scale = max(abs_vals) if abs_vals else 1.0
    scale = max(scale, 0.5)

    R_ring = 1.0

    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_ring * np.cos(th), R_ring * np.sin(th),
            color='#333333', linewidth=1.5, zorder=2)

    ax.text(0, R_ring + 0.10, r'$\infty$', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#333333')

    a0 = lambda_to_angle(0.0, scale)
    x0, y0 = angle_to_xy(a0, R_ring)
    ax.plot(x0, y0, '|', color='#666666', markersize=6, markeredgewidth=1.5,
            zorder=5)
    ax.text(x0, y0 - 0.12, r'$0$', ha='center', va='top',
            fontsize=9, color='#666666')

    r_inner = 0.85
    r_outer = 0.98

    def _draw_band(a_start, a_end, color):
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
                             edgecolor=color, linewidth=0.5, zorder=1))

    for seg in segments:
        lam1 = seg['lam_entry']
        lam2 = seg['lam_exit']
        a1 = lambda_to_angle(lam1, scale) if lam1 is not None else np.pi / 2
        a2 = lambda_to_angle(lam2, scale) if lam2 is not None else np.pi / 2

        if seg.get('infinity_spanning', False):
            a_hi = max(a1, a2)
            a_lo = min(a1, a2)
            _draw_band(a_hi, a_lo + 2 * np.pi, seg['color'])
        else:
            if a1 > a2:
                a1, a2 = a2, a1
            _draw_band(a1, a2, seg['color'])

    # Q roots
    q_label_r = 0.60
    q_label_radii = [q_label_r] * len(Q_roots)
    q_angles = [lambda_to_angle(r, scale) for r in Q_roots]
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
        lr = q_label_radii[i]
        lx, ly = angle_to_xy(a, lr)
        mx, my = angle_to_xy(a, R_ring - 0.06)
        ax.plot([mx, lx], [my, ly], '-', color='#888888', linewidth=0.6)
        ax.text(lx, ly, f'{r:.2f}',
                ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#cccccc', linewidth=0.5))

    # ── Ring annotations ──
    category = case_data.get('category', '')
    cv_cw_default = '#008800'
    ring_annots = []

    # SR/ISR
    sr_ring_label = 'ISR' if 'ISR' in category else 'SR'
    if 'SR' in category:
        for qi, qr in enumerate(Q_roots):
            P = case_data['P_coeffs']
            for k in range(3):
                if abs(poly_eval(P[k], qr)) < 1e-6 * max(1.0, abs(qr)):
                    ring_annots.append(dict(angle=lambda_to_angle(qr, scale),
                        tag=sr_ring_label, lam_str=f'$\\lambda$={qr:.2f}',
                        color='#ff8800', bg='#fff4ee', marker='D', ms=9))
                    break

    # D00
    V = np.array(case_data['V'])
    W = np.array(case_data['W'])
    cv_type_r = case_data.get('Cv', 0)
    cw_type_r = case_data.get('Cw', 0)
    for vi in range(3):
        det_vw = V[vi][0] * W[vi][1] - V[vi][1] * W[vi][0]
        if abs(det_vw) < 1e-10 * max(np.linalg.norm(V[vi]) * np.linalg.norm(W[vi]), 1e-30):
            v_zero = np.all(V[vi] == 0)
            w_zero = np.all(W[vi] == 0)
            # Skip if already tagged as Cv0 or Cw0
            if v_zero and cv_type_r == 3:
                continue
            if w_zero and cw_type_r == 3:
                continue
            if w_zero:
                a = np.pi / 2
                lam_str = r'$\lambda\!\to\!\infty$'
            elif v_zero:
                a = lambda_to_angle(0.0, scale)
                lam_str = r'$\lambda\!=\!0$'
            else:
                lam_d = None
                for k in range(2):
                    if W[vi][k] != 0:
                        lam_d = -V[vi][k] / W[vi][k]
                        break
                a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
                lam_str = (f'$\\lambda$={lam_d:.2f}' if lam_d is not None
                           else r'$\lambda\!=\!\infty$')
            ring_annots.append(dict(angle=a, tag='D00', lam_str=lam_str,
                color='#9900cc', bg='#f4eeff', marker='s', ms=9))

    # D01
    for pi in punctures:
        if pi.get('is_edge', False) and not pi.get('is_D00', False):
            lam_d = pi.get('lambda')
            a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
            lam_str = (f'$\\lambda$={lam_d:.2f}' if lam_d is not None
                       else r'$\lambda\!\to\!\infty$')
            ring_annots.append(dict(angle=a, tag='D01', lam_str=lam_str,
                color='#9900cc', bg='#f4eeff', marker='v', ms=9))

    # D11
    if 'D11' in category:
        d11_edges = find_d11_edges(case_data)
        for vi, vj, eidx, lam in d11_edges:
            a = lambda_to_angle(lam, scale) if not np.isinf(lam) else np.pi / 2
            lam_str = (f'$\\lambda$={lam:.2f}' if not np.isinf(lam)
                       else r'$\lambda\!\to\!\infty$')
            ring_annots.append(dict(angle=a, tag='D11', lam_str=lam_str,
                color='#cc6600', bg='#fff4ee', marker='h', ms=9))

    # TN on ring
    if 'TN' in category:
        P = case_data.get('P_coeffs', case_data.get('P'))
        Q = case_data.get('Q_coeffs', case_data.get('Q'))
        if P and Q:
            P_f = [[float(int(c)) for c in pk] if isinstance(pk[0], str) else pk
                   for pk in P]
            Q_f = [float(int(c)) for c in Q] if isinstance(Q[0], str) else Q
            tn_color = '#9933cc'
            for k in range(3):
                pk = P_f[k]
                if len(pk) < 3 or abs(pk[2]) < 0.5:
                    continue
                disc = pk[1]**2 - 4 * pk[2] * pk[0]
                if abs(disc) > 1e-6 * max(abs(pk[1]**2), abs(4*pk[2]*pk[0]), 1):
                    continue
                lam = -pk[1] / (2 * pk[2])
                Q_val = poly_eval(Q_f, lam)
                Ppp = 2 * pk[2]
                if Ppp * Q_val <= 0:
                    continue
                ring_annots.append(dict(angle=lambda_to_angle(lam, scale),
                    tag='TN', lam_str=f'$\\lambda$={lam:.2f}',
                    color=tn_color, bg='#f4eeff', marker='^', ms=8))

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

    # Merge co-located ring annotations
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

    label_offsets = [0.35, 0.55, 0.75, 0.45, 0.65]
    for gi, group in enumerate(ring_groups):
        a = group[0]['angle']
        for g in group:
            sx, sy = angle_to_xy(g['angle'], R_ring)
            ax.plot(sx, sy, g['marker'], color=g['color'], markersize=g['ms'],
                    zorder=9, markeredgecolor='black', markeredgewidth=0.8)
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

    # ── Puncture ticks ──
    punc_color = {}
    for seg in segments:
        if seg['pi_entry'] >= 0:
            punc_color.setdefault(seg['pi_entry'], seg['color'])
        if seg['pi_exit'] >= 0:
            punc_color.setdefault(seg['pi_exit'], seg['color'])

    punc_angles = []
    for i, pi in enumerate(punctures):
        lam = pi.get('lambda')
        if lam is None:
            punc_angles.append(np.pi / 2)
        else:
            punc_angles.append(lambda_to_angle(lam, scale))

    label_radii = [R_ring + 0.35] * len(punctures)
    sorted_idx = sorted(range(len(punctures)), key=lambda i: punc_angles[i])
    for j in range(1, len(sorted_idx)):
        i_prev = sorted_idx[j - 1]
        i_curr = sorted_idx[j]
        if abs(punc_angles[i_curr] - punc_angles[i_prev]) < 0.25:
            label_radii[i_curr] = (R_ring + 0.55
                                   if label_radii[i_prev] < R_ring + 0.50
                                   else R_ring + 0.35)

    skip_marker_ring = set()
    skip_label_ring = set()
    for i, pi in enumerate(punctures):
        if pi.get('is_edge', False) or pi.get('is_D00', False):
            skip_marker_ring.add(i)
        if 'Cv' in category and pi.get('lambda') is not None and pi['lambda'] == 0.0:
            skip_label_ring.add(i)

    for i, pi in enumerate(punctures):
        if i in skip_marker_ring:
            continue
        lam = pi.get('lambda')
        color = punc_color.get(i, 'black')
        a = punc_angles[i]
        lam_str = f'{lam:.2f}' if lam is not None else r'$\infty$'

        x1, y1 = angle_to_xy(a, 0.88)
        x2, y2 = angle_to_xy(a, 1.07)
        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=2.0, zorder=7)

        if i in skip_label_ring:
            continue
        if 'Cw' in category and (lam is None or (lam is not None and abs(lam) > 1e30)):
            continue

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

    mat_lines = []
    for i in range(3):
        vr = '[' + ', '.join(f'{v:4d}' for v in V[i]) + ']'
        wr = '[' + ', '.join(f'{v:4d}' for v in W[i]) + ']'
        pv = 'V = ' if i == 0 else '    '
        pw = '    W = ' if i == 0 else '        '
        mat_lines.append(f'{pv}{vr}{pw}{wr}')

    ax.text(0.01, 0.95, '\n'.join(mat_lines), fontsize=8,
            fontfamily='monospace', transform=ax.transAxes, va='top')

    category = case_data['category']
    seg_lines = ['Segments:']
    for i, seg in enumerate(segments):
        l1 = seg['lam_entry']
        l2 = seg['lam_exit']
        if l1 is None and l2 is None:
            # Bubble or full-range curve
            seg_lines.append(f'  S{i+1}: closed curve (all lambda)')
        elif seg.get('infinity_spanning', False):
            if l1 is not None and l2 is not None:
                # Both finite, wrapping through infinity
                lo, hi = min(l1, l2), max(l1, l2)
                seg_lines.append(f'  S{i+1}: ({hi:.3f}, +inf) U (-inf, {lo:.3f})')
            elif l1 is not None:
                seg_lines.append(f'  S{i+1}: ({l1:.3f}, +/-inf)')
            elif l2 is not None:
                seg_lines.append(f'  S{i+1}: ({l2:.3f}, +/-inf)')
            else:
                seg_lines.append(f'  S{i+1}: closed curve (all lambda)')
        elif l1 is None or l2 is None:
            fin = l1 if l2 is None else l2
            fin_s = f'{fin:.3f}' if fin is not None else '?'
            seg_lines.append(f'  S{i+1}: ({fin_s}, inf)')
        else:
            lo_s = f'{min(l1, l2):.3f}'
            hi_s = f'{max(l1, l2):.3f}'
            seg_lines.append(f'  S{i+1}: ({lo_s}, {hi_s})')

    if 'D00' in category:
        for vi in find_d00_vertices(case_data):
            Wi = np.array(case_data['W'][vi], dtype=float)
            Vi = np.array(case_data['V'][vi], dtype=float)
            w_zero = np.allclose(Wi, 0, atol=0.5)
            v_zero = np.allclose(Vi, 0, atol=0.5)
            if w_zero:
                lam_s = '→∞'
            elif v_zero:
                lam_s = '=0.000'
            else:
                d00_lam = None
                for comp in range(2):
                    if Wi[comp] != 0:
                        d00_lam = -Vi[comp] / Wi[comp]
                        break
                lam_s = f'={d00_lam:.3f}' if d00_lam is not None else '→∞'
            seg_lines.append(f'  D00: vertex v{vi}, lambda{lam_s}')

    ax.text(0.01, 0.22, '\n'.join(seg_lines), fontsize=7.5,
            fontfamily='monospace', transform=ax.transAxes, va='top')

    poly_strs = [poly_to_latex(Q, 'Q')]
    for k in range(3):
        poly_strs.append(poly_to_latex(P[k], f'P_{k}'))

    y = 0.95
    for j, ps in enumerate(poly_strs):
        ax.text(0.50, y - j * 0.19, ps, fontsize=8,
                transform=ax.transAxes, va='top')


# ─── Figure assembly ─────────────────────────────────────────────────────────

def visualize_case(case_data, output_path=None):
    """Generate three-panel figure for a single triangle case."""
    ensure_float_fields(case_data)

    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 2, height_ratios=[3, 1.2], figure=fig,
                  hspace=0.25, wspace=0.3)

    cat = case_data['category']
    cat_display = cat.replace('_', ' ')
    seed = case_data['seed']
    fig.suptitle(f'{cat_display}  (seed={seed})', fontsize=12,
                 fontweight='bold', y=0.98, fontfamily='monospace')

    segments = collect_segments(case_data)

    # Panel (a): 2D triangle
    ax2d = fig.add_subplot(gs[0, 0])
    draw_tri_wireframe(ax2d)

    # D11: highlight degenerate edge
    if 'D11' in cat:
        d11_info = case_data.get('D11')
        if d11_info is not None and isinstance(d11_info, int):
            ev = EDGE_VERTS[d11_info]
            p1, p2 = TRI_VERTS[ev[0]], TRI_VERTS[ev[1]]
            ax2d.plot([p1[0], p2[0]], [p1[1], p2[1]],
                      color='#cc6600', linewidth=4.0, alpha=0.9, zorder=10)

    draw_vector_arrows(ax2d, case_data)
    draw_pv_curves(ax2d, segments)

    # Bubble: closed PV curve inside triangle (0 punctures)
    if case_data.get('B') and len(case_data.get('punctures', [])) == 0:
        Q = case_data['Q_coeffs']
        P = case_data['P_coeffs']
        t = np.linspace(-0.499 * np.pi, 0.499 * np.pi, 400)
        lam_vals = np.tan(t)
        pts = []
        for lam in lam_vals:
            mu = lambda_to_bary_tri(lam, Q, P)
            if mu is not None and all(m >= -0.01 for m in mu):
                mu_clip = np.clip(mu, 0, 1)
                s = mu_clip.sum()
                if s > 1e-10:
                    mu_clip /= s
                pts.append(bary_to_2d(mu_clip))
        if len(pts) > 2:
            pts.append(pts[0])
            pts = np.array(pts)
            ax2d.plot(pts[:, 0], pts[:, 1],
                      color=SEGMENT_COLORS[0], linewidth=2.5, zorder=5)
            segments = [{'color': SEGMENT_COLORS[0], 'pts_list': [pts],
                         'pi_entry': -1, 'pi_exit': -1,
                         'lam_entry': None, 'lam_exit': None,
                         'infinity_spanning': True}]

    draw_puncture_markers(ax2d, case_data, segments)
    draw_special_points(ax2d, case_data, segments)

    n_punc = len(case_data.get('punctures', []))
    n_seg = len(segments)
    ax2d.set_title(f'{n_punc} puncture{"s" if n_punc != 1 else ""}, '
                   f'{n_seg} segment{"s" if n_seg != 1 else ""}',
                   fontsize=10, pad=5)
    ax2d.set_xlim(-0.15, 1.15)
    ax2d.set_ylim(-0.15, 1.05)
    ax2d.set_aspect('equal')
    ax2d.axis('off')

    # Panel (b): Lambda ring
    ax_ring = fig.add_subplot(gs[0, 1])
    draw_lambda_ring(ax_ring, case_data, segments)

    degQ = case_data.get('degQ', 2)
    n_qr = len(case_data.get('Q_roots', []))
    if degQ < 2:
        disc_str = f'Q degree {degQ} ({n_qr} root{"s" if n_qr != 1 else ""})'
    else:
        Q_coeffs = case_data['Q_coeffs']
        disc = Q_coeffs[1]**2 - 4 * Q_coeffs[2] * Q_coeffs[0]
        if disc > 0:
            disc_str = r'$\Delta_Q > 0$ (2 roots)'
        elif disc < 0:
            disc_str = r'$\Delta_Q < 0$ (0 roots)'
        else:
            disc_str = r'$\Delta_Q = 0$'
    ax_ring.set_title(f'$\\lambda$-ring: {disc_str}', fontsize=10)

    # Panel (c): Info
    ax_info = fig.add_subplot(gs[1, :])
    draw_info_panel(ax_info, case_data, segments)

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        print(f"  Saved: {output_path}", file=sys.stderr)
    else:
        plt.show()

    plt.close(fig)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize 2D PV triangle cases')
    parser.add_argument('input', help='JSONL file from pv_tri_case_finder_2d')
    parser.add_argument('--output-dir', '-o', default='figures_2d',
                        help='Output directory (default: figures_2d/)')
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

    print(f"Loaded {len(cases)} cases", file=sys.stderr)

    cat_counts = defaultdict(int)
    for c in cases:
        cat_counts[c['category']] += 1
    print("\nCategory distribution:", file=sys.stderr)
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:40s} {cat_counts[cat]}", file=sys.stderr)

    if args.categories:
        cases = [c for c in cases if c['category'] in args.categories]
        print(f"\nFiltered to {len(cases)} cases", file=sys.stderr)

    if args.seeds:
        seed_set = set(args.seeds)
        cases = [c for c in cases if c['seed'] in seed_set]
        print(f"\nFiltered to {len(cases)} cases by seed", file=sys.stderr)

    if args.first_per_category:
        seen = set()
        filtered = []
        for c in cases:
            if c['category'] not in seen:
                seen.add(c['category'])
                filtered.append(c)
        cases = filtered
        print(f"\nFirst per category: {len(cases)} cases", file=sys.stderr)

    if len(cases) > args.max_cases:
        cases = cases[:args.max_cases]
        print(f"Truncated to {args.max_cases} cases", file=sys.stderr)

    print(f"\nGenerating {len(cases)} figures...", file=sys.stderr)
    for i, case_data in enumerate(cases):
        cat = case_data['category'].replace('/', '_')
        seed = case_data['seed']
        filename = f"pvtri2d_{cat}_seed{seed}.{args.format}"
        output_path = os.path.join(args.output_dir, filename)
        print(f"[{i+1}/{len(cases)}] {case_data['category']} (seed={seed})",
              file=sys.stderr)
        try:
            visualize_case(case_data, output_path)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"\nDone. {len(cases)} figures saved to {args.output_dir}/",
          file=sys.stderr)


if __name__ == '__main__':
    main()
