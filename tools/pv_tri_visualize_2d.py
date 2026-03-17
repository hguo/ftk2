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


# ─── Integer polynomial helpers (threshold-free) ─────────────────────────

def _ipoly_deg(p):
    d = len(p) - 1
    while d > 0 and p[d] == 0:
        d -= 1
    return d


def _isign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _sign_A_plus_Bs_sqrtD(A, Bs, D):
    """Sign of A + Bs*sqrt(D), all integers, D >= 0. Pure integer."""
    if Bs == 0:
        return _isign(A)
    if Bs > 0:
        if A >= 0:
            return 1
        return _isign(Bs * Bs * D - A * A)
    else:  # Bs < 0
        if A <= 0:
            return -1
        return _isign(A * A - Bs * Bs * D)


def _eval_sign_at_root_int(pj, pk, root_idx):
    """Sign of polynomial pj evaluated at root root_idx of polynomial pk.
    Both are integer-coefficient lists [c0, c1, ...]. Pure integer arithmetic.
    pk must have degree 1 or 2 with real roots. Returns +1, -1, or 0."""
    dk = _ipoly_deg(pk)
    dj = _ipoly_deg(pj)
    if dj == 0:
        return _isign(pj[0])

    if dk == 1:
        # root = -pk[0]/pk[1]. Evaluate pj(-pk[0]/pk[1]) * pk[1]^dj
        a, b = pk[0], pk[1]
        # val = sum(pj[i] * (-a)^i * b^(dj-i))
        val = 0
        neg_a_pow = 1  # (-a)^i
        b_pow = b ** dj  # b^(dj-i) starts at b^dj
        for i in range(dj + 1):
            val += pj[i] * neg_a_pow * b_pow
            neg_a_pow *= -a
            if i < dj:
                b_pow //= b
        # sign(pj(root)) = sign(val) * sign(b^dj)
        # b^dj sign: if dj even, always +1. If dj odd, sign(b).
        b_dj_sign = 1 if (dj % 2 == 0) else _isign(b)
        return _isign(val) * b_dj_sign

    if dk == 2:
        a, b, c = pk[0], pk[1], pk[2]
        D = b * b - 4 * a * c  # discriminant, >= 0
        s = -1 if root_idx == 0 else 1  # smaller/larger root

        if dj == 1:
            d, e = pj[0], pj[1]
            # pj(root) * 2c = (2cd - eb) + e*s*sqrt(D)
            A = 2 * c * d - e * b
            Bs = e * s
            return _sign_A_plus_Bs_sqrtD(A, Bs, D) * _isign(c)

        if dj == 2:
            d, e, f = pj[0], pj[1], pj[2]
            # pj(root) * 4c² = A + B*s*sqrt(D)
            A = 4*c*c*d - 2*b*c*e + 2*f*b*b - 4*a*c*f
            B = 2 * (c * e - b * f)
            Bs = B * s
            return _sign_A_plus_Bs_sqrtD(A, Bs, D)

    return 0  # fallback


def _compare_roots_int(P_red_int, fA, rA, fB, rB):
    """Compare λ_A (root rA of P_red[fA]) with λ_B (root rB of P_red[fB]).
    Returns +1 if λ_A > λ_B, -1 if λ_A < λ_B, 0 if equal. Pure integer."""
    if fA == fB:
        if rA == rB:
            return 0
        return -1 if rA < rB else 1
    pkB = P_red_int[fB]
    dB = _ipoly_deg(pkB)
    sign_B = _eval_sign_at_root_int(pkB, P_red_int[fA], rA)
    if sign_B == 0:
        return 0
    if dB == 1:
        # Linear: monotonic. sign_B * sign(leading) gives comparison.
        return sign_B * _isign(pkB[1])
    if dB == 2:
        c = pkB[2]
        # Between roots: sign_B opposite to leading coeff c
        if sign_B * c < 0:
            # λ_A is between root_0 and root_1 of P_B
            return 1 if rB == 0 else -1
        else:
            # λ_A is outside both roots. Use derivative to determine side.
            # P_B'(x) = pkB[1] + 2*pkB[2]*x (linear)
            deriv = [pkB[1], 2 * pkB[2]]
            sign_d = _eval_sign_at_root_int(deriv, P_red_int[fA], rA)
            if sign_d * c > 0:
                # Right of root_1 → λ_A > root_1 ≥ root_0
                return 1
            else:
                # Left of root_0 → λ_A < root_0 ≤ root_1
                return -1
    return 0

def _ipoly_content(p, d):
    from math import gcd as igcd
    g = 0
    for i in range(d + 1):
        g = igcd(g, abs(p[i]))
    return max(g, 1)

def _ipoly_prem(A, dA, B, dB):
    """Pseudo-remainder of A by B, integer arithmetic."""
    from math import gcd as igcd
    R = list(A[:dA + 1])
    lc = B[dB]
    d = dA
    while d >= dB:
        coeff = R[d]
        if coeff == 0:
            d -= 1
            continue
        R_new = [r * lc for r in R]
        for j in range(dB + 1):
            R_new[d - dB + j] -= coeff * B[j]
        R = R_new
        d -= 1
    dR = len(R) - 1
    while dR > 0 and R[dR] == 0:
        dR -= 1
    c = _ipoly_content(R, dR)
    return [R[i] // c for i in range(dR + 1)], dR

def _ipoly_gcd(A, dA, B, dB):
    """GCD of two integer polynomials. Content-reduced, positive leading coeff."""
    if dA == 0 and A[0] == 0:
        if dB == 0 and B[0] == 0:
            return [0], 0
        c = _ipoly_content(B, dB)
        h = [B[i] // c for i in range(dB + 1)]
        if h[dB] < 0: h = [-x for x in h]
        return h, dB
    if dB == 0 and B[0] == 0:
        c = _ipoly_content(A, dA)
        h = [A[i] // c for i in range(dA + 1)]
        if h[dA] < 0: h = [-x for x in h]
        return h, dA
    if dA < dB:
        A, dA, B, dB = list(B[:dB + 1]), dB, list(A[:dA + 1]), dA
    else:
        A, B = list(A[:dA + 1]), list(B[:dB + 1])
    while dB >= 1:
        R, dR = _ipoly_prem(A, dA, B, dB)
        if dR == 0 and R[0] == 0:
            break
        A, dA = B, dB
        B, dB = R, dR
    if dB < 1:
        return [1], 0
    c = _ipoly_content(B, dB)
    h = [B[i] // c for i in range(dB + 1)]
    if h[dB] < 0:
        h = [-x for x in h]
    return h, dB

def _ipoly_exact_div(A, dA, B, dB):
    """Exact polynomial division A/B. B must divide A."""
    if dA < dB:
        return [0]
    if dB == 0:
        b0 = B[0]
        return [A[i] // b0 for i in range(dA + 1)]
    A = list(A[:dA + 1])
    dQ = dA - dB
    Q = [0] * (dQ + 1)
    for i in range(dQ, -1, -1):
        Q[i] = A[i + dB] // B[dB]
        for j in range(dB + 1):
            A[i + j] -= Q[i] * B[j]
    return Q


# ─── Integer GCD reduction ────────────────────────────────────────────────

def _poly_gcd_reduce_int(P_strs, Q_strs):
    """Compute h = gcd(P[0], P[1], P[2]) and divide it out, pure integer.

    The C++ solver factors out h from P and Q before finding face roots.
    root_idx in the output refers to P_red = P/h roots, not original P roots.
    This function replicates that reduction so the visualizer uses the
    same reduced polynomials.

    Returns (P_red_int, Q_red_int, h_int, h_deg).
    """
    P = [[int(x) for x in pk] for pk in P_strs]
    Q = [int(x) for x in Q_strs]
    for pk in P:
        while len(pk) < 3:
            pk.append(0)
    while len(Q) < 3:
        Q.append(0)

    degP = [_ipoly_deg(pk) for pk in P]
    degQ = _ipoly_deg(Q)

    # h = gcd(P[0], P[1], P[2])
    h, dh = _ipoly_gcd(P[0], degP[0], P[1], degP[1])
    h, dh = _ipoly_gcd(h, dh, P[2], degP[2])

    if dh == 0:
        return P, Q, [1], 0

    # Reduce
    P_red = []
    for k in range(3):
        q = _ipoly_exact_div(P[k], degP[k], h, dh)
        while len(q) < 3:
            q.append(0)
        P_red.append(q)

    Q_red = _ipoly_exact_div(Q, degQ, h, dh)
    while len(Q_red) < 3:
        Q_red.append(0)

    return P_red, Q_red, h, dh


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

    # ── 1. Original P/Q from i128 strings ──
    Q_coeffs = [float(int(c)) for c in case_data['Q']]
    P_coeffs = [[float(int(c)) for c in row] for row in case_data['P']]
    case_data['Q_coeffs'] = Q_coeffs
    case_data['P_coeffs'] = P_coeffs

    # ── 2. GCD reduction (pure integer arithmetic) ──
    # The C++ solver divides out h = gcd(P[0], P[1], P[2]) and uses
    # P_red = P/h for face roots and Q_red = Q/h for interval boundaries.
    # root_idx in JSON refers to P_red roots, NOT original P roots.
    P_red_int, Q_red_int, h_int, h_deg = _poly_gcd_reduce_int(
        case_data['P'], case_data['Q'])
    P_red_f = [[float(c) for c in pk] for pk in P_red_int]
    Q_red_f = [float(c) for c in Q_red_int]
    case_data['P_red_coeffs'] = P_red_f
    case_data['Q_red_coeffs'] = Q_red_f
    case_data['P_red_int'] = P_red_int  # pure integer, no float
    case_data['h_deg'] = h_deg

    # SR info: per-face integer GCD data for shared-root visualization.
    # Stores integer gcd coefficients directly — no float matching needed.
    Q_int = [int(c) for c in case_data['Q']]
    P_int = [[int(c) for c in pk] for pk in case_data['P']]
    degQ_orig = _ipoly_deg(Q_int)
    sr_info = []
    for k in range(3):
        degPk = _ipoly_deg(P_int[k])
        g, dg = _ipoly_gcd(Q_int, degQ_orig, P_int[k], degPk)
        if dg == 1:
            # Linear gcd: one root at -g[0]/g[1]
            sr_info.append({'g': list(g), 'dg': 1,
                            'lam': -g[0] / g[1]})
        elif dg == 2:
            disc_g = g[1] * g[1] - 4 * g[2] * g[0]
            if disc_g > 0:
                # Two distinct roots
                g_f = [float(c) for c in g]
                roots = sorted(np.roots([g_f[2], g_f[1], g_f[0]]).real.tolist())
                for r in roots:
                    sr_info.append({'g': list(g), 'dg': 2, 'lam': r})
            elif disc_g == 0:
                # Double root at -g[1]/(2*g[2])
                sr_info.append({'g': list(g), 'dg': 2,
                                'lam': -g[1] / (2.0 * g[2])})
    # Deduplicate sr_info entries at the same lambda value
    seen_lams = set()
    deduped_sr = []
    for si in sr_info:
        lam_key = si['lam']
        if lam_key not in seen_lams:
            seen_lams.add(lam_key)
            deduped_sr.append(si)
    case_data['sr_info'] = deduped_sr

    # ── 3. Q_red degree and roots (interval boundaries) ──
    degQ_red = 2
    while degQ_red > 0 and abs(Q_red_f[degQ_red]) < 0.5:
        degQ_red -= 1
    case_data['degQ'] = degQ_red

    # Q_red roots define the interval boundaries
    if degQ_red == 2:
        disc_qr = Q_red_f[1]**2 - 4 * Q_red_f[2] * Q_red_f[0]
        n_qr = 2 if disc_qr > 0 else (1 if disc_qr == 0 else 0)
    elif degQ_red == 1:
        n_qr = 1
    else:
        n_qr = 0
    Q_roots = _solve_poly_roots(Q_red_f, degQ_red, n_qr)
    case_data['Q_roots'] = Q_roots
    # Override n_Q_roots with the Q_red root count (the solver uses Q_red)
    case_data['n_Q_roots'] = n_qr
    # Q2o: double root is tangent point, not interval boundary
    Q_int_raw = [int(c) for c in case_data['Q']]
    disc_Q_int = Q_int_raw[1]*Q_int_raw[1] - 4*Q_int_raw[0]*Q_int_raw[2]
    case_data['Q_disc_zero'] = (disc_Q_int == 0 and n_qr == 1)

    # ── 4. P_red face roots (match C++ root_idx) ──
    face_roots = {}
    for k in range(3):
        pk = P_red_f[k]
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

    # ── 5. Puncture lambda and bary ──
    for pi in case_data.get('punctures', []):
        f = pi['face']
        ri = pi['root_idx']

        if ri < 0:
            # Infinity puncture: bary from leading coefficients
            pi['lambda'] = None
            dQ = degQ_red
            if dQ >= 1 and abs(Q_red_f[dQ]) > 0.5:
                mu = [poly_eval(P_red_f[k2], 0) for k2 in range(3)]
                # Use leading-coeff ratio: P_red[k][dQ] / Q_red[dQ]
                mu = [P_red_f[k2][dQ] / Q_red_f[dQ] for k2 in range(3)]
                pi['bary'] = mu
            else:
                pi['bary'] = [0.33, 0.33, 0.34]
        elif ri < len(face_roots.get(f, [])):
            lam = face_roots[f][ri]
            pi['lambda'] = lam
            # Use Q_red/P_red for bary — no 0/0 at non-SR lambdas
            Q_val = poly_eval(Q_red_f, lam)
            if abs(Q_val) > 1e-30:
                mu = [poly_eval(P_red_f[k2], lam) / Q_val for k2 in range(3)]
                pi['bary'] = mu
            else:
                # Fallback: L'Hôpital on original P/Q
                Q_prime = [Q_coeffs[j] * j for j in range(1, len(Q_coeffs))]
                Qp_val = poly_eval(Q_prime, lam)
                if abs(Qp_val) > 1e-30:
                    mu = [poly_eval([P_coeffs[k2][j] * j
                                     for j in range(1, len(P_coeffs[k2]))],
                                    lam) / Qp_val
                          for k2 in range(3)]
                    pi['bary'] = mu
                else:
                    pi['bary'] = [0.33, 0.33, 0.34]
        else:
            pi['lambda'] = 0.0
            pi['bary'] = [0.33, 0.33, 0.34]

    # ── 6. Build intervals from Q_red roots ──
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
        # Prefer pairs where the sub-segment's lambdas are BETWEEN the pair endpoints
        d = min(_lam_dist(lam_e, l1), _lam_dist(lam_e, l2),
                _lam_dist(lam_x, l1), _lam_dist(lam_x, l2))
        # Bonus: if sub-segment midpoint is between pair endpoints, reduce distance
        if l1 is not None and l2 is not None and lam_e is not None and lam_x is not None:
            lo, hi = min(l1, l2), max(l1, l2)
            mid = (lam_e + lam_x) / 2
            if lo <= mid <= hi:
                d *= 0.01  # strongly prefer this pair
        if d < best_dist:
            best_dist = d
            best_pair = idx
    return best_pair


def collect_segments(case_data):
    """Build segments from pre-computed puncture pairing."""
    # Use reduced P_red/Q_red for curve sampling (avoids 0/0 near SR roots)
    Q = case_data.get('Q_red_coeffs', case_data['Q_coeffs'])
    P = case_data.get('P_red_coeffs', case_data['P_coeffs'])
    intervals = case_data['intervals']
    punctures = case_data['punctures']
    puncture_lambdas = [p.get('lambda') for p in punctures]

    # Read pairs from C++ output: [{a, b, inf, cw}, ...]
    raw_pairs = case_data.get('pairs', [])
    pairs = []
    pair_inf_span = []
    pair_cw = []  # band direction: True=clockwise to ∞
    for p in raw_pairs:
        if isinstance(p, dict):
            pairs.append([p['a'], p['b']])
            pair_inf_span.append(p.get('inf', False))
            pair_cw.append(p.get('cw', False))
        else:
            pairs.append(list(p))
            pair_inf_span.append(False)
            pair_cw.append(False)

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

    # ── inf_span directly from C++ output ──
    segments = []
    for idx, (pi1, pi2) in enumerate(pairs):
        color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
        l1 = punctures[pi1].get('lambda')
        l2 = punctures[pi2].get('lambda')
        inf_span = pair_inf_span[idx] if idx < len(pair_inf_span) else False
        cw = pair_cw[idx] if idx < len(pair_cw) else False

        segments.append({
            'pts_list': pair_subsegs.get(idx, []),
            'color': color,
            'cw': cw,
            'pi_entry': pi1,
            'pi_exit': pi2,
            'lam_entry': l1,
            'lam_exit': l2,
            'infinity_spanning': inf_span,
        })

    # Connect infinity-spanning sub-segments through the λ→∞ point.
    # When a curve wraps through ∞, sample_pv_curve produces two disjoint
    # sub-segments (one going to +∞, one coming from -∞).  Stitch them
    # together through the ∞ position so the curve is visually continuous.
    Q_r = case_data.get('Q_red_coeffs', case_data.get('Q_coeffs'))
    P_r = case_data.get('P_red_coeffs', case_data.get('P_coeffs'))
    dQ = case_data.get('degQ', 2)
    for seg in segments:
        pts_list = seg.get('pts_list', [])
        if not seg.get('infinity_spanning') or len(pts_list) < 2:
            continue
        if dQ < 1 or abs(Q_r[dQ]) < 0.5:
            continue
        mu_inf = np.array([P_r[k][dQ] / Q_r[dQ] for k in range(3)])
        if np.any(mu_inf < -0.01):
            continue
        pos_inf = bary_to_2d(np.clip(mu_inf, 0, 1))
        # Find sub-seg whose LAST point is closest to pos_inf (+∞ arm)
        best_end = min(range(len(pts_list)),
                       key=lambda i: np.linalg.norm(pts_list[i][-1] - pos_inf))
        # Find sub-seg whose FIRST point is closest to pos_inf (-∞ arm)
        best_start = min(range(len(pts_list)),
                         key=lambda i: np.linalg.norm(pts_list[i][0] - pos_inf))
        if best_end != best_start:
            merged = np.vstack([pts_list[best_end], [pos_inf],
                                pts_list[best_start]])
            new_list = [merged]
            for i in range(len(pts_list)):
                if i != best_end and i != best_start:
                    new_list.append(pts_list[i])
            seg['pts_list'] = new_list

    # Unpaired sub-segments: only create catch-all for genuine bubble (B tag).
    if not pairs and all_subsegs:
        n_punc = len(punctures)
        cat = case_data.get('category', '')
        if n_punc == 0 and '_B' in cat:
            segments = [{
                'pts_list': [pts for pts, _, _ in all_subsegs],
                'color': SEGMENT_COLORS[0],
                'pi_entry': -1,
                'pi_exit': -1,
                'lam_entry': None,
                'lam_exit': None,
                'infinity_spanning': True,
            }]

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
    """Find triangle vertices where det(V, W) = 0 (D00 degeneracy).
    Pure integer arithmetic — no float thresholds."""
    V = [[int(v) for v in row] for row in case_data['V']]
    W = [[int(w) for w in row] for row in case_data['W']]
    d00 = []
    for i in range(3):
        det = V[i][0] * W[i][1] - V[i][1] * W[i][0]
        if det == 0:
            d00.append(i)
    return d00


def find_d11_edges(case_data):
    """Find triangle edges where both endpoints have det(V,W)=0 at the same lambda.
    Pure integer arithmetic — no float thresholds."""
    V = [[int(v) for v in row] for row in case_data['V']]
    W = [[int(w) for w in row] for row in case_data['W']]

    pv_info = {}
    for i in range(3):
        det = V[i][0] * W[i][1] - V[i][1] * W[i][0]
        if det != 0:
            continue
        v_zero = all(v == 0 for v in V[i])
        w_zero = all(w == 0 for w in W[i])
        if v_zero and w_zero:
            pv_info[i] = (0, 0, True)
        elif v_zero:
            pv_info[i] = (0, 1, False)
        elif w_zero:
            pv_info[i] = (1, 0, False)
        else:
            for k in range(2):
                if W[i][k] != 0:
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
        elif d1 != 0 and d2 != 0 and n1 * d2 == n2 * d1:
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
    Pure integer arithmetic for all sign/zero checks.
    Float only for final 2D position (display).
    Returns 2D position or None."""
    F_int = [[int(f) for f in row] for row in F]

    # Check vertex zeros first (exact integer)
    for i in range(3):
        if all(f == 0 for f in F_int[i]):
            return bary_to_2d(np.eye(3)[i])

    # Cramer's rule: mu_0(F_0-F_2) + mu_1(F_1-F_2) = -F_2
    a00 = F_int[0][0] - F_int[2][0]
    a01 = F_int[1][0] - F_int[2][0]
    a10 = F_int[0][1] - F_int[2][1]
    a11 = F_int[1][1] - F_int[2][1]
    det = a00 * a11 - a01 * a10

    if det != 0:
        rhs_x = -F_int[2][0]
        rhs_y = -F_int[2][1]
        mu0_num = rhs_x * a11 - rhs_y * a01
        mu1_num = a00 * rhs_y - a10 * rhs_x
        mu2_num = det - mu0_num - mu1_num
        # Inside triangle: all mu_i/det >= 0 (same sign as det)
        if det > 0:
            inside = mu0_num >= 0 and mu1_num >= 0 and mu2_num >= 0
        else:
            inside = mu0_num <= 0 and mu1_num <= 0 and mu2_num <= 0
        if inside:
            mu = np.array([mu0_num / det, mu1_num / det, mu2_num / det])
            return bary_to_2d(np.clip(mu, 0, 1))
        return None

    # Degenerate: all F vectors are parallel (det=0).
    # Project onto any nonzero F vector using integer dot products.
    d_idx = None
    for i in range(3):
        if any(f != 0 for f in F_int[i]):
            d_idx = i
            break
    if d_idx is None:
        # All F vectors are zero
        return bary_to_2d(np.array([1/3, 1/3, 1/3]))

    d = F_int[d_idx]
    projs = [F_int[i][0] * d[0] + F_int[i][1] * d[1] for i in range(3)]
    # Check each edge for sign change (integer sign check)
    for edge_k, (vi, vj) in enumerate(EDGE_VERTS):
        if projs[vi] * projs[vj] < 0:
            # Zero crossing on this edge (float for position only)
            t = float(projs[vi]) / float(projs[vi] - projs[vj])
            mu = np.zeros(3)
            mu[vi] = 1 - t
            mu[vj] = t
            return bary_to_2d(mu)
    return None


def compute_cv_position(case_data):
    """Compute Cv position on the PV curve at λ=0: bary = P[k](0)/Q(0).
    When Q(0)=0 (SR at λ=0), L'Hôpital: bary = P[k][1]/Q[1].
    When Q≡0 (Qz), fall back to geometric V=0 check.
    Returns None if bary is outside triangle (integer sign check)."""
    Q_int = [int(c) for c in case_data['Q']]
    P_int = [[int(c) for c in pk] for pk in case_data['P']]
    denom = Q_int[0]
    if denom == 0:
        denom = Q_int[1] if len(Q_int) > 1 else 0
        mu_num = [P_int[k][1] if len(P_int[k]) > 1 else 0 for k in range(3)]
    else:
        mu_num = [P_int[k][0] for k in range(3)]
    if denom == 0:
        return _find_field_zero_2d(case_data['V'])
    # Integer inside check: all mu_num same sign as denom
    for m in mu_num:
        if m * denom < 0:
            return None
    mu = np.array([float(m) / float(denom) for m in mu_num])
    return bary_to_2d(mu)


def compute_cw_position(case_data):
    """Compute Cw position: PV curve limit at λ→∞ → bary = P[k][d]/Q[d].
    Returns None if bary is outside triangle (integer sign check)."""
    Q_int = [int(c) for c in case_data['Q']]
    P_int = [[int(c) for c in pk] for pk in case_data['P']]
    d = _ipoly_deg(Q_int)
    if d == 0:
        return None
    for k in range(3):
        if _ipoly_deg(P_int[k]) > d:
            return None
    denom = Q_int[d]
    if denom == 0:
        return None
    mu_num = [P_int[k][d] if len(P_int[k]) > d else 0 for k in range(3)]
    # Integer inside check
    for m in mu_num:
        if m * denom < 0:
            return None
    mu = np.array([float(m) / float(denom) for m in mu_num])
    return bary_to_2d(mu)


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
    # Position ON THE CURVE via L'Hôpital at D00 lambda.
    cv_type = case_data.get('Cv', 0)
    cw_type = case_data.get('Cw', 0)
    if 'D00' in category:
        Q_int_d = [int(c) for c in case_data['Q']]
        P_int_d = [[int(c) for c in pk] for pk in case_data['P']]
        for vi in find_d00_vertices(case_data):
            Vi = [int(v) for v in case_data['V'][vi]]
            Wi = [int(w) for w in case_data['W'][vi]]
            v_zero = all(v == 0 for v in Vi)
            w_zero = all(w == 0 for w in Wi)
            # Skip if Cv0/Cw0 will draw its own marker at same position
            if v_zero and cv_type == 3:
                continue
            if w_zero and cw_type == 3:
                continue

            # Compute D00 lambda (integer: -V[k]/W[k])
            d00_lam_num, d00_lam_den = 0, 1
            if w_zero:
                lam_str = r'$\lambda\!\to\!\infty$'
                # Position: Cw leading-coeff limit
                p = compute_cw_position(case_data)
                if p is None:
                    p = TRI_VERTS[vi]
            elif v_zero:
                lam_str = r'$\lambda=0$'
                # Position: Cv on-curve limit
                p = compute_cv_position(case_data)
                if p is None:
                    p = TRI_VERTS[vi]
            else:
                # D00 at finite nonzero λ = -V[k]/W[k]
                for comp in range(2):
                    if Wi[comp] != 0:
                        d00_lam_num = -Vi[comp]
                        d00_lam_den = Wi[comp]
                        break
                d00_lam = float(d00_lam_num) / float(d00_lam_den)
                lam_str = f'$\\lambda={d00_lam:.2f}$'
                # Position on curve: P(λ)/Q(λ), L'Hôpital if Q(λ)=0
                # Evaluate Q at λ = num/den in integer: Q(n/d)*d² = Q[0]*d² + Q[1]*n*d + Q[2]*n²
                n, d = d00_lam_num, d00_lam_den
                q_nd2 = Q_int_d[0]*d*d + Q_int_d[1]*n*d + Q_int_d[2]*n*n
                if q_nd2 != 0:
                    mu_num = [P_int_d[k][0]*d*d + P_int_d[k][1]*n*d + P_int_d[k][2]*n*n
                              for k in range(3)]
                    denom_d00 = q_nd2
                else:
                    # L'Hôpital: Q'(λ) = Q[1] + 2*Q[2]*λ → Q'(n/d)*d = Q[1]*d + 2*Q[2]*n
                    qp_d = Q_int_d[1]*d + 2*Q_int_d[2]*n
                    if qp_d != 0:
                        mu_num = [P_int_d[k][1]*d + 2*P_int_d[k][2]*n for k in range(3)]
                        denom_d00 = qp_d
                    else:
                        continue  # can't compute position
                # Inside check (integer): all mu same sign as denom
                if any(m * denom_d00 < 0 for m in mu_num):
                    # On-curve position is outside → fall back to vertex
                    p = TRI_VERTS[vi]
                else:
                    mu = [float(m) / float(denom_d00) for m in mu_num]
                    p = bary_to_2d(mu)

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

    # SR/ISR — shared root visualization (pure integer arithmetic).
    # Uses sr_info from ensure_float_fields: integer gcd coefficients stored
    # directly — no float matching or recomputation needed.
    if 'SR' in category:
        P_int = [[int(c) for c in pk] for pk in case_data['P']]
        Q_int = [int(c) for c in case_data['Q']]
        sr_label = 'ISR' if 'ISR' in category else 'SR'
        for si in case_data.get('sr_info', []):
            g = si['g']
            dg = si['dg']
            lam_float = si['lam']

            # SR at λ=0: use L'Hôpital on-curve position (colocated with Cv).
            # Detect λ=0 root from integer gcd: g[0]==0 means λ divides g(λ).
            if g[0] == 0:
                pos = compute_cv_position(case_data)
                if pos is not None:
                    annots.append(dict(pos=pos, tag=sr_label,
                                       lam_str=r'$\lambda=0$',
                                       color='#ff8800', bg='#fff8ee',
                                       marker='D', ms=120, mc='#ff8800'))
                continue

            if dg == 1:
                g0, g1 = g[0], g[1]
                # Integer L'Hôpital: denom = g1·Q'(λ) = g1·Q[1] - 2·Q[2]·g0
                denom = g1 * Q_int[1] - 2 * Q_int[2] * g0
                if denom == 0:
                    continue
                mu_num = [g1 * P_int[j][1] - 2 * P_int[j][2] * g0
                          for j in range(3)]
            elif dg == 2:
                ga, gb, g0 = g[2], g[1], g[0]
                if ga == 0:
                    continue
                # Quadratic gcd: evaluate L'Hôpital at specific root.
                # Q'(λ) = Q[1]+2Q[2]λ, P'(λ) = P[j][1]+2P[j][2]λ (both linear).
                # Evaluate these linear polys at root of g using integer
                # eval_sign for inside check.
                Qp_poly = [Q_int[1], 2*Q_int[2]]  # Q'(λ) as polynomial
                denom_sign = _eval_sign_at_root_int(Qp_poly, [g0,gb,ga],
                    0 if si == case_data.get('sr_info',[])[0] else 1)
                # Use sr_info index to determine root_idx
                sr_list = case_data.get('sr_info', [])
                ri_g = 0
                for si_idx, si_item in enumerate(sr_list):
                    if si_item is si:
                        ri_g = si_idx  # 0 for first root, 1 for second
                        break
                denom_sign = _eval_sign_at_root_int(Qp_poly, [g0,gb,ga], ri_g)
                if denom_sign == 0:
                    continue
                inside = True
                for j in range(3):
                    Pp_poly = [P_int[j][1], 2*P_int[j][2]]
                    mu_sign = _eval_sign_at_root_int(Pp_poly, [g0,gb,ga], ri_g)
                    if mu_sign * denom_sign < 0:
                        inside = False
                        break
                if not inside:
                    continue
                # Float position (display only)
                Qp_val = Q_int[1] + 2*Q_int[2]*lam_float
                mu_num = [P_int[j][1] + 2*P_int[j][2]*lam_float for j in range(3)]
                denom = Qp_val
            else:
                continue

            # Suppress if any mu_j opposite sign to denom (outside triangle)
            outside = any((m > 0) != (denom > 0) and m != 0 for m in mu_num)
            if outside:
                continue
            mu = [float(m) / float(denom) for m in mu_num]
            pos = bary_to_2d(mu)
            annots.append(dict(pos=pos, tag=sr_label,
                               lam_str=f'$\\lambda={lam_float:.2f}$',
                               color='#ff8800', bg='#fff8ee',
                               marker='D', ms=120, mc='#ff8800'))

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

    # TN: tangency — pure integer arithmetic.
    # TN at double root of P[k]: λ = -b/(2a) where a=P[k][2], b=P[k][1].
    # Show "TN" from C++ category tag — NO reclassification as ITN.
    # Float only for final bary → 2D position.
    if 'TN' in category:
        P_int = [[int(c) for c in pk] for pk in case_data['P']]
        Q_int = [int(c) for c in case_data['Q']]
        for k in range(3):
            a = P_int[k][2]
            b = P_int[k][1]
            c0 = P_int[k][0]
            if a == 0:
                continue
            disc = b * b - 4 * a * c0
            if disc != 0:
                continue  # Not a double root (exact integer check)

            # TN at λ = -b/(2a).  Float only for label/position.
            lam_float = -b / (2.0 * a)

            # Evaluate 4a²·Q(λ) in integer arithmetic (no division):
            q4a2 = 4*a*a*Q_int[0] - 2*a*b*Q_int[1] + b*b*Q_int[2]

            if q4a2 != 0:
                mu_num = []
                for j in range(3):
                    pj4a2 = (4*a*a*P_int[j][0]
                             - 2*a*b*P_int[j][1]
                             + b*b*P_int[j][2])
                    mu_num.append(pj4a2)
                mu = [float(m) / float(q4a2) for m in mu_num]
            else:
                # Q(λ)=0 (SR+TN): L'Hôpital — μ_j = P[j]'(λ)/Q'(λ)
                aqp = a * Q_int[1] - Q_int[2] * b
                if aqp == 0:
                    continue  # Higher-order degeneracy
                mu_num = []
                for j in range(3):
                    apjp = a * P_int[j][1] - P_int[j][2] * b
                    mu_num.append(apjp)
                mu = [float(m) / float(aqp) for m in mu_num]

            # Suppress if point is at a vertex (2 zero bary coords = D00)
            n_zero_mu = sum(1 for m in mu_num if m == 0)
            if n_zero_mu >= 2:
                continue

            tn_color = '#9933cc'
            pos = bary_to_2d(mu)
            seg_color = _find_segment_for_lambda(lam_float, segments)
            mc = seg_color if seg_color != '#333333' else tn_color
            annots.append(dict(pos=pos, tag='TN',
                               lam_str=f'$\\lambda={lam_float:.2f}$',
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

    r_band_inner = 0.85
    r_band_outer = 0.98
    n_segs = max(len(segments), 1)

    def _draw_band(a_start, a_end, color, ri, ro):
        arc_th = np.linspace(a_start, a_end, 80)
        inner_x = ri * np.cos(arc_th)
        inner_y = ri * np.sin(arc_th)
        outer_x = ro * np.cos(arc_th)
        outer_y = ro * np.sin(arc_th)
        verts_x = np.concatenate([outer_x, inner_x[::-1]])
        verts_y = np.concatenate([outer_y, inner_y[::-1]])
        verts = np.column_stack([verts_x, verts_y])
        ax.add_patch(Polygon(verts, closed=True,
                             facecolor=color, alpha=0.45,
                             edgecolor=color, linewidth=0.5, zorder=1))

    for si, seg in enumerate(segments):
        lam1 = seg['lam_entry']
        lam2 = seg['lam_exit']
        a1 = lambda_to_angle(lam1, scale) if lam1 is not None else np.pi / 2
        a2 = lambda_to_angle(lam2, scale) if lam2 is not None else np.pi / 2

        # Stack bands concentrically so they never overlap
        band_w = (r_band_outer - r_band_inner) / n_segs
        ri = r_band_inner + si * band_w
        ro = r_band_inner + (si + 1) * band_w

        if seg.get('infinity_spanning', False):
            if (lam1 is None) != (lam2 is None):
                # One endpoint at ∞: direction from C++ cw flag
                finite_lam = lam2 if lam1 is None else lam1
                a_finite = lambda_to_angle(finite_lam, scale)
                if seg.get('cw', False):
                    _draw_band(a_finite, -3 * np.pi / 2, seg['color'], ri, ro)
                else:
                    _draw_band(a_finite, np.pi / 2, seg['color'], ri, ro)
            else:
                # Both finite: complement arc through ∞
                a_hi = max(a1, a2)
                a_lo = min(a1, a2)
                _draw_band(a_hi, a_lo + 2 * np.pi, seg['color'], ri, ro)
        else:
            if a1 > a2:
                a1, a2 = a2, a1
            _draw_band(a1, a2, seg['color'], ri, ro)

    # Q roots
    q_label_r = 0.60
    q_label_radii = [q_label_r] * len(Q_roots)
    q_angles = [lambda_to_angle(r, scale) for r in Q_roots]
    q_sorted = sorted(range(len(Q_roots)), key=lambda i: q_angles[i])
    for j in range(1, len(q_sorted)):
        ip, ic = q_sorted[j - 1], q_sorted[j]
        if abs(q_angles[ic] - q_angles[ip]) < 0.30:
            q_label_radii[ic] = 0.42 if q_label_radii[ip] >= 0.55 else 0.60

    # Skip Q-root circles for Q2o (double root = tangent, not boundary)
    q_disc_zero = case_data.get('Q_disc_zero', False)
    for i, r in enumerate(Q_roots):
        if q_disc_zero:
            continue  # Q2o: tangent point, not interval boundary
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

    # SR/ISR on ring — uses sr_info directly (pure integer, no thresholds)
    if 'SR' in category:
        P_int = [[int(c) for c in pk] for pk in case_data['P']]
        Q_int = [int(c) for c in case_data['Q']]
        sr_ring_label = 'ISR' if 'ISR' in category else 'SR'
        for si in case_data.get('sr_info', []):
            g = si['g']
            dg = si['dg']
            lam_float = si['lam']

            # SR at λ=0 (g[0]==0): show only if on-curve position is inside
            if g[0] == 0:
                if compute_cv_position(case_data) is not None:
                    ring_annots.append(dict(angle=lambda_to_angle(0.0, scale),
                        tag=sr_ring_label, lam_str=r'$\lambda$=0',
                        color='#ff8800', bg='#fff4ee', marker='D', ms=9))
                continue

            if dg == 1:
                g0, g1 = g[0], g[1]
                denom = g1 * Q_int[1] - 2 * Q_int[2] * g0
                if denom == 0:
                    continue
                mu_num = [g1 * P_int[j][1] - 2 * P_int[j][2] * g0
                          for j in range(3)]
            elif dg == 2:
                # Per-root: evaluate Q'(λ) and P'(λ) at specific root
                ga, gb, g0 = g[2], g[1], g[0]
                if ga == 0:
                    continue
                sr_list = case_data.get('sr_info', [])
                ri_g = 0
                for si_idx, si_item in enumerate(sr_list):
                    if si_item is si:
                        ri_g = si_idx
                        break
                Qp_poly = [Q_int[1], 2*Q_int[2]]
                denom_sign = _eval_sign_at_root_int(Qp_poly, [g0,gb,ga], ri_g)
                if denom_sign == 0:
                    continue
                inside = True
                for j in range(3):
                    Pp_poly = [P_int[j][1], 2*P_int[j][2]]
                    mu_sign = _eval_sign_at_root_int(Pp_poly, [g0,gb,ga], ri_g)
                    if mu_sign * denom_sign < 0:
                        inside = False; break
                if not inside:
                    continue
                ring_annots.append(dict(angle=lambda_to_angle(lam_float, scale),
                    tag=sr_ring_label, lam_str=f'$\\lambda$={lam_float:.2f}',
                    color='#ff8800', bg='#fff4ee', marker='D', ms=9))
                continue
            else:
                continue

            outside = any((m > 0) != (denom > 0) and m != 0 for m in mu_num)
            if outside:
                continue
            ring_annots.append(dict(angle=lambda_to_angle(lam_float, scale),
                tag=sr_ring_label, lam_str=f'$\\lambda$={lam_float:.2f}',
                color='#ff8800', bg='#fff4ee', marker='D', ms=9))

    # D00 — only show when C++ category includes D00 (pure integer checks)
    if 'D00' in category:
        V_int = [[int(v) for v in row] for row in case_data['V']]
        W_int = [[int(w) for w in row] for row in case_data['W']]
        cv_type_r = case_data.get('Cv', 0)
        cw_type_r = case_data.get('Cw', 0)
        for vi in range(3):
            det_vw = V_int[vi][0] * W_int[vi][1] - V_int[vi][1] * W_int[vi][0]
            if det_vw == 0:
                v_zero = all(v == 0 for v in V_int[vi])
                w_zero = all(w == 0 for w in W_int[vi])
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
                        if W_int[vi][k] != 0:
                            lam_d = -float(V_int[vi][k]) / float(W_int[vi][k])
                            break
                    a = lambda_to_angle(lam_d, scale) if lam_d is not None else np.pi / 2
                    lam_str = (f'$\\lambda$={lam_d:.2f}' if lam_d is not None
                               else r'$\lambda\!=\!\infty$')
                ring_annots.append(dict(angle=a, tag='D00', lam_str=lam_str,
                    color='#9900cc', bg='#f4eeff', marker='s', ms=9))

    # D01 — only show when C++ category includes D01
    if 'D01' in category:
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

    # TN on ring — show "TN" from C++ tag, no reclassification
    if 'TN' in category:
        P_int_r = [[int(c) for c in pk] for pk in case_data['P']]
        Q_int_r = [int(c) for c in case_data['Q']]
        for k in range(3):
            a = P_int_r[k][2]
            b = P_int_r[k][1]
            c0 = P_int_r[k][0]
            if a == 0:
                continue
            disc = b * b - 4 * a * c0
            if disc != 0:
                continue
            lam_float = -b / (2.0 * a)
            q4a2 = 4*a*a*Q_int_r[0] - 2*a*b*Q_int_r[1] + b*b*Q_int_r[2]
            if q4a2 != 0:
                mu_num = []
                for j in range(3):
                    pj4a2 = (4*a*a*P_int_r[j][0]
                             - 2*a*b*P_int_r[j][1]
                             + b*b*P_int_r[j][2])
                    mu_num.append(pj4a2)
            else:
                aqp = a * Q_int_r[1] - Q_int_r[2] * b
                if aqp == 0:
                    continue
                mu_num = []
                for j in range(3):
                    apjp = a * P_int_r[j][1] - P_int_r[j][2] * b
                    mu_num.append(apjp)
            # Suppress if point is at a vertex (2 zero bary coords = D00)
            n_zero_mu = sum(1 for m in mu_num if m == 0)
            if n_zero_mu >= 2:
                continue
            ring_annots.append(dict(angle=lambda_to_angle(lam_float, scale),
                tag='TN', lam_str=f'$\\lambda$={lam_float:.2f}',
                color='#9933cc', bg='#f4eeff', marker='^', ms=8))

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
    # Extract lambda values from lam_str for distance comparison.
    # Only merge annotations at the SAME lambda (exact match from C++ tags).
    def _extract_lam(annot):
        """Extract numeric lambda from lam_str for comparison."""
        s = annot.get('lam_str', '')
        if 'infty' in s or '\\infty' in s:
            return float('inf')
        import re
        m = re.search(r'[-+]?\d*\.?\d+', s)
        return float(m.group()) if m else None

    ring_groups = []
    r_used = [False] * len(ring_annots)
    for i, a in enumerate(ring_annots):
        if r_used[i]:
            continue
        group = [a]
        r_used[i] = True
        lam_i = _extract_lam(a)
        for j in range(i + 1, len(ring_annots)):
            if r_used[j]:
                continue
            lam_j = _extract_lam(ring_annots[j])
            # Merge only if lambdas match (both inf, or same finite value)
            same = False
            if lam_i is not None and lam_j is not None:
                if lam_i == float('inf') and lam_j == float('inf'):
                    same = True
                elif lam_i != float('inf') and lam_j != float('inf'):
                    if lam_i == lam_j:
                        same = True
            if same:
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
            Wi = [int(w) for w in case_data['W'][vi]]
            Vi = [int(v) for v in case_data['V'][vi]]
            w_zero = all(w == 0 for w in Wi)
            v_zero = all(v == 0 for v in Vi)
            if w_zero:
                lam_s = '→∞'
            elif v_zero:
                lam_s = '=0.000'
            else:
                d00_lam = None
                for comp in range(2):
                    if Wi[comp] != 0:
                        d00_lam = -float(Vi[comp]) / float(Wi[comp])
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
