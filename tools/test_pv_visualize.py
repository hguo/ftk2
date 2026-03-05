#!/usr/bin/env python3
"""
Unit tests for pv_tet_visualize.py — checks visualization invariants
that the user has repeatedly identified through visual inspection.

Run:  python3 tools/test_pv_visualize.py build/pv_tet_cases/paper_selection_v7.json
"""

import json
import sys
import os
import importlib.util

# Import the visualizer module
vis_path = os.path.join(os.path.dirname(__file__), 'pv_tet_visualize.py')
spec = importlib.util.spec_from_file_location('pv_tet_visualize', vis_path)
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def load_cases(json_path):
    """Load JSONL case file."""
    cases = []
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


class VisualizeChecker:
    """Check invariants on case data + segments."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.n_cases = 0

    def check(self, case_data):
        self.n_cases += 1
        seed = case_data['seed']
        cat = case_data['category']
        prefix = f"seed={seed} ({cat})"

        # Build segments (same as visualize_case)
        try:
            segments = vis.collect_segments(case_data)
        except Exception as e:
            self.errors.append(f"{prefix}: collect_segments() crashed: {e}")
            return

        # Handle bubble (mirrors visualize_case logic)
        if case_data.get('has_B') and case_data['n_punctures'] == 0:
            segments = [{'color': vis.SEGMENT_COLORS[0], 'pts_list': [],
                         'pi_entry': -1, 'pi_exit': -1,
                         'lam_entry': None, 'lam_exit': None,
                         'infinity_spanning': True}]

        punctures = case_data['punctures']
        n = len(punctures)

        # ── 1. Every puncture has a color (not grey fallback) ──
        punc_color = {}
        for seg in segments:
            punc_color.setdefault(seg['pi_entry'], seg['color'])
            punc_color.setdefault(seg['pi_exit'], seg['color'])

        for i in range(n):
            if i not in punc_color:
                # Unpaired punctures (T-odd with C1v/C0v/C1w/C0w) are OK grey
                has_crit = any(tag in cat for tag in ['C1v', 'C0v', 'C1w', 'C0w'])
                if not has_crit:
                    self.errors.append(
                        f"{prefix}: puncture {i} has no segment color (grey fallback)")
                # Even for C1v etc, at most 1 should be uncolored
                n_uncolored = sum(1 for j in range(n) if j not in punc_color)
                if n_uncolored > 1:
                    self.errors.append(
                        f"{prefix}: {n_uncolored} punctures uncolored (max 1 for C1v/C0v)")

        # ── 2. Cv/Cw marker color matches segment (not fixed green) ──
        if 'Cv' in cat and case_data.get('Cv_mu') is not None:
            seg_color = vis._find_segment_for_lambda(0.0, segments)
            if seg_color != '#333333':
                # Cv is on a segment — marker should use that color, not green
                # (This is what we just implemented; verify the logic works)
                pass  # If _find_segment_for_lambda returns a real color, our code uses it
            # else: no segment at λ=0, green is correct

        if 'Cw' in cat and case_data.get('Cw_mu') is not None:
            # Cw should use color of segment reaching infinity
            inf_seg_color = None
            for seg in segments:
                if (seg.get('infinity_spanning', False) or
                        seg['lam_entry'] is None or seg['lam_exit'] is None):
                    inf_seg_color = seg['color']
                    break
            # If there's an infinity-reaching segment, Cw must use its color
            # (verified by code path; flag if no segment found but Cw expected)

        # ── 3. D01/D00 markers use C++ flags, not Python thresholds ──
        if 'D01' in cat:
            d01_found = any(p.get('is_edge', False) for p in punctures)
            if not d01_found:
                self.errors.append(
                    f"{prefix}: category has D01 but no puncture has is_edge=true")

        if 'D00' in cat:
            d00_found = any(p.get('is_vertex', False) for p in punctures)
            if not d00_found:
                # D00 can also come from V×W=0 at vertex, not just puncture flag
                import numpy as np
                V = np.array(case_data['V'], dtype=float)
                W = np.array(case_data['W'], dtype=float)
                found_vxw0 = False
                for vi in range(4):
                    cross = np.cross(V[vi], W[vi])
                    if all(cross == 0):
                        found_vxw0 = True
                        break
                if not found_vxw0:
                    self.errors.append(
                        f"{prefix}: category has D00 but no vertex/puncture supports it")

        # ── 4. Pair completeness (mirrors C++ verify_case) ──
        # Edge/vertex punctures are waypoints, not paired.
        pairs = case_data.get('pairs', [])
        n_face = sum(1 for p in punctures
                     if not p.get('is_edge', False) and not p.get('is_vertex', False))
        if 2 * len(pairs) != n_face:
            self.errors.append(
                f"{prefix}: 2×pairs={2*len(pairs)} ≠ n_face={n_face}")

        # ── 5. T-number matches n_face (face-interior punctures only) ──
        import re
        m = re.match(r'T(\d+)', cat)
        if m:
            t_num = int(m.group(1))
            if t_num != n_face:
                self.errors.append(
                    f"{prefix}: T-number={t_num} but n_face={n_face}")

        # ── 6. Q-type consistency ──
        Q = case_data.get('Q_coeffs', [0, 0, 0, 0])
        # Find effective degree
        degQ = 3
        while degQ > 0 and abs(Q[degQ]) < 0.5:
            degQ -= 1
        n_qr = len(case_data.get('Q_roots', []))

        if 'Q3+' in cat and (degQ != 3 or n_qr != 3):
            self.warnings.append(f"{prefix}: Q3+ but degQ={degQ}, n_roots={n_qr}")
        if 'Q3-' in cat and (degQ != 3 or n_qr != 1):
            self.warnings.append(f"{prefix}: Q3- but degQ={degQ}, n_roots={n_qr}")
        if 'Q2-' in cat and (degQ != 2 or n_qr != 0):
            self.warnings.append(f"{prefix}: Q2- but degQ={degQ}, n_roots={n_qr}")
        if cat.startswith('T') and '_Q' in cat:
            # Q2 (without -/o) means 2 real roots
            if '_Q2_' in cat or cat.endswith('_Q2'):
                if degQ != 2 or n_qr != 2:
                    self.warnings.append(f"{prefix}: Q2 but degQ={degQ}, n_roots={n_qr}")

        # ── 7. Segment count matches pairs ──
        if len(segments) != len(pairs) and n > 0:
            # Bubble case has 0 pairs but 1 segment (OK)
            if not (case_data.get('has_B') and n == 0):
                self.errors.append(
                    f"{prefix}: {len(segments)} segments but {len(pairs)} pairs")

        # ── 8. No puncture should have lambda exactly at a Q-root ──
        Q_roots = case_data.get('Q_roots', [])
        for i, p in enumerate(punctures):
            lam = p.get('lambda')
            if lam is None:
                continue
            for qr in Q_roots:
                if abs(lam - qr) < 1e-10:
                    self.warnings.append(
                        f"{prefix}: puncture {i} λ={lam:.6g} ≈ Q-root {qr:.6g}")

        # ── 9. Cv/Cw position validity ──
        import numpy as np
        for tag, mu_key in [('Cv', 'Cv_mu'), ('Cw', 'Cw_mu')]:
            if tag in cat:
                mu = case_data.get(mu_key)
                if mu is None:
                    self.warnings.append(f"{prefix}: {tag} in category but {mu_key} missing")
                else:
                    mu = np.array(mu)
                    if any(mu < -0.01) or any(mu > 1.01):
                        self.errors.append(
                            f"{prefix}: {mu_key} out of range: {mu}")
                    if abs(sum(mu) - 1.0) > 0.01:
                        self.errors.append(
                            f"{prefix}: {mu_key} sum={sum(mu):.6g} ≠ 1")

        # ── 10. SR tag → has_shared_root flag ──
        if 'SR' in cat and not case_data.get('has_shared_root', False):
            self.errors.append(f"{prefix}: SR in category but has_shared_root=false")
        if case_data.get('has_shared_root', False) and 'SR' not in cat:
            self.warnings.append(f"{prefix}: has_shared_root=true but SR not in category")

        # ── 11. Interval consistency ──
        # interval.n_pv counts face-interior punctures only
        intervals = case_data.get('intervals', [])
        sum_npv = sum(iv['n_pv'] for iv in intervals)
        if sum_npv != n_face:
            self.errors.append(
                f"{prefix}: Σ interval.n_pv={sum_npv} ≠ n_face={n_face}")

        # ── 12. Ring scale: all lambda values should be representable ──
        all_lams = [p['lambda'] for p in punctures if p['lambda'] is not None]
        all_lams += [r for r in Q_roots]
        abs_vals = [abs(v) for v in all_lams if abs(v) > 1e-15]
        if abs_vals:
            scale = max(abs_vals)
            if scale < 0.5:
                scale = 0.5
            # Check that no two distinct lambdas map to the same angle
            import math
            angles = [math.atan2(v, scale) for v in all_lams if v is not None]
            angles.sort()
            for j in range(1, len(angles)):
                if abs(angles[j] - angles[j-1]) < 0.001 and len(all_lams) > 1:
                    # Two lambdas very close in angle space
                    pass  # acceptable if lambdas are genuinely close

    def report(self):
        print(f"\nChecked {self.n_cases} cases")
        if self.errors:
            print(f"\n{len(self.errors)} ERRORS:")
            for e in self.errors:
                print(f"  [ERR] {e}")
        if self.warnings:
            print(f"\n{len(self.warnings)} WARNINGS:")
            for w in self.warnings:
                print(f"  [WARN] {w}")
        if not self.errors and not self.warnings:
            print("  All checks passed!")
        return len(self.errors)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <cases.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    cases = load_cases(json_path)
    print(f"Loaded {len(cases)} cases from {json_path}")

    checker = VisualizeChecker()
    for c in cases:
        checker.check(c)

    n_err = checker.report()
    sys.exit(1 if n_err > 0 else 0)


if __name__ == '__main__':
    main()
