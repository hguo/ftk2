#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>

using namespace ftk2;

// Field evaluation: (x,y,z) -> {Ux,Uy,Uz,Vx,Vy,Vz}
using FieldEval = std::function<std::array<double,6>(double,double,double)>;

struct TestCase {
    std::string name;
    std::string description;
    FieldEval   eval;
};

struct PunctureConnection {
    int puncture1_idx, puncture2_idx;
    uint64_t tet_id;
};

// ─── GCD-normalize __int128 polynomial for safe double Sturm chain ───────────
// Divides all coefficients by their GCD so products in S₂/S₃ fit in double.
// Without this, quantized coefficients (~2^52–2^73) overflow in S₂ products.
static void gcd_normalize_poly(const __int128 P[4], double P_d[4]) {
    __int128 g = gcd_i128(gcd_i128(P[0] < 0 ? -P[0] : P[0],
                                    P[1] < 0 ? -P[1] : P[1]),
                           gcd_i128(P[2] < 0 ? -P[2] : P[2],
                                    P[3] < 0 ? -P[3] : P[3]));
    if (g == 0) g = 1;
    for (int i = 0; i < 4; ++i) P_d[i] = (double)(P[i] / g);
}

// ─── Shared-root detection via Sylvester resultant ───────────────────────────
// Returns true iff Q and some P[k] share a common root, i.e. Res(Q, P[k]) = 0.
// Uses exact __int128 Bareiss fraction-free elimination (no thresholds).
static bool has_shared_root_resultant(const __int128 Q_i128[4],
                                      const __int128 P_i128[4][4])
{
    int degQ = 3;
    while (degQ > 0 && Q_i128[degQ] == 0) degQ--;
    if (degQ == 0) return false;

    for (int k = 0; k < 4; ++k) {
        int degP = 3;
        while (degP > 0 && P_i128[k][degP] == 0) degP--;
        if (degP == 0) continue;

        int N = degQ + degP;
        __int128 M[6][6] = {};
        for (int i = 0; i < degP; i++)
            for (int j = 0; j <= degQ; j++)
                M[i][i + degQ - j] = Q_i128[j];
        for (int i = 0; i < degQ; i++)
            for (int j = 0; j <= degP; j++)
                M[degP + i][i + degP - j] = P_i128[k][j];

        // Bareiss fraction-free elimination: exact integer determinant
        __int128 prev_pivot = 1;
        bool zero_det = false;
        for (int col = 0; col < N; col++) {
            int pivot = -1;
            for (int row = col; row < N; row++)
                if (M[row][col] != 0) { pivot = row; break; }
            if (pivot < 0) { zero_det = true; break; }
            if (pivot != col)
                for (int j = 0; j < N; j++)
                    std::swap(M[col][j], M[pivot][j]);
            for (int row = col + 1; row < N; row++) {
                for (int j = col + 1; j < N; j++)
                    M[row][j] = (M[col][col] * M[row][j]
                               - M[row][col] * M[col][j]) / prev_pivot;
                M[row][col] = 0;
            }
            prev_pivot = M[col][col];
        }
        if (zero_det || M[N-1][N-1] == 0) return true;
    }
    return false;
}

// ─── Fully combinatorial stitching for tet with >2 punctures ─────────────────
// Algorithm:
//   1. Sort punctures by λ
//   2. Compute Sturm count of Q-roots below each puncture's λ (certified)
//   3. Build effective groups: merge adjacent intervals when Q-roots between
//      them are pass-throughs (detected via Sturm counts on P[k])
//   4. Pair (0,1),(2,3),... within each effective group
static void stitch_ambiguous_tet(
    const std::vector<int>&    tet_punctures,
    const FeatureComplex&      complex,
    const __int128             Q_i128[4],
    const __int128             P_i128[4][4],
    uint64_t                   tet_id,
    std::vector<PunctureConnection>& connections)
{
    int n = (int)tet_punctures.size();

    // 1. Sort by λ
    std::vector<std::pair<double,int>> lam_idx;
    lam_idx.reserve(n);
    for (int p : tet_punctures)
        lam_idx.push_back({(double)complex.vertices[p].scalar, p});
    std::sort(lam_idx.begin(), lam_idx.end());

    // 2. GCD-normalize Q_i128 → double, determine effective degree
    //    Without normalization, quantized coefficients (~2^52–2^73) cause
    //    overflow in S₂/S₃ products of build_sturm_double.
    double Q_d[4];
    gcd_normalize_poly(Q_i128, Q_d);
    int degQ = 3;
    while (degQ > 0 && Q_d[degQ] == 0.0) --degQ;

    if (degQ == 0) {
        // Q is constant → no poles → single group → pair all
        for (int j = 0; j + 1 < n; j += 2)
            connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
        return;
    }

    // Build Sturm sequence for Q (GCD-normalized coefficients fit in double)
    SturmSeqDouble Q_seq;
    build_sturm_double(Q_d, Q_seq);

    // 3. For each puncture, compute Sturm count (# Q-roots below λ)
    std::vector<int> qi(n);
    for (int i = 0; i < n; ++i) {
        double lam = lam_idx[i].first;
        auto [count, cert] = sturm_count_at_certified(Q_seq, lam);
        if (!cert) {
            // SoS perturbation: fixed +delta (any fixed direction works;
            // must not depend on the uncertifiable float value itself)
            double delta = 4.0 * std::numeric_limits<double>::epsilon()
                         * std::max(1.0, std::abs(lam));
            count = sturm_count_at(Q_seq, lam + delta);
        }
        qi[i] = count;
    }

    // 4. Build effective groups with shared-root detection
    //    If Q and some P[k] share a root (Sylvester resultant = 0), ALL
    //    Q-roots are pass-throughs and all intervals merge into one group.
    bool is_sr = has_shared_root_resultant(Q_i128, P_i128);
    std::vector<int> qi_eff(n);
    qi_eff[0] = 0;
    for (int i = 1; i < n; ++i) {
        if (qi[i] == qi[i-1]) {
            qi_eff[i] = qi_eff[i-1];          // same Q-interval
        } else if (is_sr) {
            qi_eff[i] = qi_eff[i-1];          // shared root → merge
        } else {
            qi_eff[i] = qi_eff[i-1] + 1;      // genuine pole → new group
        }
    }

    // 5. Group by qi_eff, pair (0,1),(2,3),... within each group
    int group_start = 0;
    for (int i = 1; i <= n; ++i) {
        if (i == n || qi_eff[i] != qi_eff[i-1]) {
            // Emit pairs from group_start to i-1
            for (int j = group_start; j + 1 < i; j += 2)
                connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
            group_start = i;
        }
    }
}

// ─── Write curves to VTP ─────────────────────────────────────────────────────
static void write_curves(
    const std::vector<std::vector<int>>&  curves,   // each curve: list of puncture indices
    const std::vector<bool>&              closed,
    const FeatureComplex&                 complex,
    std::shared_ptr<Mesh>                 mesh,
    const std::string&                    filename)
{
    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    auto points   = vtkSmartPointer<vtkPoints>::New();
    auto lines    = vtkSmartPointer<vtkCellArray>::New();

    std::map<int,vtkIdType> pid_to_vtk;
    vtkIdType next_id = 0;

    // Insert unique points
    for (size_t ci = 0; ci < curves.size(); ++ci) {
        for (int p : curves[ci]) {
            if (pid_to_vtk.count(p)) continue;
            const auto& v = complex.vertices[p];
            double pos[3] = {0,0,0};
            for (int i = 0; i < 3; ++i) {
                auto vc = mesh->get_vertex_coordinates(v.simplex.vertices[i]);
                for (int j = 0; j < 3; ++j) pos[j] += v.barycentric_coords[0][i]*vc[j];
            }
            points->InsertNextPoint(pos);
            pid_to_vtk[p] = next_id++;
        }
    }

    // Insert polyline cells
    int total_segs = 0;
    for (size_t ci = 0; ci < curves.size(); ++ci) {
        std::vector<vtkIdType> cell;
        for (int p : curves[ci]) cell.push_back(pid_to_vtk[p]);
        if (closed[ci] && !cell.empty()) cell.push_back(cell[0]);
        lines->InsertNextCell((vtkIdType)cell.size(), cell.data());
        total_segs += (int)cell.size() - 1;
    }

    polydata->SetPoints(points);
    polydata->SetLines(lines);

    auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polydata);
    writer->SetDataModeToAscii();
    writer->Write();

    std::cout << "  Wrote " << filename << ": "
              << points->GetNumberOfPoints() << " pts, "
              << curves.size() << " curve(s), "
              << total_segs << " segs\n";
}

// ─── Run one test case ────────────────────────────────────────────────────────
static void run_test_case(const TestCase& tc, int N)
{
    // Build ndarray from field eval
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)N,(uint64_t)N,(uint64_t)N});
    ftk::ndarray<double> uv({6, (size_t)N, (size_t)N, (size_t)N});
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                auto fv = tc.eval((double)x, (double)y, (double)z);
                for (int c = 0; c < 6; ++c)
                    uv.f(c, x, y, z) = fv[c];
            }

    std::map<std::string, ftk::ndarray<double>> data;
    data["uv"] = uv;

    // Extract punctures
    ExactPVPredicate<double> pred;
    pred.vector_var_name = "uv";
    SimplicialEngine<double, ExactPVPredicate<double>> engine(mesh, pred);
    engine.execute(data, {"uv"});
    auto complex = engine.get_complex();

    // Collect non-degenerate punctures (dimension==2 face)
    std::vector<int> plist;
    for (size_t i = 0; i < complex.vertices.size(); ++i)
        if (complex.vertices[i].simplex.dimension == 2)
            plist.push_back((int)i);
    std::cout << "  " << plist.size() << " face punctures\n";

    // Per-triangle histogram
    {
        std::map<std::set<uint64_t>,int> tri_cnt;
        for (int i : plist) {
            const auto& v = complex.vertices[i];
            tri_cnt[{v.simplex.vertices, v.simplex.vertices+3}]++;
        }
        std::map<int,int> hist;
        for (auto& [k,v] : tri_cnt) hist[v]++;
        std::cout << "  Per-triangle puncture histogram:\n";
        for (auto& [n,cnt] : hist)
            std::cout << "    " << cnt << " triangles with " << n << " puncture(s)\n";
    }

    // D00 global deduplication: a vertex puncture (two barycentric coords = 0)
    // can be claimed by multiple faces incident to the same mesh vertex.  The
    // solver D00 rule (claim iff v_m is triangle minimum) emits one record per
    // qualifying face, but we need exactly one globally.  Keep the puncture from
    // the lexicographically smallest face (vertex-index tuple), which is unique.
    {
        // Map from mesh-vertex-id → list of D00 puncture indices at that vertex.
        // For integer-derived barycentric coords, zero bary coords are exactly 0.0.
        std::map<uint64_t, std::vector<int>> vtx_g2;
        for (int p : plist) {
            const auto& v = complex.vertices[p];
            const float* bc = v.barycentric_coords[0];
            int n_zero = 0, m = -1;
            for (int k = 0; k < 3; k++) {
                if (bc[k] == 0.0f) n_zero++;
                else m = k;
            }
            if (n_zero == 2 && m >= 0)
                vtx_g2[v.simplex.vertices[m]].push_back(p);
        }
        std::set<int> to_remove;
        for (auto& [vid, puncs] : vtx_g2) {
            if (puncs.size() <= 1) continue;
            // Canonical = puncture whose face has the lexicographically smallest
            // sorted vertex tuple.
            int canonical = puncs[0];
            std::vector<uint64_t> can_f(complex.vertices[canonical].simplex.vertices,
                                        complex.vertices[canonical].simplex.vertices + 3);
            std::sort(can_f.begin(), can_f.end());
            for (int p : puncs) {
                std::vector<uint64_t> f(complex.vertices[p].simplex.vertices,
                                        complex.vertices[p].simplex.vertices + 3);
                std::sort(f.begin(), f.end());
                if (f < can_f) { canonical = p; can_f = f; }
            }
            for (int p : puncs) if (p != canonical) to_remove.insert(p);
        }
        if (!to_remove.empty()) {
            plist.erase(std::remove_if(plist.begin(), plist.end(),
                        [&](int p){ return to_remove.count(p) > 0; }), plist.end());
            std::cout << "  D00 global dedup: removed " << to_remove.size()
                      << " duplicate vertex punctures\n";
        }
    }

    // Build face → puncture list map
    std::map<std::set<uint64_t>, std::vector<int>> face_to_punc;
    for (int i : plist) {
        const auto& v = complex.vertices[i];
        face_to_punc[{v.simplex.vertices, v.simplex.vertices+3}].push_back(i);
    }

    // Stitch through tetrahedra
    std::vector<PunctureConnection> connections;
    int n2=0, n_more=0;
    std::map<int,int> tet_hist;

    mesh->iterate_simplices(3, [&](const Simplex& s) {
        std::vector<std::set<uint64_t>> faces = {
            {s.vertices[0],s.vertices[1],s.vertices[2]},
            {s.vertices[0],s.vertices[1],s.vertices[3]},
            {s.vertices[0],s.vertices[2],s.vertices[3]},
            {s.vertices[1],s.vertices[2],s.vertices[3]}
        };
        std::vector<int> tp;
        for (auto& f : faces)
            if (face_to_punc.count(f))
                for (int p : face_to_punc[f]) tp.push_back(p);
        if (tp.empty()) return true;

        tet_hist[tp.size()]++;

        // Use min(vertices) as tet_id for SoS consistency
        uint64_t tet_id = *std::min_element(s.vertices, s.vertices + 4);

        if (tp.size() == 2) {
            n2++;
            connections.push_back({tp[0], tp[1], tet_id});
        } else if (tp.size() > 2) {
            n_more++;
            // Get exact integer Q and P polynomials for this tet
            double V_arr[4][3], W_arr[4][3];
            for (int i = 0; i < 4; ++i) {
                auto c = mesh->get_vertex_coordinates(s.vertices[i]);
                auto fv = tc.eval(c[0], c[1], c[2]);
                for (int j = 0; j < 3; ++j) V_arr[i][j] = fv[j];
                for (int j = 0; j < 3; ++j) W_arr[i][j] = fv[3+j];
            }
            __int128 Q_i128[4], P_i128[4][4];
            compute_tet_QP_i128(V_arr, W_arr, Q_i128, P_i128);

            // Exact Q≡0 check (all integer coefficients zero)
            bool q_zero = (Q_i128[0] == 0 && Q_i128[1] == 0 &&
                           Q_i128[2] == 0 && Q_i128[3] == 0);

            if (!q_zero) {
                // Sturm-based combinatorial stitching
                stitch_ambiguous_tet(tp, complex, Q_i128, P_i128, tet_id, connections);
            } else {
                // Q≡0: no poles at all → single group → pair by λ-sort
                std::vector<std::pair<double,int>> lam_idx;
                for (int p : tp)
                    lam_idx.push_back({(double)complex.vertices[p].scalar, p});
                std::sort(lam_idx.begin(), lam_idx.end());
                for (size_t j = 0; j + 1 < lam_idx.size(); j += 2)
                    connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
            }
        }
        return true;
    });

    std::cout << "  Tet histogram: ";
    for (auto& [n,c] : tet_hist) std::cout << c << "×" << n << "  ";
    std::cout << "\n";
    std::cout << "  " << n2 << " tets with 2, " << n_more << " tets with >2 punctures\n";
    std::cout << "  " << connections.size() << " connections (before dedup)\n";

    // Deduplicate connections: shared-face tets may both produce the same pair
    {
        std::set<std::pair<int,int>> seen;
        std::vector<PunctureConnection> unique;
        for (auto& c : connections) {
            auto key = std::make_pair(std::min(c.puncture1_idx, c.puncture2_idx),
                                      std::max(c.puncture1_idx, c.puncture2_idx));
            if (seen.insert(key).second)
                unique.push_back(c);
        }
        connections = std::move(unique);
    }
    std::cout << "  " << connections.size() << " connections (after dedup)\n";

    // Build adjacency graph
    std::map<int,std::vector<int>> adj;
    for (auto& c : connections) {
        adj[c.puncture1_idx].push_back(c.puncture2_idx);
        adj[c.puncture2_idx].push_back(c.puncture1_idx);
    }

    // Check degrees
    int bad = 0;
    std::map<int,int> degree_hist;
    for (int i : plist) {
        degree_hist[adj[i].size()]++;
        if (adj[i].size() != 2) bad++;
    }
    std::cout << "  Punctures with degree≠2: " << bad << "  (";
    for (auto& [d,c] : degree_hist) std::cout << c << "×deg" << d << " ";
    std::cout << ")\n";

    // Trace curves — start from degree-1 endpoints first to avoid
    // fragmenting open paths, then trace remaining closed curves.
    std::set<int> visited;
    std::vector<std::vector<int>> curves;
    std::vector<bool> curve_closed;

    // Collect start candidates: degree-1 (endpoints) first, then degree-2
    std::vector<int> starts;
    for (int p : plist) if (adj[p].size() == 1) starts.push_back(p);
    for (int p : plist) if (adj[p].size() == 2) starts.push_back(p);

    for (int start : starts) {
        if (visited.count(start) || adj[start].empty()) continue;
        std::vector<int> path;
        int curr = start, prev = -1;
        bool closed = false;
        while (true) {
            if (visited.count(curr)) { closed = (curr == start); break; }
            visited.insert(curr);
            path.push_back(curr);
            int next = -1;
            for (int nb : adj[curr]) if (nb != prev) { next = nb; break; }
            if (next == -1) break;
            prev = curr; curr = next;
        }
        if (path.size() > 1) {
            curves.push_back(std::move(path));
            curve_closed.push_back(closed);
        }
    }
    std::cout << "  " << curves.size() << " curve(s):";
    for (size_t i = 0; i < curves.size(); ++i)
        std::cout << " [" << curves[i].size() << "pts,"
                  << (curve_closed[i] ? "closed" : "open") << "]";
    std::cout << "\n";

    write_curves(curves, curve_closed, complex, mesh, tc.name + ".vtp");
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main()
{
    const int N = 16;
    const double cx = N/2.0, cy = N/2.0;
    const double z0 = N/2.0 + 0.5;   // half-integer to avoid plane coincidence
    const double R  = N/3.0;         // single-circle radius
    const double R1 = N/4.5;         // inner concentric radius
    const double R2 = N/3.0;         // outer concentric radius
    const double z1 = N/3.0 + 0.5;  // lower stacked circle height
    const double z2 = 2.0*N/3.0 + 0.5; // upper stacked circle height

    std::vector<TestCase> cases = {
        // ── F1: baseline, single circle, Q≡0 ────────────────────────────────
        // U constant → A=0.  V_x=1 constant → det(B)=0.  Q≡0.
        // PV locus: z=z0, r=R.  λ=1 everywhere on PV curve.
        // Expected: 1 closed curve.
        {
            "field1_one_circle",
            "Q≡0 | 1 circle | U=(1,0,0), V=(1,z-z0,r²-R²)",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,z-z0,r2-R*R};
            }
        },
        // ── F2: two circles at different heights, Q≡0 ───────────────────────
        // V_y = (z-z1)(z-z2) vanishes at z=z1 and z=z2.
        // PV locus: two circles, one at z=z1 and one at z=z2, both radius R.
        // Expected: 2 closed curves.
        {
            "field2_two_stacked_circles",
            "Q≡0 | 2 circles (stacked) | U=(1,0,0), V=(1,(z-z1)(z-z2),r²-R²)",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,(z-z1)*(z-z2),r2-R*R};
            }
        },
        // ── F3: two concentric circles at same height, Q≡0 ──────────────────
        // V_z = (r²-R1²)(r²-R2²) vanishes on two cylinders r=R1 and r=R2.
        // PV locus: two concentric circles at z=z0.
        // Within each tet straddling R1 or R2: one puncture (linear approx).
        // Expected: 2 closed curves.
        {
            "field3_two_concentric_circles",
            "Q≡0 | 2 concentric circles | U=(1,0,0), V=(1,z-z0,(r²-R1²)(r²-R2²))",
            [=](double x, double y, double z) -> std::array<double,6> {
                double r2 = (x-cx)*(x-cx)+(y-cy)*(y-cy);
                return {1,0,0, 1,z-z0,(r2-R1*R1)*(r2-R2*R2)};
            }
        },
        // ── F4: V constant → Q = det(A) = constant (non-zero) ───────────────
        // B=0 → Q(λ) = det(A) = constant (degree-0 polynomial).
        // No Q-roots → single interval (-∞,+∞) → pair by λ-sort.
        // PV: U ∥ (1,0,0) → y=cy AND z=z0 → a horizontal line in x.
        // λ at a puncture = (x-cx)/1 = x-cx (varies along line).
        // Expected: 1 open line (enters/exits mesh boundary).
        {
            "field4_v_constant",
            "Q=det(A)≠0 (constant) | open PV line | U=(x-cx,y-cy,z-z0), V=(1,0,0)",
            [=](double x, double y, double z) -> std::array<double,6> {
                return {x-cx, y-cy, z-z0, 1,0,0};
            }
        },
        // ── F5: cyclic field with irrational offsets → Q genuine cubic ───────
        // U=(y-p,z-q,x-r), V=(z-q,x-r,y-p): both vary in all 3 directions.
        // Q(λ)=det(A-λB) is a genuine cubic per tet (both A and B full-rank).
        //
        // PV: U×V=0.  Let a=y-p, b=z-q, c=x-r.  U=(a,b,c), V=(b,c,a).
        // U×V = (ab-c², bc-a², ac-b²) = 0 → a³=c³ → a=c; then b=a.
        // So the PV set is ONE line: x-r = y-p = z-q, direction (1,1,1).
        //
        // Offsets (r,p,q) chosen non-integer so the line avoids all mesh
        // vertices, edges, and faces of the RegularSimplicialMesh, preventing
        // the "line on tet-edge" degeneracy that gave 52×1-puncture tets when
        // the center was at the integer-aligned diagonal.
        {
            "field5_cyclic_q_cubic",
            "Q cubic | 1 open PV line (dir (1,1,1)) | U=(y-p,z-q,x-r), V=(z-q,x-r,y-p)",
            [=](double x, double y, double z) -> std::array<double,6> {
                // Center deliberately off-grid and non-symmetric to avoid all
                // mesh-edge alignments.
                const double r = 8.37, p = 8.13, q = 8.51;
                return {y-p, z-q, x-r,   z-q, x-r, y-p};
            }
        },
        // ── F6: three PV lines through a common center → 6 punctures/tet ────
        // U = S·x', V = x'  where x' = (x-cx6, y-cy6, z-z06)
        // S = symmetric matrix with eigenvalues 1,2,3 and eigenvectors:
        //   d1=(1,1,1)/√3  (λ=1), d2=(1,-1,0)/√2  (λ=2), d3=(1,1,-2)/√6 (λ=3)
        //   S = [[11/6, -1/6, -2/3], [-1/6, 11/6, -2/3], [-2/3, -2/3, 7/3]]
        //
        // PV: S·x' = λ·x' → x' is eigenvector of S with eigenvalue λ.
        //   Three PV lines through center in directions d1, d2, d3 with
        //   λ=1, 2, 3 respectively.
        //
        // Q(λ) = det(A-λB).  Since A = S·B (B = V-differences),
        //   Q(λ) = det((S-λI)·B) = (1-λ)(2-λ)(3-λ)·det(B).
        // So Q-roots are exactly λ=1,2,3 — one per PV line.
        //
        // For the tet containing the center: all 3 lines pass through →
        //   6 punctures (2 per line) with λ∈{1,1,2,2,3,3}.
        //   stitch_ambiguous_tet groups {1,1} → (1,2], {2,2} → (2,3] correctly.
        // Adjacent tets intersected by 2 lines get 4 punctures.
        //
        // Center chosen non-integer so no line lies on any mesh edge or vertex.
        {
            "field6_three_pv_lines",
            "Q cubic | 3 open PV lines (distinct λ=1,2,3) | U=S·x', V=x'",
            [](double x, double y, double z) -> std::array<double,6> {
                const double cx6 = 8.37, cy6 = 8.13, z06 = 8.51;
                double xp = x - cx6, yp = y - cy6, zp = z - z06;
                // U = S * (xp, yp, zp)
                // S = 1*d1⊗d1 + 2*d2⊗d2 + 3*d3⊗d3
                // S[0][0]=11/6, S[0][1]=-1/6, S[0][2]=-2/3
                // S[1][1]=11/6, S[1][2]=-2/3,  S[2][2]=7/3
                double ux = (11.0/6)*xp + (-1.0/6)*yp + (-2.0/3)*zp;
                double uy = (-1.0/6)*xp + (11.0/6)*yp + (-2.0/3)*zp;
                double uz = (-2.0/3)*xp + (-2.0/3)*yp + (7.0/3)*zp;
                // V = x'
                return {ux, uy, uz, xp, yp, zp};
            }
        },
    };

    for (const auto& tc : cases) {
        std::cout << "\n" << std::string(60,'=') << "\n";
        std::cout << "CASE: " << tc.name << "\n";
        std::cout << "DESC: " << tc.description << "\n";
        std::cout << std::string(60,'=') << "\n";
        run_test_case(tc, N);
    }
    return 0;
}
