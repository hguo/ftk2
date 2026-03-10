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

// ─── ExactPV2 stitching via solve_pv_tet_v2 ──────────────────────────────────
// Uses pure-integer per-tet solver for topological decisions.
// Float only for spatial output coordinates (barycentric interpolation).
static void stitch_v2(
    const std::vector<int>&    plist,
    const FeatureComplex&      complex,
    std::shared_ptr<Mesh>      mesh,
    const FieldEval&           eval,
    std::vector<PunctureConnection>& connections)
{
    mesh->iterate_simplices(3, [&](const Simplex& s) {
        // Collect face → puncture map for this tet
        std::map<std::set<uint64_t>, std::vector<int>> face_puncs;
        for (int p : plist) {
            const auto& v = complex.vertices[p];
            std::set<uint64_t> face(v.simplex.vertices, v.simplex.vertices + 3);
            // Check if face belongs to this tet
            bool belongs = true;
            for (uint64_t fv : face) {
                bool found = false;
                for (int i = 0; i < 4; i++) if (s.vertices[i] == fv) { found = true; break; }
                if (!found) { belongs = false; break; }
            }
            if (belongs) face_puncs[face].push_back(p);
        }
        if (face_puncs.empty()) return true;

        // Get field values at tet vertices
        double V_arr[4][3], W_arr[4][3];
        for (int i = 0; i < 4; i++) {
            auto c = mesh->get_vertex_coordinates(s.vertices[i]);
            auto fv = eval(c[0], c[1], c[2]);
            for (int j = 0; j < 3; j++) V_arr[i][j] = fv[j];
            for (int j = 0; j < 3; j++) W_arr[i][j] = fv[3+j];
        }

        __int128 Q_i128[4], P_i128[4][4];
        compute_tet_QP_i128(V_arr, W_arr, Q_i128, P_i128);

        ExactPV2Result v2 = solve_pv_tet_v2(Q_i128, P_i128);

        // Collect all puncture indices from this tet
        std::vector<int> tp;
        for (auto& [face, puncs] : face_puncs)
            for (int p : puncs) tp.push_back(p);

        if (tp.size() < 2) return true;

        uint64_t tet_id = *std::min_element(s.vertices, s.vertices + 4);

        // If v2 found pairs, use its pairing.  Otherwise fall back to simple λ-sort.
        if (v2.n_pairs > 0 && (int)tp.size() >= 2) {
            // Match v2 punctures to actual puncture indices by λ ordering.
            // Sort tp by λ (float, for matching only — topological decision was already made by v2)
            std::vector<std::pair<double,int>> lam_idx;
            for (int p : tp)
                lam_idx.push_back({(double)complex.vertices[p].scalar, p});
            std::sort(lam_idx.begin(), lam_idx.end());

            // v2 punctures are already sorted by λ.  Map v2 index → actual puncture.
            // If counts match, direct mapping works.
            if (v2.n_punctures == (int)lam_idx.size()) {
                for (int pi = 0; pi < v2.n_pairs; pi++) {
                    int a = v2.pairs[pi].a, b = v2.pairs[pi].b;
                    if (a < (int)lam_idx.size() && b < (int)lam_idx.size())
                        connections.push_back({lam_idx[a].second, lam_idx[b].second, tet_id});
                }
            } else {
                // Mismatch in counts — fall back to simple pairing
                for (size_t j = 0; j + 1 < lam_idx.size(); j += 2)
                    connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tet_id});
            }
        } else if (tp.size() == 2) {
            connections.push_back({tp[0], tp[1], tet_id});
        }
        return true;
    });
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

    // Stitch through tetrahedra using ExactPV2 (pure-integer solver)
    std::vector<PunctureConnection> connections;
    std::cout << "  Stitching (pure integer)...\n";
    stitch_v2(plist, complex, mesh, tc.eval, connections);

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
int main(int argc, char** argv)
{
    (void)argc; (void)argv;

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
