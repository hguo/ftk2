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

// ─── Real roots of Q[0]+Q[1]λ+Q[2]λ²+Q[3]λ³=0, sorted ─────────────────────
static std::vector<double> q_real_roots(const double Q[4]) {
    std::vector<double> roots;
    int deg = 3;
    while (deg > 0 && Q[deg] == 0.0) --deg;
    if (deg == 0) return roots;
    if (deg == 1) { roots.push_back(-Q[0]/Q[1]); return roots; }
    if (deg == 2) {
        double d = Q[1]*Q[1] - 4.0*Q[0]*Q[2];
        if (d < 0) return roots;
        double sq = std::sqrt(d);
        roots.push_back((-Q[1]-sq)/(2.0*Q[2]));
        if (d > 0) roots.push_back((-Q[1]+sq)/(2.0*Q[2]));
        std::sort(roots.begin(), roots.end());
        return roots;
    }
    // Cubic: normalize to monic x³+bx²+cx+d
    double inv = 1.0/Q[3];
    double b = Q[2]*inv, c = Q[1]*inv, d = Q[0]*inv;
    // Depress: x = t - b/3  →  t³ + pt + q = 0
    double p = c - b*b/3.0;
    double q = d - b*c/3.0 + 2.0*b*b*b/27.0;
    double shift = -b/3.0;
    double disc = -4.0*p*p*p - 27.0*q*q;  // >0: three real roots
    if (disc >= 0.0 && p < 0.0) {
        // Three real roots via trigonometric method
        double m     = 2.0*std::sqrt(-p/3.0);
        double arg   = std::max(-1.0, std::min(1.0, (1.5*q/p)*std::sqrt(-3.0/p)));
        double theta = std::acos(arg)/3.0;
        for (int k = 0; k < 3; ++k)
            roots.push_back(shift + m*std::cos(theta - 2.0*M_PI*k/3.0));
    } else {
        // One real root via Cardano
        double sq_arg = q*q/4.0 + p*p*p/27.0;
        double D = (sq_arg >= 0.0) ? std::sqrt(sq_arg) : 0.0;
        double u = std::cbrt(-q/2.0 + D);
        double v = std::cbrt(-q/2.0 - D);
        roots.push_back(shift + u + v);
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

// ─── Correct stitching for tet with any even number of punctures ─────────────
// Algorithm (no proximity):
//   1. Sort punctures by λ (stored in complex.vertices[i].scalar)
//   2. Find real roots of Q → interval boundaries
//   3. Within each Q-interval: pair adjacent punctures (0,1), (2,3), ...
static void stitch_ambiguous_tet(
    const std::vector<int>&    tet_punctures,
    const FeatureComplex&      complex,
    const double               Q_poly[4],
    bool                       q_zero,
    uint64_t                   tet_id,
    std::vector<PunctureConnection>& connections)
{
    // 1. Sort by λ
    std::vector<std::pair<double,int>> lam_idx;
    lam_idx.reserve(tet_punctures.size());
    for (int p : tet_punctures)
        lam_idx.push_back({(double)complex.vertices[p].scalar, p});
    std::sort(lam_idx.begin(), lam_idx.end());

    // 2. Build interval boundary list: {-∞, r₁, r₂, ..., +∞}
    std::vector<double> bounds = {-1e300};
    if (!q_zero) {
        auto roots = q_real_roots(Q_poly);
        for (double r : roots) bounds.push_back(r);
    }
    bounds.push_back(1e300);

    // 3. Walk through sorted punctures; emit pairs within each Q-interval
    size_t bi = 0;      // index into bounds (left edge of current interval)
    std::vector<int> group;
    for (auto& [lam, pidx] : lam_idx) {
        // Advance to the interval that contains lam
        while (bi + 2 < bounds.size() && lam > bounds[bi+1]) {
            for (size_t j = 0; j + 1 < group.size(); j += 2)
                connections.push_back({group[j], group[j+1], tet_id});
            group.clear();
            ++bi;
        }
        group.push_back(pidx);
    }
    // Emit remaining group
    for (size_t j = 0; j + 1 < group.size(); j += 2)
        connections.push_back({group[j], group[j+1], tet_id});
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

        if (tp.size() == 2) {
            n2++;
            connections.push_back({tp[0], tp[1], s.vertices[0]});
        } else if (tp.size() > 2) {
            n_more++;
            // Get Q_poly for this tet
            double V_arr[4][3], W_arr[4][3];
            for (int i = 0; i < 4; ++i) {
                auto c = mesh->get_vertex_coordinates(s.vertices[i]);
                auto fv = tc.eval(c[0], c[1], c[2]);
                for (int j = 0; j < 3; ++j) V_arr[i][j] = fv[j];
                for (int j = 0; j < 3; ++j) W_arr[i][j] = fv[3+j];
            }
            double Q_poly[4], P_poly[4][4];
            characteristic_polynomials_pv_tetrahedron(V_arr, W_arr, Q_poly, P_poly);

            bool q_zero = true;
            for (int k = 0; k <= 3; ++k) if (Q_poly[k] != 0.0) { q_zero = false; break; }

            // Q-interval + λ-sort pairing (no proximity)
            stitch_ambiguous_tet(tp, complex, Q_poly, q_zero, s.vertices[0], connections);
        }
        return true;
    });

    std::cout << "  Tet histogram: ";
    for (auto& [n,c] : tet_hist) std::cout << c << "×" << n << "  ";
    std::cout << "\n";
    std::cout << "  " << n2 << " tets with 2, " << n_more << " tets with >2 punctures\n";
    std::cout << "  " << connections.size() << " connections\n";

    // Build adjacency graph
    std::map<int,std::vector<int>> adj;
    for (auto& c : connections) {
        adj[c.puncture1_idx].push_back(c.puncture2_idx);
        adj[c.puncture2_idx].push_back(c.puncture1_idx);
    }

    // Check degrees
    int bad = 0;
    for (int i : plist) if (adj[i].size() != 2) bad++;
    std::cout << "  Punctures with degree≠2: " << bad << "\n";

    // Trace curves
    std::set<int> visited;
    std::vector<std::vector<int>> curves;
    std::vector<bool> curve_closed;
    for (int start : plist) {
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
        // PV: U×V=0 gives TWO lines:
        //   Line 1: x-r = y-p = z-q  (direction (1,1,1))
        //   Line 2: x-r = z-q, y-p = -(x-r)  (direction (1,-1,1))
        // Both lines intersect at the center (r,p,q); tets near that point
        // may carry punctures from BOTH lines → >2 punctures per tet.
        //
        // Offsets (r,p,q) chosen non-integer so the lines avoid all mesh
        // vertices, edges, and faces of the RegularSimplicialMesh, preventing
        // the "line on tet-edge" degeneracy that gave 52×1-puncture tets when
        // the center was at the integer-aligned diagonal.
        {
            "field5_cyclic_q_cubic",
            "Q cubic | 2 open PV lines (intersecting) | U=(y-p,z-q,x-r), V=(z-q,x-r,y-p)",
            [=](double x, double y, double z) -> std::array<double,6> {
                // Center deliberately off-grid and non-symmetric to avoid all
                // mesh-edge alignments.
                const double r = 8.37, p = 8.13, q = 8.51;
                return {y-p, z-q, x-r,   z-q, x-r, y-p};
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
