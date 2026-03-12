// 2D parallel-vector stitching — pure integer extraction + stitching.
//
// Architecture mirrors exact_pv_stitching.cpp (3D):
//   - Edge extraction: solve_pv_edge_2d (pure __int128, no floats)
//   - Triangle stitching: solve_pv_tri_2d (pure __int128, no floats)
//   - Floats only for approximate output coordinates
//
// Usage:  ./ftk2_exact_pv_stitching_2d [--grid-size N]

#include <ftk2/core/mesh.hpp>
#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <fstream>
#include <cstring>

using namespace ftk2;

// Field evaluation: (x,y) -> {Vx, Vy, Wx, Wy}
using FieldEval2D = std::function<std::array<double,4>(double, double)>;

struct TestCase2D {
    std::string name;
    std::string description;
    FieldEval2D eval;
};

struct Puncture2DInfo {
    uint64_t edge_v0, edge_v1;   // sorted edge vertex IDs
    int root_idx;                // which root of Q_edge (0=smaller)
    double lambda;               // approximate λ (float, for output/matching only)
    double x, y;                 // approximate position (float, for output only)
};

struct Connection2D {
    int p1, p2;
    uint64_t tri_id;
};

using EdgeKey = std::pair<uint64_t, uint64_t>;

static EdgeKey make_edge_key(uint64_t a, uint64_t b) {
    return {std::min(a,b), std::max(a,b)};
}

// Approximate roots of quadratic Q[0]+Q[1]λ+Q[2]λ² (double precision)
static int approx_roots_quadratic(const double Q[3], double roots[2]) {
    int degQ = (std::abs(Q[2]) > 1e-15) ? 2 : ((std::abs(Q[1]) > 1e-15) ? 1 : 0);
    if (degQ == 0) return 0;
    if (degQ == 1) {
        roots[0] = -Q[0] / Q[1];
        return 1;
    }
    double disc = Q[1]*Q[1] - 4*Q[2]*Q[0];
    if (disc < 0) return 0;
    double sq = std::sqrt(std::max(0.0, disc));
    roots[0] = (-Q[1] - sq) / (2*Q[2]);
    roots[1] = (-Q[1] + sq) / (2*Q[2]);
    if (roots[0] > roots[1]) std::swap(roots[0], roots[1]);
    return (disc > 0) ? 2 : 1;
}

// ─── Run one test case ───────────────────────────────────────────────────────
static void run_test_case_2d(const TestCase2D& tc, int N)
{
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)N, (uint64_t)N}
    );

    // Evaluate field at all N×N vertices
    int nv = N * N;
    std::vector<std::array<double,4>> field(nv);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            field[j*N + i] = tc.eval((double)i, (double)j);

    // ─── Edge extraction ─────────────────────────────────────────────────
    std::vector<Puncture2DInfo> all_punctures;
    std::map<EdgeKey, std::vector<int>> edge_punc_map;

    mesh->iterate_simplices(1, [&](const Simplex& s) {
        uint64_t va = s.vertices[0], vb = s.vertices[1];
        auto& fa = field[va];
        auto& fb = field[vb];

        // Quantize to __int128
        double Vd[2][2] = {{fa[0], fa[1]}, {fb[0], fb[1]}};
        double Wd[2][2] = {{fa[2], fa[3]}, {fb[2], fb[3]}};
        int64_t Vq[2][2], Wq[2][2];
        quantize_field_2x2(Vd, Vq);
        quantize_field_2x2(Wd, Wq);
        __int128 V128[2][2], W128[2][2];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) {
                V128[i][j] = (__int128)Vq[i][j];
                W128[i][j] = (__int128)Wq[i][j];
            }

        PunctureResult2D pr = solve_pv_edge_2d(V128, W128);
        if (pr.count == 0) return;

        EdgeKey ek = make_edge_key(va, vb);

        // Float Q for approximate λ
        double Qd[3];
        Qd[0] = Vd[0][0]*Vd[1][1] - Vd[0][1]*Vd[1][0];
        Qd[2] = Wd[0][0]*Wd[1][1] - Wd[0][1]*Wd[1][0];
        Qd[1] = Vd[0][0]*Wd[1][1] + Wd[0][0]*Vd[1][1]
               - Vd[0][1]*Wd[1][0] - Wd[0][1]*Vd[1][0];
        double roots[2];
        int n_real = approx_roots_quadratic(Qd, roots);

        auto ca = mesh->get_vertex_coordinates(va);
        auto cb = mesh->get_vertex_coordinates(vb);

        for (int pi = 0; pi < pr.count; pi++) {
            int ri = pr.root_idx[pi];
            double lam = (ri < n_real) ? roots[ri] : 0.0;

            // Approximate position: U₀+t(U₁-U₀) = 0 → t = U₀/(U₀-U₁)
            double U0[2] = {Vd[0][0]+lam*Wd[0][0], Vd[0][1]+lam*Wd[0][1]};
            double U1[2] = {Vd[1][0]+lam*Wd[1][0], Vd[1][1]+lam*Wd[1][1]};
            double t = 0.5;
            for (int c = 0; c < 2; c++) {
                double denom = U0[c] - U1[c];
                if (std::abs(denom) > 1e-15) { t = U0[c]/denom; break; }
            }

            Puncture2DInfo pinfo;
            pinfo.edge_v0 = ek.first;
            pinfo.edge_v1 = ek.second;
            pinfo.root_idx = ri;
            pinfo.lambda = lam;
            pinfo.x = (1-t)*ca[0] + t*cb[0];
            pinfo.y = (1-t)*ca[1] + t*cb[1];

            int gidx = (int)all_punctures.size();
            edge_punc_map[ek].push_back(gidx);
            all_punctures.push_back(pinfo);
        }
    });

    std::cout << "  " << all_punctures.size() << " edge punctures\n";

    // Per-edge histogram
    {
        std::map<int,int> hist;
        for (auto& [ek, puncs] : edge_punc_map) hist[puncs.size()]++;
        std::cout << "  Per-edge puncture histogram:\n";
        for (auto& [n, cnt] : hist)
            std::cout << "    " << cnt << " edges with " << n << " puncture(s)\n";
    }

    // ─── Triangle stitching ──────────────────────────────────────────────
    std::vector<Connection2D> connections;

    mesh->iterate_simplices(2, [&](const Simplex& s) {
        uint64_t v[3] = {s.vertices[0], s.vertices[1], s.vertices[2]};

        // Quantize
        double Vd[3][2], Wd[3][2];
        for (int i = 0; i < 3; i++) {
            auto& fi = field[v[i]];
            Vd[i][0] = fi[0]; Vd[i][1] = fi[1];
            Wd[i][0] = fi[2]; Wd[i][1] = fi[3];
        }

        __int128 Q[3], P[3][3];
        compute_tri_QP_2d_from_fields(Vd, Wd, Q, P);
        ExactPV2Result2D v2 = solve_pv_tri_2d(Q, P);
        // Face k → edge key  (face k is opposite vertex k)
        EdgeKey face_edges[3] = {
            make_edge_key(v[1], v[2]),   // face 0
            make_edge_key(v[0], v[2]),   // face 1
            make_edge_key(v[0], v[1])    // face 2
        };

        // Collect all edge punctures on this triangle's edges
        std::vector<std::pair<double,int>> lam_idx;
        for (int k = 0; k < 3; k++) {
            auto it = edge_punc_map.find(face_edges[k]);
            if (it == edge_punc_map.end()) continue;
            for (int pidx : it->second)
                lam_idx.push_back({all_punctures[pidx].lambda, pidx});
        }
        if ((int)lam_idx.size() < 2) return;

        // Sort by λ (float, for matching — topology already decided by v2)
        std::sort(lam_idx.begin(), lam_idx.end());

        uint64_t tri_id = *std::min_element(v, v+3);

        if (v2.n_pairs > 0 && v2.n_punctures == (int)lam_idx.size()) {
            // Direct index mapping: v2 puncture i ↔ sorted edge puncture i
            for (int pi = 0; pi < v2.n_pairs; pi++) {
                int a = v2.pairs[pi].a, b = v2.pairs[pi].b;
                if (a < (int)lam_idx.size() && b < (int)lam_idx.size())
                    connections.push_back({lam_idx[a].second, lam_idx[b].second, tri_id});
            }
        } else {
            // Fallback: pair consecutive punctures sorted by λ.
            // Handles pass-through cases where v2 finds 0 pairs
            // (all P[k] share common factor → reduced P_red has no roots).
            for (size_t j = 0; j + 1 < lam_idx.size(); j += 2)
                connections.push_back({lam_idx[j].second, lam_idx[j+1].second, tri_id});
        }
    });

    std::cout << "  " << connections.size() << " connections (before dedup)\n";

    // Deduplicate connections (shared-edge triangles may produce same pair)
    {
        std::set<std::pair<int,int>> seen;
        std::vector<Connection2D> unique;
        for (auto& c : connections) {
            auto key = std::make_pair(std::min(c.p1, c.p2), std::max(c.p1, c.p2));
            if (seen.insert(key).second)
                unique.push_back(c);
        }
        connections = std::move(unique);
    }
    std::cout << "  " << connections.size() << " connections (after dedup)\n";

    // ─── Curve tracing ───────────────────────────────────────────────────
    std::map<int,std::vector<int>> adj;
    for (auto& c : connections) {
        adj[c.p1].push_back(c.p2);
        adj[c.p2].push_back(c.p1);
    }

    std::map<int,int> degree_hist;
    for (auto& [p, nbs] : adj) degree_hist[nbs.size()]++;
    std::cout << "  Degree histogram: ";
    for (auto& [d,c] : degree_hist) std::cout << c << "×deg" << d << " ";
    std::cout << "\n";

    // Trace: start from degree-1 endpoints, then degree-2 (closed curves)
    std::set<int> visited;
    std::vector<std::vector<int>> curves;
    std::vector<bool> curve_closed;

    std::vector<int> starts;
    for (auto& [p, nbs] : adj) if (nbs.size() == 1) starts.push_back(p);
    for (auto& [p, nbs] : adj) if (nbs.size() == 2) starts.push_back(p);

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
    for (size_t i = 0; i < curves.size(); i++)
        std::cout << " [" << curves[i].size() << "pts,"
                  << (curve_closed[i] ? "closed" : "open") << "]";
    std::cout << "\n";

    // Write CSV
    std::string fname = tc.name + "_curves.csv";
    std::ofstream ofs(fname);
    ofs << "curve_id,point_idx,x,y,lambda\n";
    for (size_t ci = 0; ci < curves.size(); ci++) {
        for (size_t pi = 0; pi < curves[ci].size(); pi++) {
            int idx = curves[ci][pi];
            ofs << ci << "," << pi << ","
                << all_punctures[idx].x << "," << all_punctures[idx].y << ","
                << all_punctures[idx].lambda << "\n";
        }
        // Close the curve in CSV
        if (curve_closed[ci] && !curves[ci].empty()) {
            int idx = curves[ci][0];
            ofs << ci << "," << curves[ci].size() << ","
                << all_punctures[idx].x << "," << all_punctures[idx].y << ","
                << all_punctures[idx].lambda << "\n";
        }
    }
    ofs.close();
    std::cout << "  Wrote " << fname << "\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    int N = 32;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--grid-size") == 0 && i+1 < argc)
            N = atoi(argv[++i]);
    }

    // Center at half-integer to avoid PV curve passing through any mesh vertex
    const double cx = (N-1)/2.0;  // e.g. 15.5 for N=32
    const double cy = cx;

    std::vector<TestCase2D> cases = {
        // ── F1: Diagonal line ───────────────────────────────────────────
        // V = (x-cx, y-cy+0.2), W = (1, 1)
        // det(V,W) = (x-cx) - (y-cy+0.2) = x - y - 0.2
        // PV locus: x = y + 0.2  (diagonal line, avoids all integer vertices)
        // Q constant (W constant), P[k] linear → simple case
        // Expected: 1 open curve
        {
            "field1_diagonal_line",
            "det = x-y-0.2 | 1 open diagonal line",
            [=](double x, double y) -> std::array<double,4> {
                return {x-cx, y-cy+0.2, 1.0, 1.0};
            }
        },
        // ── F2: Circle ──────────────────────────────────────────────────
        // V = (x-cx, y-cy-R), W = (y-cy+R, -(x-cx))
        // det(V,W) = -(x-cx)² - (y-cy)² + R²
        // PV locus: (x-cx)² + (y-cy)² = R²  (circle)
        // R = 10.3 ensures circle avoids all vertices
        // Q quadratic (W varies), P[k] quadratic → generic case
        // Expected: 1 closed curve
        {
            "field2_circle",
            "det = R²-r² | 1 closed circle (R=10.3)",
            [=](double x, double y) -> std::array<double,4> {
                const double R = 10.3;
                return {x-cx, y-cy-R, y-cy+R, -(x-cx)};
            }
        },
        // ── F3: Two vertical lines ──────────────────────────────────────
        // V = ((x-cx)²-R², y-cy+0.3), W = (0, 1)
        // det(V,W) = (x-cx)² - R²
        // PV locus: x = cx ± R = 10.2 and 20.8 (avoids integer vertices)
        // W constant → Q linear, P[k] linear → no pass-through
        // Expected: 2 open curves
        {
            "field3_two_lines",
            "det = (x-cx)^2 - R^2 | 2 open vertical lines (R=5.3)",
            [=](double x, double y) -> std::array<double,4> {
                const double R = 5.3;
                double vx = (x-cx)*(x-cx) - R*R;
                return {vx, y-cy+0.3, 0.0, 1.0};
            }
        },
        // ── F4: Horizontal line ─────────────────────────────────────────
        // V = (x-cx, y-cy+0.3), W = (1, 0)
        // det(V,W) = -(y-cy+0.3) = -(y-15.8)
        // PV locus: y = 15.8  (horizontal line, avoids all vertices)
        // Q constant (W constant), P[k] linear → simple case
        // Expected: 1 open curve
        {
            "field4_horizontal_line",
            "det = -(y-15.8) | 1 open horizontal line",
            [=](double x, double y) -> std::array<double,4> {
                return {x-cx, y-cy+0.3, 1.0, 0.0};
            }
        },
    };

    for (const auto& tc : cases) {
        std::cout << "\n" << std::string(60,'=') << "\n";
        std::cout << "CASE: " << tc.name << "\n";
        std::cout << "DESC: " << tc.description << "\n";
        std::cout << std::string(60,'=') << "\n";
        run_test_case_2d(tc, N);
    }
    return 0;
}
