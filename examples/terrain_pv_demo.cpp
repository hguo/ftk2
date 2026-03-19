// 2D Terrain Ridge/Valley Extraction Demo
//
// Extracts ridge and valley lines from 2D terrain elevation data using
// ExactPV 2D (gradient ∥ Hessian·gradient).
//
// Architecture: triangle-only extraction via solve_pv_tri_2d.
//   - No separate edge extraction step
//   - Each triangle yields punctures (face + root_idx) and pairs
//   - Punctures registered globally by (edge_key, root_idx)
//   - Pairs become connections; dedup + curve tracing follows
//
// Output:
//   terrain.vti          — elevation grid (ImageData)
//   terrain_punctures.vtp — all PV puncture points (no stitching)
//   terrain_ridges.vtp   — stitched ridge/valley curves (PolyData)
//
// Usage:
//   ./ftk2_terrain_pv_demo [--grid-size N] [--input file.bin --nx NX --ny NY]

#include <ftk2/core/mesh.hpp>
#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <iostream>
#include <iomanip>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <cassert>
#include <tuple>

using namespace ftk2;

// ─── Puncture / connection structs ──────────────────────────────────────────

struct Puncture2DInfo {
    uint64_t edge_v0, edge_v1;
    int root_idx;
    double lambda;
    double x, y;
    double vx, vy;
    double wx, wy;
};

struct Connection2D {
    int p1, p2;
    uint64_t tri_id;
};

using EdgeKey = std::pair<uint64_t, uint64_t>;

static EdgeKey make_edge_key(uint64_t a, uint64_t b) {
    return {std::min(a,b), std::max(a,b)};
}

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

// ─── MATLAB peaks function ──────────────────────────────────────────────────

static double peaks(double x, double y) {
    return 3.0*(1-x)*(1-x)*std::exp(-x*x - (y+1)*(y+1))
         - 10.0*(x/5.0 - x*x*x - y*y*y*y*y)*std::exp(-x*x - y*y)
         - (1.0/3.0)*std::exp(-(x+1)*(x+1) - y*y);
}

// ─── Bilinear elevation interpolation ───────────────────────────────────────

static double interp_elevation(double px, double py,
                               const std::vector<double>& elevation,
                               int nx, int ny,
                               double ox, double oy, double dx, double dy)
{
    double gx = (px - ox) / dx;
    double gy = (py - oy) / dy;
    int ix = std::max(0, std::min(nx-2, (int)std::floor(gx)));
    int iy = std::max(0, std::min(ny-2, (int)std::floor(gy)));
    double fx = gx - ix, fy = gy - iy;
    return (1-fx)*(1-fy)*elevation[iy*nx + ix]
         + fx*(1-fy)*elevation[iy*nx + ix+1]
         + (1-fx)*fy*elevation[(iy+1)*nx + ix]
         + fx*fy*elevation[(iy+1)*nx + ix+1];
}

// ─── VTI writer ─────────────────────────────────────────────────────────────

static void write_vti(const std::string& filename,
                      const std::vector<double>& elevation,
                      int nx, int ny,
                      double ox, double oy, double dx, double dy)
{
    std::ofstream f(filename);
    f << std::setprecision(15);
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <ImageData WholeExtent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 0\""
      << " Origin=\"" << ox << " " << oy << " 0\""
      << " Spacing=\"" << dx << " " << dy << " 1\">\n";
    f << "    <Piece Extent=\"0 " << (nx-1) << " 0 " << (ny-1) << " 0 0\">\n";
    f << "      <PointData Scalars=\"elevation\">\n";
    f << "        <DataArray type=\"Float64\" Name=\"elevation\" format=\"ascii\">\n";
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++)
            f << "          " << elevation[j * nx + i] << "\n";
    f << "        </DataArray>\n";
    f << "      </PointData>\n";
    f << "    </Piece>\n";
    f << "  </ImageData>\n";
    f << "</VTKFile>\n";
    f.close();
    std::cout << "Wrote " << filename << " (" << nx << "x" << ny << ")\n";
}

// ─── VTP writer for puncture points ─────────────────────────────────────────

static void write_punctures_vtp(const std::string& filename,
                                const std::vector<Puncture2DInfo>& punctures,
                                const std::vector<double>& elevation,
                                int nx, int ny,
                                double ox, double oy, double dx, double dy)
{
    int np = (int)punctures.size();
    std::ofstream f(filename);
    f << std::setprecision(15);
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <PolyData>\n";
    f << "    <Piece NumberOfPoints=\"" << np
      << "\" NumberOfVerts=\"1\" NumberOfLines=\"0\">\n";

    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < np; i++) {
        double z = interp_elevation(punctures[i].x, punctures[i].y,
                                    elevation, nx, ny, ox, oy, dx, dy);
        f << "          " << punctures[i].x << " " << punctures[i].y << " " << z << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    f << "      <Verts>\n";
    f << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    f << "         ";
    for (int i = 0; i < np; i++) f << " " << i;
    f << "\n";
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    f << "          " << np << "\n";
    f << "        </DataArray>\n";
    f << "      </Verts>\n";

    f << "      <PointData Scalars=\"elevation\">\n";
    for (const char* name : {"elevation", "lambda"}) {
        f << "        <DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
        for (int i = 0; i < np; i++) {
            double val = (std::string(name) == "elevation")
                ? interp_elevation(punctures[i].x, punctures[i].y, elevation, nx, ny, ox, oy, dx, dy)
                : punctures[i].lambda;
            f << "          " << val << "\n";
        }
        f << "        </DataArray>\n";
    }
    f << "        <DataArray type=\"Float64\" Name=\"v\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < np; i++)
        f << "          " << punctures[i].vx << " " << punctures[i].vy << " 0\n";
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Float64\" Name=\"w\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < np; i++)
        f << "          " << punctures[i].wx << " " << punctures[i].wy << " 0\n";
    f << "        </DataArray>\n";
    f << "      </PointData>\n";

    f << "    </Piece>\n";
    f << "  </PolyData>\n";
    f << "</VTKFile>\n";
    f.close();
    std::cout << "Wrote " << filename << " (" << np << " points)\n";
}

// ─── VTP writer for stitched curves ─────────────────────────────────────────

static void write_curves_vtp(const std::string& filename,
                             const std::vector<std::vector<int>>& curves,
                             const std::vector<bool>& curve_closed,
                             const std::vector<Puncture2DInfo>& punctures,
                             const std::vector<double>& elevation,
                             int nx, int ny,
                             double ox, double oy, double dx, double dy)
{
    std::vector<int> pt_indices;
    for (size_t ci = 0; ci < curves.size(); ci++) {
        for (int idx : curves[ci]) pt_indices.push_back(idx);
        if (curve_closed[ci] && !curves[ci].empty())
            pt_indices.push_back(curves[ci][0]);
    }
    int total_pts = (int)pt_indices.size();

    std::ofstream f(filename);
    f << std::setprecision(15);
    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    f << "  <PolyData>\n";
    f << "    <Piece NumberOfPoints=\"" << total_pts
      << "\" NumberOfLines=\"" << curves.size() << "\">\n";

    f << "      <Points>\n";
    f << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < total_pts; i++) {
        auto& p = punctures[pt_indices[i]];
        f << "          " << p.x << " " << p.y << " "
          << interp_elevation(p.x, p.y, elevation, nx, ny, ox, oy, dx, dy) << "\n";
    }
    f << "        </DataArray>\n";
    f << "      </Points>\n";

    f << "      <Lines>\n";
    f << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    f << "         ";
    for (int i = 0; i < total_pts; i++) f << " " << i;
    f << "\n";
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    f << "         ";
    int offset = 0;
    for (size_t ci = 0; ci < curves.size(); ci++) {
        offset += (int)curves[ci].size() + (curve_closed[ci] ? 1 : 0);
        f << " " << offset;
    }
    f << "\n";
    f << "        </DataArray>\n";
    f << "      </Lines>\n";

    f << "      <PointData Scalars=\"elevation\">\n";
    f << "        <DataArray type=\"Float64\" Name=\"elevation\" format=\"ascii\">\n";
    for (int i = 0; i < total_pts; i++) {
        auto& p = punctures[pt_indices[i]];
        f << "          " << interp_elevation(p.x, p.y, elevation, nx, ny, ox, oy, dx, dy) << "\n";
    }
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Float64\" Name=\"lambda\" format=\"ascii\">\n";
    for (int i = 0; i < total_pts; i++)
        f << "          " << punctures[pt_indices[i]].lambda << "\n";
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Float64\" Name=\"v\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < total_pts; i++) {
        auto& p = punctures[pt_indices[i]];
        f << "          " << p.vx << " " << p.vy << " 0\n";
    }
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Float64\" Name=\"w\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i = 0; i < total_pts; i++) {
        auto& p = punctures[pt_indices[i]];
        f << "          " << p.wx << " " << p.wy << " 0\n";
    }
    f << "        </DataArray>\n";
    f << "        <DataArray type=\"Int32\" Name=\"curve_id\" format=\"ascii\">\n";
    f << "         ";
    for (size_t ci = 0; ci < curves.size(); ci++) {
        int n = (int)curves[ci].size() + (curve_closed[ci] ? 1 : 0);
        for (int i = 0; i < n; i++) f << " " << (int)ci;
    }
    f << "\n";
    f << "        </DataArray>\n";
    f << "      </PointData>\n";

    f << "    </Piece>\n";
    f << "  </PolyData>\n";
    f << "</VTKFile>\n";
    f.close();
    std::cout << "Wrote " << filename << " (" << curves.size() << " curves, "
              << total_pts << " points)\n";
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int N = 128;
    std::string input_file;
    int input_nx = 0, input_ny = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--grid-size") == 0 && i+1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--input") == 0 && i+1 < argc)
            input_file = argv[++i];
        else if (strcmp(argv[i], "--nx") == 0 && i+1 < argc)
            input_nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ny") == 0 && i+1 < argc)
            input_ny = atoi(argv[++i]);
    }

    // ─── Step 1: Generate or load elevation ─────────────────────────────
    int nx, ny;
    double ox, oy, dx, dy;
    std::vector<double> elevation;

    if (!input_file.empty()) {
        assert(input_nx > 0 && input_ny > 0);
        nx = input_nx; ny = input_ny;
        elevation.resize(nx * ny);
        std::ifstream fin(input_file, std::ios::binary);
        fin.read(reinterpret_cast<char*>(elevation.data()), nx * ny * sizeof(double));
        fin.close();
        ox = 0; oy = 0; dx = 1; dy = 1;
        std::cout << "Loaded " << input_file << " (" << nx << "x" << ny << ")\n";
    } else {
        nx = N; ny = N;
        ox = -3.0; oy = -3.0;
        dx = 6.0 / (N - 1);
        dy = 6.0 / (N - 1);
        elevation.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double x = ox + i * dx;
                double y = oy + j * dy;
                elevation[j * nx + i] = peaks(x, y);
            }
        std::cout << "Generated peaks terrain (" << nx << "x" << ny << ")\n";
    }

    {
        double emin = elevation[0], emax = elevation[0];
        for (double e : elevation) { emin = std::min(emin, e); emax = std::max(emax, e); }
        std::cout << "  Elevation range: [" << emin << ", " << emax << "]\n";
    }

    // ─── Step 2: Compute gradient V and Hessian·gradient W ──────────────
    int M = nx - 2;
    int My = ny - 2;
    std::vector<std::array<double,2>> V_field(My * M);
    std::vector<std::array<double,2>> W_field(My * M);

    for (int j = 0; j < My; j++) {
        for (int i = 0; i < M; i++) {
            int gi = i + 1, gj = j + 1;

            double fx_p = elevation[gj * nx + (gi+1)];
            double fx_m = elevation[gj * nx + (gi-1)];
            double fy_p = elevation[(gj+1) * nx + gi];
            double fy_m = elevation[(gj-1) * nx + gi];
            double f_c  = elevation[gj * nx + gi];

            double Vx = fx_p - fx_m;
            double Vy = fy_p - fy_m;

            double Hxx = fx_p - 2*f_c + fx_m;
            double Hyy = fy_p - 2*f_c + fy_m;
            double Hxy = (elevation[(gj+1)*nx + (gi+1)] - elevation[(gj+1)*nx + (gi-1)]
                        - elevation[(gj-1)*nx + (gi+1)] + elevation[(gj-1)*nx + (gi-1)]) * 0.25;

            double Wx = Hxx * Vx + Hxy * Vy;
            double Wy = Hxy * Vx + Hyy * Vy;

            V_field[j * M + i] = {Vx, Vy};
            W_field[j * M + i] = {Wx, Wy};
        }
    }

    double max_v = 0, max_w = 0;
    for (int k = 0; k < My * M; k++) {
        max_v = std::max(max_v, std::max(std::abs(V_field[k][0]), std::abs(V_field[k][1])));
        max_w = std::max(max_w, std::max(std::abs(W_field[k][0]), std::abs(W_field[k][1])));
    }
    if (max_v > 0) for (auto& v : V_field) { v[0] /= max_v; v[1] /= max_v; }
    if (max_w > 0) for (auto& w : W_field) { w[0] /= max_w; w[1] /= max_w; }

    std::cout << "V,W on " << M << "x" << My << " interior grid"
              << " (max_v=" << max_v << ", max_w=" << max_w << ")\n";

    // ─── Step 3: Triangle-only PV extraction ────────────────────────────
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)M, (uint64_t)My}
    );

    int nv = M * My;
    std::vector<std::array<double,4>> field(nv);
    for (int k = 0; k < nv; k++)
        field[k] = {V_field[k][0], V_field[k][1], W_field[k][0], W_field[k][1]};

    // Global puncture registry: (edge_v0, edge_v1, root_idx) → global index
    // For D01 (vertex punctures): (vertex_id, vertex_id, 0)
    using PuncKey = std::tuple<uint64_t, uint64_t, int>;
    std::map<PuncKey, int> punc_registry;
    std::vector<Puncture2DInfo> all_punctures;
    std::vector<Connection2D> connections;

    // Face k of triangle (v0,v1,v2): opposite vertex k
    // face 0 → det(U₁, U₂) → edge (v[1], v[2])
    // face 1 → det(U₂, U₀) → edge (v[2], v[0])
    // face 2 → det(U₀, U₁) → edge (v[0], v[1])
    static const int face_v[3][2] = {{1,2}, {2,0}, {0,1}};

    // Triangle statistics
    std::map<int,int> tri_punc_hist;
    int n_tri_passthrough = 0, n_tri_edge_punc = 0, n_tri_inf_punc = 0;
    int n_tri_merge_inf = 0, n_tri_multi_interval = 0, n_tri_total = 0;

    mesh->iterate_simplices(2, [&](const Simplex& s) {
        uint64_t v[3] = {s.vertices[0], s.vertices[1], s.vertices[2]};
        n_tri_total++;

        double Vd[3][2], Wd[3][2];
        for (int i = 0; i < 3; i++) {
            auto& fi = field[v[i]];
            Vd[i][0] = fi[0]; Vd[i][1] = fi[1];
            Wd[i][0] = fi[2]; Wd[i][1] = fi[3];
        }

        __int128 Q[3], P[3][3];
        compute_tri_QP_2d_from_fields(Vd, Wd, Q, P);
        ExactPV2Result2D v2 = solve_pv_tri_2d(Q, P);

        tri_punc_hist[v2.n_punctures]++;
        if (v2.has_passthrough) n_tri_passthrough++;
        if (v2.merge_infinity) n_tri_merge_inf++;
        if (v2.n_qr_roots > 1) n_tri_multi_interval++;
        for (int i = 0; i < v2.n_punctures; i++) {
            if (v2.punctures[i].is_edge) n_tri_edge_punc++;
            if (v2.punctures[i].root_idx == -1) n_tri_inf_punc++;
        }

        if (v2.n_punctures == 0) return;

        // ── Register punctures and record pairs ──────────────────────
        auto register_puncture = [&](PuncKey key, int face, int root_idx) -> int {
            auto it = punc_registry.find(key);
            if (it != punc_registry.end()) return it->second;

            Puncture2DInfo pinfo;
            pinfo.edge_v0 = std::get<0>(key);
            pinfo.edge_v1 = std::get<1>(key);
            pinfo.root_idx = root_idx;

            if (std::get<0>(key) == std::get<1>(key)) {
                uint64_t vid = std::get<0>(key);
                auto coords = mesh->get_vertex_coordinates(vid);
                pinfo.x = ox + (coords[0] + 1.0) * dx;
                pinfo.y = oy + (coords[1] + 1.0) * dy;
                pinfo.lambda = 0;
                pinfo.vx = field[vid][0]; pinfo.vy = field[vid][1];
                pinfo.wx = field[vid][2]; pinfo.wy = field[vid][3];
            } else {
                int j = face_v[face][0], l = face_v[face][1];
                double Pk[3];
                Pk[0] = Vd[j][0]*Vd[l][1] - Vd[j][1]*Vd[l][0];
                Pk[2] = Wd[j][0]*Wd[l][1] - Wd[j][1]*Wd[l][0];
                Pk[1] = Vd[j][0]*Wd[l][1] + Wd[j][0]*Vd[l][1]
                      - Vd[j][1]*Wd[l][0] - Wd[j][1]*Vd[l][0];
                double Uj[2], Ul[2];
                if (root_idx >= 0) {
                    double roots[2];
                    int n_real = approx_roots_quadratic(Pk, roots);
                    double lam = (root_idx < n_real) ? roots[root_idx] : 0.0;
                    pinfo.lambda = lam;
                    Uj[0] = Vd[j][0]+lam*Wd[j][0]; Uj[1] = Vd[j][1]+lam*Wd[j][1];
                    Ul[0] = Vd[l][0]+lam*Wd[l][0]; Ul[1] = Vd[l][1]+lam*Wd[l][1];
                } else {
                    pinfo.lambda = 1e30;
                    Uj[0] = Wd[j][0]; Uj[1] = Wd[j][1];
                    Ul[0] = Wd[l][0]; Ul[1] = Wd[l][1];
                }
                double t = 0.5;
                for (int c = 0; c < 2; c++) {
                    double denom = Uj[c] - Ul[c];
                    if (std::abs(denom) > 1e-15) { t = Uj[c]/denom; break; }
                }
                t = std::max(0.0, std::min(1.0, t));
                auto cj = mesh->get_vertex_coordinates(v[j]);
                auto cl = mesh->get_vertex_coordinates(v[l]);
                double mx = (1-t)*cj[0] + t*cl[0];
                double my = (1-t)*cj[1] + t*cl[1];
                pinfo.x = ox + (mx + 1.0) * dx;
                pinfo.y = oy + (my + 1.0) * dy;
                pinfo.vx = (1-t)*Vd[j][0] + t*Vd[l][0];
                pinfo.vy = (1-t)*Vd[j][1] + t*Vd[l][1];
                pinfo.wx = (1-t)*Wd[j][0] + t*Wd[l][0];
                pinfo.wy = (1-t)*Wd[j][1] + t*Wd[l][1];
            }
            int gidx = (int)all_punctures.size();
            punc_registry[key] = gidx;
            all_punctures.push_back(pinfo);
            return gidx;
        };

        int local_to_global[ExactPV2Result2D::MAX_PUNCTURES];
        for (int pi = 0; pi < v2.n_punctures; pi++) {
            auto& punct = v2.punctures[pi];
            PuncKey key;
            if (punct.is_edge) {
                int vk = 3 - punct.edge_faces[0] - punct.edge_faces[1];
                key = {v[vk], v[vk], 0};
            } else {
                int j = face_v[punct.face][0], l = face_v[punct.face][1];
                EdgeKey ek = make_edge_key(v[j], v[l]);
                key = {ek.first, ek.second, punct.root_idx};
            }
            local_to_global[pi] = register_puncture(key, punct.face, punct.root_idx);
        }

        uint64_t tri_id = *std::min_element(v, v+3);
        for (int pi = 0; pi < v2.n_pairs; pi++) {
            int a = v2.pairs[pi].a, b = v2.pairs[pi].b;
            connections.push_back({local_to_global[a], local_to_global[b], tri_id});
        }
    });

    std::cout << all_punctures.size() << " punctures (from triangle solver)\n";
    if (!all_punctures.empty()) {
        double xmin = all_punctures[0].x, xmax = xmin;
        double ymin = all_punctures[0].y, ymax = ymin;
        for (auto& p : all_punctures) {
            xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y);
        }
        std::cout << "  Range: [" << xmin << "," << xmax
                  << "] x [" << ymin << "," << ymax << "]\n";
    }

    std::cout << "\n  Triangle statistics (" << n_tri_total << " total):\n";
    std::cout << "    v2 puncture histogram:\n";
    for (auto& [n, cnt] : tri_punc_hist)
        std::cout << "      " << cnt << " triangles with " << n << " puncture(s)\n";
    std::cout << "    Passthrough:    " << n_tri_passthrough << "\n";
    std::cout << "    D01 (edge):     " << n_tri_edge_punc << "\n";
    std::cout << "    Cw (infinity):  " << n_tri_inf_punc << "\n";
    std::cout << "    Merge-infinity: " << n_tri_merge_inf << "\n";
    std::cout << "    Multi Q-intv:   " << n_tri_multi_interval << "\n";

    write_punctures_vtp("terrain_punctures.vtp", all_punctures,
                        elevation, nx, ny, ox, oy, dx, dy);

    // ─── Deduplicate connections ─────────────────────────────────────────
    // Cross-edge pairs: both adjacent triangles produce the same pair
    //   → keep 1 (standard dedup)
    // Same-edge pairs (bubble): both punctures on same edge, each triangle
    //   produces the pair → keep BOTH (they are distinct curve segments,
    //   one in each triangle, forming a closed loop)
    {
        std::map<std::pair<int,int>, int> seen;
        std::vector<Connection2D> unique;
        for (auto& c : connections) {
            auto key = std::make_pair(std::min(c.p1, c.p2), std::max(c.p1, c.p2));
            int count = seen[key]++;
            bool same_edge =
                all_punctures[c.p1].edge_v0 == all_punctures[c.p2].edge_v0 &&
                all_punctures[c.p1].edge_v1 == all_punctures[c.p2].edge_v1 &&
                all_punctures[c.p1].edge_v0 != all_punctures[c.p1].edge_v1;
            if (count == 0 || (same_edge && count == 1))
                unique.push_back(c);
        }
        connections = std::move(unique);
    }
    std::cout << connections.size() << " connections\n";

    // ─── Edge-following curve tracer ─────────────────────────────────────
    // Handles multi-edges (same-edge bubble pairs) correctly by tracking
    // used connections rather than visited nodes.
    std::map<int, std::vector<std::pair<int,int>>> adj;  // node → [(neighbor, conn_idx)]
    for (int i = 0; i < (int)connections.size(); i++) {
        adj[connections[i].p1].push_back({connections[i].p2, i});
        adj[connections[i].p2].push_back({connections[i].p1, i});
    }

    {
        std::map<int,int> dhist;
        for (auto& [p, nbs] : adj) dhist[nbs.size()]++;
        std::cout << "  Degree histogram: ";
        for (auto& [d,c] : dhist) std::cout << c << "x deg" << d << "  ";
        std::cout << "\n";
        int deg0 = (int)all_punctures.size() - (int)adj.size();
        if (deg0 > 0) std::cout << "  " << deg0 << " punctures with degree 0\n";
    }

    // A puncture is a legitimate open endpoint if it's on the spatial
    // boundary OR at λ→∞ (Cw).  Cw endpoints are inherent to the RP1
    // analysis: the Q-gate correctly excludes them from the neighbor
    // triangle (degP[k] >= degQ_red → μ_k ≠ 0 at ∞), so they're
    // one-sided by design, analogous to spatial boundary exits.
    auto is_legitimate_endpoint = [&](int pidx) {
        auto& p = all_punctures[pidx];
        // Spatial boundary
        double mx = (p.x - ox) / dx - 1.0;
        double my = (p.y - oy) / dy - 1.0;
        if (mx < 0.5 || mx > M - 1.5 || my < 0.5 || my > My - 1.5) return true;
        // Cw at infinity (ri=-1): curve exits at λ→∞
        if (p.root_idx == -1) return true;
        return false;
    };

    std::vector<bool> edge_used(connections.size(), false);
    std::vector<std::vector<int>> curves;
    std::vector<bool> curve_closed;

    // Start from degree-1 nodes first (open curves), then degree-2 (closed)
    std::vector<int> starts;
    for (auto& [p, nbs] : adj) if (nbs.size() == 1) starts.push_back(p);
    for (auto& [p, nbs] : adj) if (nbs.size() >= 2) starts.push_back(p);

    for (int start : starts) {
        // Check if start has any unused edge
        bool has_unused = false;
        for (auto& [nb, eid] : adj[start])
            if (!edge_used[eid]) { has_unused = true; break; }
        if (!has_unused) continue;

        std::vector<int> path;
        int curr = start;
        bool closed = false;
        while (true) {
            path.push_back(curr);
            int next = -1;
            for (auto& [nb, eid] : adj[curr]) {
                if (!edge_used[eid]) {
                    edge_used[eid] = true;
                    next = nb;
                    break;
                }
            }
            if (next == -1) break;
            if (next == path[0]) { closed = true; break; }
            curr = next;
        }
        if (path.size() > 1) {
            curves.push_back(std::move(path));
            curve_closed.push_back(closed);
        }
    }

    int n_open = 0, n_closed = 0;
    int n_bdy = 0, n_int = 0;
    for (size_t ci = 0; ci < curves.size(); ci++) {
        if (curve_closed[ci]) { n_closed++; continue; }
        n_open++;
        if (!is_legitimate_endpoint(curves[ci].front())) n_int++;
        else n_bdy++;
        if (!is_legitimate_endpoint(curves[ci].back())) n_int++;
        else n_bdy++;
    }
    std::cout << curves.size() << " curves (" << n_open << " open, " << n_closed << " closed)\n";
    std::cout << "  Open endpoints: " << n_bdy << " boundary, " << n_int << " interior\n";

    // Verify no jumps beyond sqrt(2) grid spacing between consecutive points
    {
        double max_step = std::sqrt(2.0) * std::max(dx, dy) * 1.01;  // sqrt(2) + tolerance
        int n_jumps = 0;
        for (size_t ci = 0; ci < curves.size(); ci++) {
            int n = (int)curves[ci].size();
            for (int i = 1; i < n; i++) {
                auto& a = all_punctures[curves[ci][i-1]];
                auto& b = all_punctures[curves[ci][i]];
                double d = std::hypot(a.x - b.x, a.y - b.y);
                if (d > max_step) {
                    n_jumps++;
                    std::cout << "  JUMP curve " << ci << " [" << (i-1) << "]->[" << i
                              << "] dist=" << d/std::max(dx,dy) << " cells"
                              << " edge(" << a.edge_v0 << "," << a.edge_v1 << ")"
                              << "->edge(" << b.edge_v0 << "," << b.edge_v1 << ")"
                              << " (" << a.x << "," << a.y << ")->(" << b.x << "," << b.y << ")\n";
                }
            }
            // Also check closing segment for closed curves
            if (curve_closed[ci] && n >= 2) {
                auto& a = all_punctures[curves[ci][n-1]];
                auto& b = all_punctures[curves[ci][0]];
                double d = std::hypot(a.x - b.x, a.y - b.y);
                if (d > max_step) {
                    n_jumps++;
                    std::cout << "  JUMP curve " << ci << " [" << (n-1) << "]->[0:close]"
                              << " dist=" << d/std::max(dx,dy) << " cells\n";
                }
            }
        }
        std::cout << "  Jump check (>" << max_step << "): " << n_jumps << " violations\n";
    }

    // Per-curve summary
    for (size_t ci = 0; ci < curves.size(); ci++) {
        int n = (int)curves[ci].size();
        if (n <= 4 || ci < 12) {
            std::cout << "    curve " << ci << ": " << n << " pts, "
                      << (curve_closed[ci] ? "closed" : "open");
            // Check for duplicate consecutive points
            int n_dup = 0;
            for (int i = 1; i < n; i++) {
                auto& a = all_punctures[curves[ci][i-1]];
                auto& b = all_punctures[curves[ci][i]];
                if (std::abs(a.x-b.x) < 1e-12 && std::abs(a.y-b.y) < 1e-12) n_dup++;
            }
            if (n_dup > 0) std::cout << " (" << n_dup << " dup)";
            // Check same-edge pairs
            int n_same = 0;
            for (int i = 1; i < n; i++) {
                auto& a = all_punctures[curves[ci][i-1]];
                auto& b = all_punctures[curves[ci][i]];
                if (a.edge_v0 == b.edge_v0 && a.edge_v1 == b.edge_v1 && a.edge_v0 != a.edge_v1)
                    n_same++;
            }
            if (n_same > 0) std::cout << " (" << n_same << " same-edge)";
            std::cout << "\n";
            if (n <= 4 || n_same > 0) {
                for (int i = 0; i < n; i++) {
                    auto& p = all_punctures[curves[ci][i]];
                    std::cout << "      [" << i << "] ri=" << p.root_idx
                              << " edge=(" << p.edge_v0 << "," << p.edge_v1 << ")"
                              << " (" << p.x << "," << p.y << ")\n";
                }
            }
        }
    }

    // Diagnose interior endpoints
    if (n_int > 0) {
        for (size_t ci = 0; ci < curves.size(); ci++) {
            if (curve_closed[ci]) continue;
            for (int endpoint : {curves[ci].front(), curves[ci].back()}) {
                if (is_legitimate_endpoint(endpoint)) continue;
                auto& p = all_punctures[endpoint];
                int deg = 0;
                for (auto& [nb, eid] : adj[endpoint]) deg++;
                bool is_d01 = (p.edge_v0 == p.edge_v1);
                // Check if paired with same-edge puncture
                bool same_edge_pair = false;
                for (auto& [nb, eid] : adj[endpoint]) {
                    auto& pn = all_punctures[nb];
                    if (pn.edge_v0 == p.edge_v0 && pn.edge_v1 == p.edge_v1)
                        same_edge_pair = true;
                }
                // Find partner
                int partner = -1;
                for (auto& [nb, eid] : adj[endpoint]) partner = nb;
                auto& pp = all_punctures[partner];
                std::cout << "    int-ep p" << endpoint
                          << " deg=" << deg
                          << (is_d01 ? " D01" : "")
                          << " ri=" << p.root_idx
                          << " edge=(" << p.edge_v0 << "," << p.edge_v1 << ")"
                          << " → partner p" << partner
                          << " ri=" << pp.root_idx
                          << " edge=(" << pp.edge_v0 << "," << pp.edge_v1 << ")"
                          << (same_edge_pair ? " SAME-EDGE" : "")
                          << "\n";
            }
        }
    }

    // ─── Write outputs ──────────────────────────────────────────────────
    write_vti("terrain.vti", elevation, nx, ny, ox, oy, dx, dy);
    write_curves_vtp("terrain_ridges.vtp", curves, curve_closed, all_punctures,
                     elevation, nx, ny, ox, oy, dx, dy);

    return 0;
}
