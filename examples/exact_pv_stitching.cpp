#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>

using namespace ftk2;

struct PunctureConnection {
    int puncture1_idx;
    int puncture2_idx;
    uint64_t tet_id;
};

int main() {
    std::cout << "ExactPV with Stitching - Creating single closed curve" << std::endl;

    // Same field configuration as before
    const int N = 16;
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{N, N, N});
    ftk::ndarray<double> uv({6, N, N, N});

    // ----------------------------------------------------------------
    // FIELD: single closed PV curve, SoS-robust
    //
    // U = (1, 0, 0) everywhere (constant)
    // V = (1, z-z0, (x-cx)^2+(y-cy)^2-R^2)
    //
    // U x V = (0, -Vz, Vy) = 0  iff  Vy=0 AND Vz=0
    //   Vy=0  =>  z = z0            (horizontal plane)
    //   Vz=0  =>  circle of radius R at height z0
    //
    // With SoS perturbation in the solver, we no longer need irrational
    // offsets to keep the locus off mesh edges/vertices.  Use exact values.
    // ----------------------------------------------------------------
    // z0 = N/2 + 0.5: circle sits exactly midway between two grid planes.
    // This is a principled half-integer — not a magic offset — and avoids
    // the plane-tangency degeneracy (circle in the same plane as a mesh layer).
    // SoS then handles any remaining edge/vertex crossings automatically.
    double z0 = N / 2.0 + 0.5;
    double cx = N / 2.0;            // center on integer grid point
    double cy = N / 2.0;
    double R  = N / 3.0;            // exact rational radius

    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                // U = (1, 0, 0) — constant first vector field
                double ux = 1.0, uy = 0.0, uz = 0.0;

                // V designed so PV locus is circle at z=z0
                double vx = 1.0;
                double vy = (double)z - z0;                                    // 0 on plane z=z0
                double vz = (x - cx)*(x - cx) + (y - cy)*(y - cy) - R*R;     // 0 on cylinder

                uv.f(0, x, y, z) = ux; uv.f(1, x, y, z) = uy; uv.f(2, x, y, z) = uz;
                uv.f(3, x, y, z) = vx; uv.f(4, x, y, z) = vy; uv.f(5, x, y, z) = vz;
            }
        }
    }

    std::map<std::string, ftk::ndarray<double>> data;
    data["uv"] = uv;

    ExactPVPredicate<double> pred;
    pred.vector_var_name = "uv";
    SimplicialEngine<double, ExactPVPredicate<double>> engine(mesh, pred);

    std::cout << "Extracting puncture points from triangles..." << std::endl;
    engine.execute(data, {"uv"});
    auto complex = engine.get_complex();

    std::cout << "Found " << complex.vertices.size() << " puncture points" << std::endl;

    // Collect all triangle punctures.
    // SoS perturbation in the solver guarantees barycentric coords are
    // generically strictly positive — no ad-hoc threshold filter needed.
    std::vector<int> non_degenerate_punctures;
    for (size_t i = 0; i < complex.vertices.size(); ++i) {
        const auto& v = complex.vertices[i];
        if (v.simplex.dimension == 2)
            non_degenerate_punctures.push_back(i);
    }
    std::cout << non_degenerate_punctures.size() << " punctures on triangles (SoS-robust)" << std::endl;

    // Diagnostic: per-triangle puncture count histogram
    {
        std::map<std::set<uint64_t>, int> tri_count;
        for (int i : non_degenerate_punctures) {
            const auto& v = complex.vertices[i];
            std::set<uint64_t> verts(v.simplex.vertices, v.simplex.vertices + 3);
            tri_count[verts]++;
        }
        std::map<int, int> count_hist;
        for (const auto& [k, v] : tri_count) count_hist[v]++;
        std::cout << "Per-triangle puncture count histogram:" << std::endl;
        for (const auto& [n, cnt] : count_hist)
            std::cout << "  " << cnt << " triangles with " << n << " puncture(s)" << std::endl;

        // Print first few 3-puncture triangles with their lambda values
        int shown = 0;
        for (int i : non_degenerate_punctures) {
            const auto& v = complex.vertices[i];
            std::set<uint64_t> verts(v.simplex.vertices, v.simplex.vertices + 3);
            if (tri_count[verts] == 3 && shown < 2) {
                // Find all punctures on this triangle
                bool first = true;
                for (int j : non_degenerate_punctures) {
                    const auto& w = complex.vertices[j];
                    std::set<uint64_t> verts2(w.simplex.vertices, w.simplex.vertices + 3);
                    if (verts2 == verts) {
                        if (first) {
                            std::cout << "  3-punc tri {";
                            for (auto v2 : verts) std::cout << v2 << " ";
                            std::cout << "}: ";
                            first = false;
                            ++shown;
                        }
                        std::cout << "lambda=" << w.scalar << " bary=("
                                  << w.barycentric_coords[0][0] << ","
                                  << w.barycentric_coords[0][1] << ","
                                  << w.barycentric_coords[0][2] << ") ";
                    }
                }
                if (!first) std::cout << std::endl;
            }
        }
    }

    // Build map: triangle face -> puncture indices (only non-degenerate)
    std::set<int> non_degen_set(non_degenerate_punctures.begin(), non_degenerate_punctures.end());
    std::map<std::set<uint64_t>, std::vector<int>> face_to_punctures;
    for (int i : non_degenerate_punctures) {
        const auto& v = complex.vertices[i];
        if (v.simplex.dimension == 2) {
            std::set<uint64_t> tri_verts(v.simplex.vertices, v.simplex.vertices + 3);
            face_to_punctures[tri_verts].push_back(i);
        }
    }

    std::cout << "\nStitching punctures through tetrahedra..." << std::endl;

    std::vector<PunctureConnection> connections;
    int tets_checked = 0;
    int tets_with_punctures = 0;
    int tets_with_2_punctures = 0;
    int tets_with_more = 0;
    std::map<int, int> puncture_count_histogram;

    mesh->iterate_simplices(3, [&](const Simplex& s) {
        tets_checked++;

        // Get 4 faces of this tet
        std::vector<std::set<uint64_t>> faces = {
            {s.vertices[0], s.vertices[1], s.vertices[2]},
            {s.vertices[0], s.vertices[1], s.vertices[3]},
            {s.vertices[0], s.vertices[2], s.vertices[3]},
            {s.vertices[1], s.vertices[2], s.vertices[3]}
        };

        // Collect all punctures on this tet's faces
        std::vector<int> tet_punctures;
        for (const auto& face : faces) {
            if (face_to_punctures.count(face)) {
                for (int p_idx : face_to_punctures[face]) {
                    tet_punctures.push_back(p_idx);
                }
            }
        }

        if (tet_punctures.empty()) {
            return true;  // Continue iteration
        }

        tets_with_punctures++;
        puncture_count_histogram[tet_punctures.size()]++;

        // Debug: check for odd counts
        if (tet_punctures.size() % 2 == 1) {
            if (puncture_count_histogram[tet_punctures.size()] <= 5) {
                std::cout << "  ODD COUNT! Tet with " << tet_punctures.size() << " punctures:" << std::endl;
                std::cout << "    Tet vertices: ";
                for (int i = 0; i < 4; ++i) std::cout << s.vertices[i] << " ";
                std::cout << std::endl;

                // Show which faces have punctures
                for (size_t f = 0; f < faces.size(); ++f) {
                    if (face_to_punctures.count(faces[f])) {
                        std::cout << "    Face " << f << ": {";
                        for (auto v : faces[f]) std::cout << v << " ";
                        std::cout << "} has " << face_to_punctures[faces[f]].size() << " puncture(s)" << std::endl;
                    }
                }
            }
        }

        // Stitch according to number of punctures
        if (tet_punctures.size() == 2) {
            // Simple case: connect the two punctures
            tets_with_2_punctures++;
            PunctureConnection conn;
            conn.puncture1_idx = tet_punctures[0];
            conn.puncture2_idx = tet_punctures[1];
            conn.tet_id = s.vertices[0];  // Use first vertex as tet ID
            connections.push_back(conn);
        } else if (tet_punctures.size() > 2) {
            // Ambiguous case: use ExactPV characteristic polynomials + all valid
            // λ-intervals to determine which punctures pair with which.
            tets_with_more++;

            // Get field values at tet vertices, separate U and V
            double V_arr[4][3], W_arr[4][3];
            for (int i = 0; i < 4; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                V_arr[i][0] = uv.f(0, coords[0], coords[1], coords[2]);
                V_arr[i][1] = uv.f(1, coords[0], coords[1], coords[2]);
                V_arr[i][2] = uv.f(2, coords[0], coords[1], coords[2]);
                W_arr[i][0] = uv.f(3, coords[0], coords[1], coords[2]);
                W_arr[i][1] = uv.f(4, coords[0], coords[1], coords[2]);
                W_arr[i][2] = uv.f(5, coords[0], coords[1], coords[2]);
            }

            // Compute characteristic polynomials for this tet
            double Q_poly[4], P_poly[4][4];
            characteristic_polynomials_pv_tetrahedron(V_arr, W_arr, Q_poly, P_poly);

            bool q_zero = true;
            for (int k = 0; k <= 3; ++k) if (Q_poly[k] != 0.0) { q_zero = false; break; }

            if (!q_zero) {
                // Get ALL valid λ-intervals where the PV curve is inside the tet
                auto intervals = solve_barycentric_inequalities(P_poly, Q_poly, 1e-10);

                // Diagnostic: print first few ambiguous tets
                static int diag_count = 0;
                if (diag_count < 5) {
                    ++diag_count;
                    std::cout << "[DIAG] Tet with " << tet_punctures.size()
                              << " punctures, intervals=" << intervals.size() << ":";
                    for (const auto& iv : intervals)
                        std::cout << " [" << iv.min << "," << iv.max << "]";
                    std::cout << std::endl;
                }

                // Build curve segment with polynomials and tet vertex positions
                PVCurveSegment seg;
                for (int k = 0; k < 4; ++k) seg.Q.coeffs[k] = Q_poly[k];
                for (int i = 0; i < 4; ++i)
                    for (int k = 0; k < 4; ++k) seg.P[i].coeffs[k] = P_poly[i][k];
                for (int i = 0; i < 4; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    for (int j = 0; j < 3; ++j) seg.tet_vertices[i][j] = coords[j];
                }

                // Helper: physical position of a puncture from its barycentric coords
                auto get_phys = [&](int p_idx) -> std::array<double, 3> {
                    const auto& punc = complex.vertices[p_idx];
                    std::array<double, 3> pos = {0.0, 0.0, 0.0};
                    for (int i = 0; i < 3; ++i) {
                        auto vc = mesh->get_vertex_coordinates(punc.simplex.vertices[i]);
                        for (int j = 0; j < 3; ++j)
                            pos[j] += punc.barycentric_coords[0][i] * vc[j];
                    }
                    return pos;
                };

                std::vector<bool> paired(tet_punctures.size(), false);

                // For each interval the PV curve is inside the tet, its two
                // endpoints correspond to the entry/exit faces.  Pair the
                // unpaired puncture closest (in physical space) to each endpoint.
                for (const auto& interval : intervals) {
                    double lam_lo = std::max(interval.min, -100.0);
                    double lam_hi = std::min(interval.max,  100.0);

                    auto pos_lo = seg.get_physical_coords(lam_lo);
                    auto pos_hi = seg.get_physical_coords(lam_hi);

                    // Closest unpaired puncture to entry endpoint
                    int best_lo = -1; double d_lo = 1e10;
                    for (size_t i = 0; i < tet_punctures.size(); ++i) {
                        if (paired[i]) continue;
                        auto pos = get_phys(tet_punctures[i]);
                        double d = 0;
                        for (int j = 0; j < 3; ++j) d += (pos[j]-pos_lo[j])*(pos[j]-pos_lo[j]);
                        if (d < d_lo) { d_lo = d; best_lo = (int)i; }
                    }
                    if (best_lo < 0) continue;
                    paired[best_lo] = true;  // claim entry puncture

                    // Closest unpaired puncture to exit endpoint
                    int best_hi = -1; double d_hi = 1e10;
                    for (size_t i = 0; i < tet_punctures.size(); ++i) {
                        if (paired[i]) continue;
                        auto pos = get_phys(tet_punctures[i]);
                        double d = 0;
                        for (int j = 0; j < 3; ++j) d += (pos[j]-pos_hi[j])*(pos[j]-pos_hi[j]);
                        if (d < d_hi) { d_hi = d; best_hi = (int)i; }
                    }

                    if (best_hi < 0) { paired[best_lo] = false; continue; }  // release
                    paired[best_hi] = true;
                    connections.push_back({tet_punctures[best_lo], tet_punctures[best_hi], s.vertices[0]});
                }

                // Fallback: pair any remaining unpaired punctures
                for (size_t i = 0; i < tet_punctures.size(); ++i) {
                    if (paired[i]) continue;
                    for (size_t j = i + 1; j < tet_punctures.size(); ++j) {
                        if (paired[j]) continue;
                        connections.push_back({tet_punctures[i], tet_punctures[j], s.vertices[0]});
                        paired[i] = paired[j] = true;
                        break;
                    }
                }
            } else {
                // Degenerate (Q=0): pair sequentially as fallback
                for (size_t i = 0; i + 1 < tet_punctures.size(); i += 2)
                    connections.push_back({tet_punctures[i], tet_punctures[i + 1], s.vertices[0]});
            }
        }

        return true;  // Continue iteration
    });

    std::cout << "Checked " << tets_checked << " tetrahedra" << std::endl;
    std::cout << "  " << tets_with_punctures << " tets have punctures on faces" << std::endl;
    std::cout << "  " << tets_with_2_punctures << " tets have exactly 2 punctures (unambiguous)" << std::endl;
    std::cout << "  " << tets_with_more << " tets have >2 punctures (ambiguous)" << std::endl;

    std::cout << "\nPuncture count histogram:" << std::endl;
    for (const auto& pair : puncture_count_histogram) {
        std::cout << "  " << pair.second << " tets with " << pair.first << " punctures";
        if (pair.first % 2 == 1) std::cout << " <-- ODD (SHOULD NOT HAPPEN!)";
        std::cout << std::endl;
    }

    std::cout << "\nCreated " << connections.size() << " connections" << std::endl;

    // Build adjacency graph from connections
    std::map<int, std::vector<int>> adjacency;
    for (const auto& conn : connections) {
        adjacency[conn.puncture1_idx].push_back(conn.puncture2_idx);
        adjacency[conn.puncture2_idx].push_back(conn.puncture1_idx);
    }

    // Check for unpaired punctures
    int unpaired_count = 0;
    for (int i : non_degenerate_punctures) {
        if (adjacency[i].size() == 0) {
            unpaired_count++;
        } else if (adjacency[i].size() != 2) {
            // Degree != 2 means not a simple path/loop
            if (unpaired_count < 10) {
                std::cout << "  Puncture " << i << " has degree " << adjacency[i].size() << std::endl;
            }
        }
    }
    std::cout << "Unpaired punctures (degree 0): " << unpaired_count << std::endl;

    // Trace connected curves as ordered paths.
    // Each node in a well-formed PV curve has degree exactly 2, so we can
    // walk the adjacency list step-by-step (never revisiting the previous
    // node) until we return to the start (closed) or hit a dead end (open).
    std::set<int> visited;
    struct Curve { std::vector<int> pts; bool closed; };
    std::vector<Curve> curves;

    for (size_t start_idx = 0; start_idx < complex.vertices.size(); ++start_idx) {
        if (visited.count(start_idx) || adjacency[start_idx].empty()) continue;

        Curve curve;
        int curr = (int)start_idx;
        int prev = -1;

        // Walk until we revisit a node (closed) or run out of unvisited neighbours (open).
        while (true) {
            if (visited.count(curr)) {
                // Closed back to a previously visited node.
                curve.closed = (curr == (int)start_idx);
                break;
            }
            visited.insert(curr);
            curve.pts.push_back(curr);

            // Pick the neighbour that is not where we came from.
            int next = -1;
            for (int nb : adjacency[curr]) {
                if (nb != prev) { next = nb; break; }
            }
            if (next == -1) { curve.closed = false; break; }  // open end
            prev = curr;
            curr = next;
        }

        if (curve.pts.size() > 1)
            curves.push_back(std::move(curve));
    }

    std::cout << "\nExtracted " << curves.size() << " connected curve(s)" << std::endl;
    for (size_t i = 0; i < std::min(curves.size(), (size_t)5); ++i) {
        std::cout << "  Curve " << i << ": " << curves[i].pts.size() << " punctures"
                  << (curves[i].closed ? " (closed)" : " (open)") << std::endl;
    }

    // Write each curve as a single vtkPolyLine cell.
    // For a closed curve, repeat the first point ID at the end so VTK
    // draws the closing segment back to the start.
    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    auto points   = vtkSmartPointer<vtkPoints>::New();
    auto lines    = vtkSmartPointer<vtkCellArray>::New();

    std::map<int, vtkIdType> puncture_to_point;
    vtkIdType point_id = 0;

    // Insert all unique points.
    for (const auto& curve : curves) {
        for (int p_idx : curve.pts) {
            if (puncture_to_point.count(p_idx)) continue;
            const auto& v = complex.vertices[p_idx];
            std::vector<double> phys_pos(3, 0.0);
            for (int i = 0; i < 3; ++i) {
                auto vert_coords = mesh->get_vertex_coordinates(v.simplex.vertices[i]);
                for (int j = 0; j < 3; ++j)
                    phys_pos[j] += v.barycentric_coords[0][i] * vert_coords[j];
            }
            points->InsertNextPoint(phys_pos[0], phys_pos[1], phys_pos[2]);
            puncture_to_point[p_idx] = point_id++;
        }
    }

    // Insert one polyline cell per curve.
    int total_segments = 0;
    for (const auto& curve : curves) {
        std::vector<vtkIdType> cell_pts;
        cell_pts.reserve(curve.pts.size() + 1);
        for (int p_idx : curve.pts)
            cell_pts.push_back(puncture_to_point[p_idx]);
        if (curve.closed)
            cell_pts.push_back(cell_pts[0]);  // close the loop
        lines->InsertNextCell((vtkIdType)cell_pts.size(), cell_pts.data());
        total_segments += (int)cell_pts.size() - 1;
    }

    polydata->SetPoints(points);
    polydata->SetLines(lines);

    auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName("exactpv_stitched.vtp");
    writer->SetInputData(polydata);
    writer->SetDataModeToAscii();
    writer->Write();

    std::cout << "\nWrote stitched curves to: exactpv_stitched.vtp" << std::endl;
    std::cout << "  " << points->GetNumberOfPoints() << " points" << std::endl;
    std::cout << "  " << lines->GetNumberOfCells() << " polyline cell(s), "
              << total_segments << " segments" << std::endl;

    return 0;
}
