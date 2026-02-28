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
            // Ambiguous case: use ExactPV tet solver to determine pairing
            tets_with_more++;

            // Get field values at tet vertices
            double values[4][6];
            for (int i = 0; i < 4; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                for (int j = 0; j < 6; ++j) {
                    values[i][j] = uv.f(j, coords[0], coords[1], coords[2]);
                }
            }

            // Run ExactPV tet solver
            PVCurveSegment segment;
            bool has_curve = pred.extract_tetrahedron(s, values, segment);

            if (has_curve) {
                // Store tet vertex positions for curve evaluation
                for (int i = 0; i < 4; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    for (int j = 0; j < 3; ++j) {
                        segment.tet_vertices[i][j] = coords[j];
                    }
                }

                // For each puncture, find closest point on curve
                std::vector<std::pair<double, int>> puncture_params;  // (lambda, puncture_idx)

                for (int p_idx : tet_punctures) {
                    const auto& puncture = complex.vertices[p_idx];

                    // Get physical position of puncture
                    std::vector<double> p_pos(3, 0.0);
                    for (int i = 0; i < 3; ++i) {
                        auto vert_coords = mesh->get_vertex_coordinates(puncture.simplex.vertices[i]);
                        for (int j = 0; j < 3; ++j) {
                            p_pos[j] += puncture.barycentric_coords[0][i] * vert_coords[j];
                        }
                    }

                    // Sample curve and find closest point
                    double best_lambda = segment.lambda_min;
                    double min_dist = 1e10;
                    int n_samples = 20;
                    double lambda_range = segment.lambda_max - segment.lambda_min;

                    for (int i = 0; i < n_samples; ++i) {
                        double lambda = segment.lambda_min + (double)i / (n_samples - 1) * lambda_range;
                        auto curve_pos = segment.get_physical_coords(lambda);

                        double dist = std::sqrt(
                            (curve_pos[0] - p_pos[0]) * (curve_pos[0] - p_pos[0]) +
                            (curve_pos[1] - p_pos[1]) * (curve_pos[1] - p_pos[1]) +
                            (curve_pos[2] - p_pos[2]) * (curve_pos[2] - p_pos[2])
                        );

                        if (dist < min_dist) {
                            min_dist = dist;
                            best_lambda = lambda;
                        }
                    }

                    puncture_params.push_back({best_lambda, p_idx});
                }

                // Sort by lambda parameter
                std::sort(puncture_params.begin(), puncture_params.end());

                // Connect adjacent punctures in sorted order
                for (size_t i = 0; i + 1 < puncture_params.size(); i += 2) {
                    PunctureConnection conn;
                    conn.puncture1_idx = puncture_params[i].second;
                    conn.puncture2_idx = puncture_params[i + 1].second;
                    conn.tet_id = s.vertices[0];
                    connections.push_back(conn);
                }
            } else {
                // No curve through tet - connect pairs sequentially as fallback
                for (size_t i = 0; i + 1 < tet_punctures.size(); i += 2) {
                    PunctureConnection conn;
                    conn.puncture1_idx = tet_punctures[i];
                    conn.puncture2_idx = tet_punctures[i + 1];
                    conn.tet_id = s.vertices[0];
                    connections.push_back(conn);
                }
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
