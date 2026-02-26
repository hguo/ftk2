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
    // REDESIGNED FIELD: guaranteed single closed PV curve
    //
    // U = (1, 0, 0) everywhere (constant)
    // V = (1, z-z0, (x-cx)^2+(y-cy)^2-R^2)
    //
    // U x V = (0, -Vz, Vy) = 0  iff  Vy=0 AND Vz=0
    //   Vy=0  =>  z = z0            (a horizontal plane)
    //   Vz=0  =>  (x-cx)^2+(y-cy)^2 = R^2  (a vertical cylinder)
    //   Intersection: exactly ONE closed circle at height z0
    //
    // Non-integer offsets ensure the circle avoids all mesh vertices/edges.
    // ----------------------------------------------------------------
    double z0 = N / 2.0 + 0.37;     // circle height (between grid planes)
    double cx = N / 2.0 + 0.13;     // circle center x
    double cy = N / 2.0 + 0.27;     // circle center y
    double R  = N / 3.0 + 0.41;     // circle radius (~5.74 for N=16)

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

    // Count interior punctures and filter out degenerate ones
    std::vector<int> non_degenerate_punctures;
    int interior_count = 0;
    int degenerate_count = 0;

    for (size_t i = 0; i < complex.vertices.size(); ++i) {
        const auto& v = complex.vertices[i];
        if (v.simplex.dimension == 2) {
            bool all_positive = true;
            for (int j = 0; j < 3; ++j) {
                if (v.barycentric_coords[0][j] <= 1e-10) {
                    all_positive = false;
                    break;
                }
            }
            if (all_positive) {
                interior_count++;
                non_degenerate_punctures.push_back(i);
            } else {
                degenerate_count++;
            }
        }
    }
    std::cout << interior_count << " punctures in triangle INTERIOR" << std::endl;
    std::cout << degenerate_count << " punctures on triangle EDGES/VERTICES (degenerate - skipping)" << std::endl;

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

    // Trace connected curves using DFS
    std::set<int> visited;
    std::vector<std::vector<int>> curves;

    for (size_t start_idx = 0; start_idx < complex.vertices.size(); ++start_idx) {
        if (visited.count(start_idx) || adjacency[start_idx].empty()) continue;

        // Trace curve from this starting point
        std::vector<int> curve;
        std::vector<int> stack = {(int)start_idx};

        while (!stack.empty()) {
            int curr = stack.back();
            stack.pop_back();

            if (visited.count(curr)) continue;
            visited.insert(curr);
            curve.push_back(curr);

            for (int neighbor : adjacency[curr]) {
                if (!visited.count(neighbor)) {
                    stack.push_back(neighbor);
                }
            }
        }

        if (curve.size() > 1) {
            curves.push_back(curve);
        }
    }

    std::cout << "\nExtracted " << curves.size() << " connected curve(s)" << std::endl;
    for (size_t i = 0; i < std::min(curves.size(), (size_t)5); ++i) {
        std::cout << "  Curve " << i << ": " << curves[i].size() << " punctures" << std::endl;
    }

    // Write stitched curves to VTP
    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    auto points = vtkSmartPointer<vtkPoints>::New();
    auto lines = vtkSmartPointer<vtkCellArray>::New();

    std::map<int, vtkIdType> puncture_to_point;
    vtkIdType point_id = 0;

    // Add all points
    for (const auto& curve : curves) {
        for (int p_idx : curve) {
            if (puncture_to_point.count(p_idx)) continue;

            const auto& v = complex.vertices[p_idx];
            // Compute physical position from barycentric coords
            std::vector<double> phys_pos(3, 0.0);
            for (int i = 0; i < 3; ++i) {
                auto vert_coords = mesh->get_vertex_coordinates(v.simplex.vertices[i]);
                for (int j = 0; j < 3; ++j) {
                    phys_pos[j] += v.barycentric_coords[0][i] * vert_coords[j];
                }
            }

            points->InsertNextPoint(phys_pos[0], phys_pos[1], phys_pos[2]);
            puncture_to_point[p_idx] = point_id++;
        }
    }

    // Add lines (connections)
    for (const auto& conn : connections) {
        vtkIdType pts[2] = {
            puncture_to_point[conn.puncture1_idx],
            puncture_to_point[conn.puncture2_idx]
        };
        lines->InsertNextCell(2, pts);
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
    std::cout << "  " << lines->GetNumberOfCells() << " line segments" << std::endl;

    return 0;
}
