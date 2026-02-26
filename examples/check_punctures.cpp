#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <map>
#include <set>

using namespace ftk2;

int main() {
    // Same config as exact_pv_simple.cpp
    const int N = 16;
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{N, N, N});
    ftk::ndarray<double> uv({6, N, N, N});

    double cx = N / 2.0 + 0.1, cy = N / 2.0 + 0.1, cz = N / 2.0;
    double radius = N / 3.0;

    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double dx = x - cx, dy = y - cy, dz = z - cz;
                double r_xy = std::sqrt(dx*dx + dy*dy) + 0.01;
                double theta = std::atan2(dy, dx);

                double ux = std::cos(theta + 0.1*dz);
                double uy = std::sin(theta + 0.1*dz);
                double uz = 0.3 + 0.1*std::cos(2*theta);

                double dist_to_circle = r_xy - radius;
                double dist_to_plane = dz;

                double vx = ux + dist_to_circle * dx/r_xy + 0.1*dist_to_plane*dx/r_xy;
                double vy = uy + dist_to_circle * dy/r_xy + 0.1*dist_to_plane*dy/r_xy;
                double vz = uz + 0.2*dist_to_plane + 0.1*dist_to_circle;

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

    std::cout << "Extracting puncture points..." << std::endl;
    engine.execute(data, {"uv"});
    auto complex = engine.get_complex();

    std::cout << "Found " << complex.vertices.size() << " puncture points" << std::endl;

    // Check how many are in triangle INTERIOR (all 3 bary coords > 0)
    int interior_count = 0;
    for (const auto& v : complex.vertices) {
        if (v.simplex.dimension == 2) {  // Triangle
            bool all_positive = true;
            for (int i = 0; i < 3; ++i) {
                if (v.barycentric_coords[0][i] <= 1e-10) {
                    all_positive = false;
                    break;
                }
            }
            if (all_positive) interior_count++;
        }
    }

    std::cout << interior_count << " / " << complex.vertices.size()
              << " punctures in triangle INTERIOR (all 3 bary coords > 0)" << std::endl;

    // Now check: for each tet, how many punctures are on its faces?
    std::cout << "\nAnalyzing tet-face connectivity..." << std::endl;

    // Build map: triangle -> puncture indices
    std::map<std::set<uint64_t>, std::vector<int>> face_to_punctures;
    for (size_t i = 0; i < complex.vertices.size(); ++i) {
        const auto& v = complex.vertices[i];
        if (v.simplex.dimension == 2) {
            std::set<uint64_t> tri_verts(v.simplex.vertices, v.simplex.vertices + 3);
            face_to_punctures[tri_verts].push_back(i);
        }
    }

    std::cout << "Triangle faces with punctures: " << face_to_punctures.size() << std::endl;

    // Debug: show first few triangles
    std::cout << "\nFirst 5 triangles with punctures:" << std::endl;
    int tri_debug = 0;
    for (const auto& pair : face_to_punctures) {
        if (tri_debug++ >= 5) break;
        std::cout << "  {";
        for (auto v : pair.first) std::cout << v << " ";
        std::cout << "} -> " << pair.second.size() << " puncture(s)" << std::endl;
    }

    // Sample first 100 tets and check their faces
    int tet_count = 0;
    int tets_with_punctures = 0;

    // Find vertex range of puncture triangles
    uint64_t min_v = UINT64_MAX, max_v = 0;
    for (const auto& pair : face_to_punctures) {
        for (auto v : pair.first) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    std::cout << "Puncture triangle vertex range: [" << min_v << ", " << max_v << "]" << std::endl;

    mesh->iterate_simplices(3, [&](const Simplex& s) {
        if (tets_with_punctures >= 20) return false;  // Stop after finding 20 tets

        // Debug first 3 tets
        if (tet_count < 3) {
            std::cout << "\nTet " << tet_count << ": verts {";
            for (int i = 0; i < 4; ++i) std::cout << s.vertices[i] << " ";
            std::cout << "}" << std::endl;
            std::cout << "  Faces: ";
            std::vector<std::set<uint64_t>> test_faces = {
                {s.vertices[0], s.vertices[1], s.vertices[2]},
                {s.vertices[0], s.vertices[1], s.vertices[3]},
                {s.vertices[0], s.vertices[2], s.vertices[3]},
                {s.vertices[1], s.vertices[2], s.vertices[3]}
            };
            for (const auto& f : test_faces) {
                std::cout << "{";
                for (auto v : f) std::cout << v << " ";
                std::cout << "} ";
            }
            std::cout << std::endl;
        }

        // Get 4 faces of this tet
        std::vector<std::set<uint64_t>> faces;
        faces.push_back({s.vertices[0], s.vertices[1], s.vertices[2]});
        faces.push_back({s.vertices[0], s.vertices[1], s.vertices[3]});
        faces.push_back({s.vertices[0], s.vertices[2], s.vertices[3]});
        faces.push_back({s.vertices[1], s.vertices[2], s.vertices[3]});

        int n_faces_with_punctures = 0;
        int total_punctures = 0;
        for (const auto& face : faces) {
            if (face_to_punctures.count(face)) {
                n_faces_with_punctures++;
                total_punctures += face_to_punctures[face].size();
            }
        }

        if (n_faces_with_punctures > 0) {
            tets_with_punctures++;
            if (tets_with_punctures <= 10) {
                std::cout << "  Tet " << tet_count << ": " << n_faces_with_punctures
                          << " faces with punctures, total " << total_punctures << " punctures" << std::endl;
            }
        }

        tet_count++;
        return true;
    });

    std::cout << "\n" << tets_with_punctures << " / " << tet_count
              << " tets have punctures on their faces" << std::endl;

    return 0;
}
