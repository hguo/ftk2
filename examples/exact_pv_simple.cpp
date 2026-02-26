#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>

using namespace ftk2;

int main() {
    std::cout << "ExactPV Simple Example - Parallel Vector Detection" << std::endl;

    // Create a 3D mesh: 8x8x8 for better tet interior coverage
    const int N = 8;
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{N, N, N});

    // Create combined vector field: [ux, uy, uz, vx, vy, vz]
    ftk::ndarray<double> uv({6, N, N, N});  // [6 components][x][y][z]

    // Generate fields with simple rotating vectors
    // U rotates with position, V is offset copy
    // This should create a volumetric PV region
    double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double dx = x - cx, dy = y - cy, dz = z - cz;
                double r = std::sqrt(dx*dx + dy*dy + dz*dz) + 0.1;

                // U points radially outward from center
                uv.f(0, x, y, z) = dx / r;
                uv.f(1, x, y, z) = dy / r;
                uv.f(2, x, y, z) = dz / r;

                // V is similar but with a twist - parallel when twist angle matches
                double theta = std::atan2(dy, dx);
                double phi = 0.3;  // Twist angle
                uv.f(3, x, y, z) = (dx * std::cos(phi) - dy * std::sin(phi)) / r;
                uv.f(4, x, y, z) = (dx * std::sin(phi) + dy * std::cos(phi)) / r;
                uv.f(5, x, y, z) = dz / r;
            }
        }
    }

    std::cout << "Fields generated: radial U, twisted V" << std::endl;
    std::cout << "Parallel vectors should occur in a volumetric region" << std::endl;

    // Prepare data map
    std::map<std::string, ftk::ndarray<double>> data;
    data["uv"] = uv;

    // Set up ExactPV predicate
    ExactPVPredicate<double> pred;
    pred.vector_var_name = "uv";

    // Create engine
    SimplicialEngine<double, ExactPVPredicate<double>> engine(mesh, pred);

    // Execute extraction for puncture points (from triangles)
    std::cout << "Extracting parallel vector puncture points from triangles..." << std::endl;
    engine.execute(data, {"uv"});

    // Get puncture point results
    auto complex = engine.get_complex();

    // Extract parametric curves from tetrahedra
    // Note: Each curve segment represents the PV curve within a single tetrahedron.
    // Multiple segments may be part of one connected curve but aren't stitched yet.
    std::cout << "\nExtracting parallel vector curves from tetrahedra..." << std::endl;
    pred.extract_curves_from_tets(mesh.get(), data);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Found " << complex.vertices.size() << " parallel vector puncture points (from triangles)" << std::endl;
    std::cout << "  Found " << pred.curve_segments.size() << " parametric curve segments (from tetrahedra)" << std::endl;

    // Display first few puncture points
    int display_count = std::min(5, (int)complex.vertices.size());
    for (int i = 0; i < display_count; ++i) {
        const auto& v = complex.vertices[i];
        std::cout << "  Point " << i << ": simplex_dim=" << v.simplex.dimension
                  << ", vertices=[" << v.simplex.vertices[0] << ","
                  << v.simplex.vertices[1] << "," << v.simplex.vertices[2] << "]"
                  << ", barycentric=(" << v.barycentric_coords[0][0] << ", "
                  << v.barycentric_coords[0][1] << ", "
                  << v.barycentric_coords[0][2] << ")"
                  << ", lambda=" << v.scalar << std::endl;
    }

    // Display curve segment info
    if (!pred.curve_segments.empty()) {
        std::cout << "\nCurve segments:" << std::endl;
        display_count = std::min(3, (int)pred.curve_segments.size());
        for (int i = 0; i < display_count; ++i) {
            const auto& seg = pred.curve_segments[i];
            std::cout << "  Curve " << i << ": simplex=" << seg.simplex_id
                      << ", lambda_range=[" << seg.lambda_min << ", " << seg.lambda_max << "]"
                      << ", critical_points=" << seg.critical_points.size() << std::endl;
        }
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
