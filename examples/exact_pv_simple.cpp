#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>

using namespace ftk2;

int main() {
    std::cout << "ExactPV Simple Example - Parallel Vector Detection" << std::endl;

    // Create a small 3D mesh: 4x4x4
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{4, 4, 4});

    // Create combined vector field: [ux, uy, uz, vx, vy, vz]
    ftk::ndarray<double> uv({6, 4, 4, 4});  // [6 components][x][y][z]

    // Generate synthetic fields with isolated parallel vector line
    // U = (1, 0, 0) everywhere
    // V varies to create PV line at y-0.5*x=1, z-0.3*x=1
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                // U components (0-2)
                uv.f(0, x, y, z) = 1.0;   // ux
                uv.f(1, x, y, z) = 0.0;   // uy
                uv.f(2, x, y, z) = 0.0;   // uz

                // V components (3-5) - creates tilted PV line
                uv.f(3, x, y, z) = 1.0;                           // vx
                uv.f(4, x, y, z) = (double)y - 0.5*x - 1.0;      // vy
                uv.f(5, x, y, z) = (double)z - 0.3*x - 1.0;      // vz
            }
        }
    }

    std::cout << "Fields generated: U=(1,0,0), V=(1, y-0.5x-1, z-0.3x-1)" << std::endl;
    std::cout << "Parallel vectors occur along tilted line y-0.5x=1, z-0.3x=1" << std::endl;

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
    std::cout << "\nExtracting parallel vector curves from tetrahedra..." << std::endl;
    pred.extract_curves_from_tets(mesh.get(), data, true);  // verbose=true

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
