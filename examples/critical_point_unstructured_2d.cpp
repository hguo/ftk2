#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <limits>

using namespace ftk2;

/**
 * @example critical_point_unstructured_2d
 * 
 * Demonstrates critical point tracking on an unstructured 2D mesh extruded in time.
 * This example uses 1x1.vtu as the base spatial mesh.
 */

int main(int argc, char** argv) {
    const std::string vtu_path = "../tests/data/1x1.vtu";
    const int n_timesteps = 10;

    std::cout << "Loading base mesh: " << vtu_path << std::endl;
    auto base_mesh = read_vtu(vtu_path);
    if (!base_mesh) {
        std::cerr << "Failed to load base mesh!" << std::endl;
        return 1;
    }

    // Print mesh bounds for verification
    double min_x = std::numeric_limits<double>::max(), max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max(), max_y = std::numeric_limits<double>::lowest();
    base_mesh->iterate_simplices(0, [&](const Simplex& s) {
        auto coords = base_mesh->get_vertex_coordinates(s.vertices[0]);
        min_x = std::min(min_x, coords[0]); max_x = std::max(max_x, coords[0]);
        min_y = std::min(min_y, coords[1]); max_y = std::max(max_y, coords[1]);
    });
    std::cout << "Mesh bounds: X=[" << min_x << ", " << max_x << "], Y=[" << min_y << ", " << max_y << "]" << std::endl;

    std::cout << "Extruding mesh for " << n_timesteps << " timesteps..." << std::endl;
    auto mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, n_timesteps - 1);

    int n_total_verts = 0;
    mesh->iterate_simplices(0, [&](const Simplex& s) { n_total_verts++; });

    std::cout << "Generating synthetic vector field (U, V) on " << n_total_verts << " vertices..." << std::endl;
    ftk::ndarray<double> u, v;
    u.reshapef({(size_t)n_total_verts});
    v.reshapef({(size_t)n_total_verts});
    
    for (int i = 0; i < n_total_verts; ++i) {
        auto coords = mesh->get_vertex_coordinates(i);
        double x = coords[0], y = coords[1];
        double t = coords.back(); // Last component is always time in ExtrudedSimplicialMesh
        
        // Define a critical point moving in a small circle well within the interior
        double phase = t * (2.0 * M_PI / (n_timesteps - 1));
        double cx = 0.5 + 0.2 * std::cos(phase);
        double cy = 0.5 + 0.2 * std::sin(phase);
        
        // Simple linear field: (u, v) = (x - cx, y - cy)
        u[i] = x - cx;
        v[i] = y - cy;
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}};

    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";

    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    
    std::cout << "Tracking critical points..." << std::endl;
    engine.execute(data);

    auto complex = engine.get_complex();
    
    std::string out_vtu = "cp_unstructured_2d.vtu";
    std::cout << "Writing tracks to " << out_vtu << "..." << std::endl;
    write_complex_to_vtu(complex, *mesh, out_vtu, 1);

    std::cout << "Done. Found " << complex.vertices.size() << " feature nodes forming tracks." << std::endl;

    return 0;
}
