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

using namespace ftk2;

/**
 * @example unstructured_3d
 * 
 * Demonstrates steady-state 3D feature extraction (contour, fiber, critical point)
 * on the large unstructured mesh 3d.vtu.
 */

int main(int argc, char** argv) {
    const std::string vtu_path = "../tests/data/3d.vtu";

    std::cout << "Loading mesh: " << vtu_path << std::endl;
    auto mesh = read_vtu(vtu_path);
    if (!mesh) {
        std::cerr << "Failed to load mesh!" << std::endl;
        return 1;
    }

    int n_verts = 0;
    mesh->iterate_simplices(0, [&](const Simplex& s) { n_verts++; });
    std::cout << "Mesh loaded with " << n_verts << " vertices." << std::endl;

    std::cout << "Generating synthetic scalar and vector fields..." << std::endl;
    ftk::ndarray<double> s, u, v, w;
    s.reshapef({(size_t)n_verts});
    u.reshapef({(size_t)n_verts});
    v.reshapef({(size_t)n_verts});
    w.reshapef({(size_t)n_verts});

    // Center of features
    const double cx = 30.0, cy = 30.0, cz = 30.0;

    for (int i = 0; i < n_verts; ++i) {
        auto coords = mesh->get_vertex_coordinates(i);
        double x = coords[0], y = coords[1], z = coords[2];
        double dx = x - cx, dy = y - cy, dz = z - cz;

        // 1. Scalar field for contour: Sphere centered at (cx, cy, cz)
        s[i] = std::sqrt(dx*dx + dy*dy + dz*dz) - 15.0; // r=15

        // 2. Vector field for CP and Fiber:
        // CP at (cx, cy, cz)
        u[i] = dx;
        v[i] = dy;
        w[i] = dz;
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"S", s}, {"U", u}, {"V", v}, {"W", w}};

    // --- 1. Extract Contour (m=1) ---
    {
        std::cout << "Extracting Contour (Isosurface)..." << std::endl;
        ContourPredicate<double> pred; pred.var_name = "S";
        SimplicialEngine<double, ContourPredicate<double>> engine(mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        std::cout << "  Found " << complex.vertices.size() << " nodes on contour." << std::endl;
        write_complex_to_vtu(complex, *mesh, "unstructured_3d_contour.vtu", 2);
    }

    // --- 2. Extract Fiber (m=2) ---
    {
        std::cout << "Extracting Fiber (Curve)..." << std::endl;
        FiberPredicate<double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V";
        SimplicialEngine<double, FiberPredicate<double>> engine(mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        std::cout << "  Found " << complex.vertices.size() << " nodes on fiber." << std::endl;
        write_complex_to_vtu(complex, *mesh, "unstructured_3d_fiber.vtu", 1);
    }

    // --- 3. Extract Critical Point (m=3) ---
    {
        std::cout << "Extracting Critical Point..." << std::endl;
        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        std::cout << "  Found " << complex.vertices.size() << " critical point nodes." << std::endl;
        write_complex_to_vtu(complex, *mesh, "unstructured_3d_cp.vtu", 0);
    }

    std::cout << "Done. All results written to VTU files." << std::endl;
    return 0;
}
