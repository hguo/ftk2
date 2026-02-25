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
 * Demonstrates 3D and 4D feature extraction/tracking on the large unstructured mesh 3d.vtu.
 */

int main(int argc, char** argv) {
    const std::string vtu_path = "../tests/data/3d.vtu";
    const int n_timesteps = 5;

    std::cout << "Loading base mesh: " << vtu_path << std::endl;
    auto base_mesh = read_vtu(vtu_path);
    if (!base_mesh) {
        std::cerr << "Failed to load base mesh!" << std::endl;
        return 1;
    }

    std::cout << "First 5 vertex coords:" << std::endl;
    for (int i=0; i<5; ++i) {
        auto c = base_mesh->get_vertex_coordinates(i);
        std::cout << "  v" << i << ": (" << c[0] << ", " << c[1] << ", " << c[2] << ")" << std::endl;
    }

    const double cx = 29.0, cy = 31.0, cz = -5.0;

    // --- 1. Steady State Extraction (3D) ---
    {
        std::cout << "\n--- Steady-State 3D Extraction ---" << std::endl;
        uint64_t nv = base_mesh->get_num_vertices();
        ftk::ndarray<double> u, v, w;
        u.reshapef({(size_t)nv}); v.reshapef({(size_t)nv}); w.reshapef({(size_t)nv});

        for (uint64_t i = 0; i < nv; ++i) {
            auto coords = base_mesh->get_vertex_coordinates(i);
            u[i] = coords[0] - cx; v[i] = coords[1] - cy; w[i] = coords[2] - cz;
        }
        std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", w}};

        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(base_mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        std::cout << "Found " << complex.vertices.size() << " critical point nodes." << std::endl;
        write_complex_to_vtu(complex, *base_mesh, "unstructured_3d_cp.vtu", 0);
    }

    // --- 2. Time-Varying Tracking (4D) ---
    {
        std::cout << "\n--- CPU Time-Varying 4D Tracking ---" << std::endl;
        auto st_mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, n_timesteps - 1);
        uint64_t nv_st = st_mesh->get_num_vertices();

        ftk::ndarray<double> u, v, w;
        u.reshapef({(size_t)nv_st}); v.reshapef({(size_t)nv_st}); w.reshapef({(size_t)nv_st});

        for (uint64_t i = 0; i < nv_st; ++i) {
            auto coords = st_mesh->get_vertex_coordinates(i);
            double x = coords[0], y = coords[1], z = coords[2], t = coords[3];

            // Critical point fixed at (cx, cy, cz)
            u[i] = x - cx; v[i] = y - cy; w[i] = z - cz;
        }
        std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", w}};

        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(st_mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        std::cout << "Found " << complex.vertices.size() << " nodes forming CPU CP tracks." << std::endl;
        write_complex_to_vtu(complex, *st_mesh, "unstructured_3d_cp_tracks.vtu", 1);
    }

#if FTK_HAVE_CUDA
    // --- 3. GPU Tracking (4D) ---
    {
        std::cout << "\n--- GPU Time-Varying 4D Tracking ---" << std::endl;
        auto st_mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, n_timesteps - 1);
        uint64_t nv_st = st_mesh->get_num_vertices();

        ftk::ndarray<double> u, v, w;
        u.reshapef({(size_t)nv_st}); v.reshapef({(size_t)nv_st}); w.reshapef({(size_t)nv_st});

        for (uint64_t i = 0; i < nv_st; ++i) {
            auto coords = st_mesh->get_vertex_coordinates(i);
            u[i] = coords[0] - cx; v[i] = coords[1] - cy; w[i] = coords[2] - cz;
        }
        std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", w}};

        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(st_mesh, pred);
        
        std::cout << "Running GPU tracking..." << std::endl;
        engine.execute_cuda(data);
        auto complex = engine.get_complex();
        std::cout << "Found " << complex.vertices.size() << " nodes forming GPU CP tracks." << std::endl;
        write_complex_to_vtu(complex, *st_mesh, "unstructured_3d_cp_tracks_cuda.vtu", 1);
    }
#endif

    std::cout << "\nDone. Results written to VTU files." << std::endl;
    return 0;
}
