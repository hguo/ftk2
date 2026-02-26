#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/utils/vtk.hpp>
#include <iostream>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ftk2;

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <vtu_file>" << std::endl; return 1; }

    std::cout << "Loading base mesh: " << argv[1] << std::endl;
    auto base_mesh = read_vtu(argv[1]);
    if (!base_mesh) { std::cerr << "Failed to load " << argv[1] << std::endl; return 1; }

    uint64_t nv = base_mesh->get_num_vertices();
    std::cout << "Mesh has " << nv << " vertices." << std::endl;

    std::vector<double> h_v0(nv), h_v1(nv), h_v2(nv);
    for (uint64_t i = 0; i < nv; ++i) {
        auto coords = base_mesh->get_vertex_coordinates(i);
        h_v0[i] = coords[0] - 29.0;
        h_v1[i] = coords[1] - 31.0;
        h_v2[i] = coords[2] + 5.0;
    }

    ftk::ndarray<double> arr0, arr1, arr2;
    arr0.copy_vector(h_v0); arr0.reshape({nv});
    arr1.copy_vector(h_v1); arr1.reshape({nv});
    arr2.copy_vector(h_v2); arr2.reshape({nv});

    std::map<std::string, ftk::ndarray<double>> data_map;
    data_map["v0"] = arr0;
    data_map["v1"] = arr1;
    data_map["v2"] = arr2;

    CriticalPointPredicate<3, double> cp_pred;
    cp_pred.var_names = {"v0", "v1", "v2"};

    auto engine = std::make_shared<SimplicialEngine<double, CriticalPointPredicate<3, double>>>(base_mesh, cp_pred);
    engine->execute(data_map);
    
    auto cp_complex = engine->get_complex();
    std::cout << "Found " << cp_complex.vertices.size() << " critical point nodes." << std::endl;
    
    write_complex_to_vtu(cp_complex, *base_mesh, "unstructured_3d_cp.vtu", 0);
    std::cout << "Results written to unstructured_3d_cp.vtu" << std::endl;

    return 0;
}
