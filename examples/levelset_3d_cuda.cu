#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>

#include <ftk2/utils/vtk.hpp>

using namespace ftk2;

int main(int argc, char** argv) {
    const int DW = 32, DH = 32, DD = 32, DT = 10;
    std::cout << "CUDA: Generating 3D+T moving sphere levelset data..." << std::endl;

    // 1. Generate scalar field on Host
    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        double cx = 16.0 + t * 0.2, cy = 16.0, cz = 16.0, r = 8.0;
        for (int z = 0; z < DD; ++z)
            for (int y = 0; y < DH; ++y)
                for (int x = 0; x < DW; ++x) {
                    double d = std::sqrt(std::pow(x-cx, 2) + std::pow(y-cy, 2) + std::pow(z-cz, 2));
                    scalar.f(x, y, z, t) = d - r;
                }
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"S", scalar}};

    // 2. Create a 4D Spacetime Mesh
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});

    // 3. Set up the Levelset Predicate
    ContourPredicate<double> levelset_pred;
    levelset_pred.var_name = "S";
    levelset_pred.threshold = 0.0;

    // 4. Initialize Unified Engine
    SimplicialEngine<double, ContourPredicate<double>> engine(mesh, levelset_pred);
    
    std::cout << "Tracking 3D levelset ON GPU (CUDA)..." << std::endl;
    // Note: This method is only available when compiled with NVCC (__CUDACC__)
    engine.execute_cuda(data);

    // 5. Output results
    auto complex = engine.get_complex();
    std::cout << "Writing results to levelset_3d_cuda.vtu..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "levelset_3d_cuda.vtu", 3);

    std::cout << "Done. Found " << complex.vertices.size() << " feature nodes on GPU." << std::endl;

    return 0;
}
