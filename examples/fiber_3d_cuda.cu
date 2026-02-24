#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>

#include <ftk2/utils/vtk.hpp>

using namespace ftk2;

int main(int argc, char** argv) {
    const int DW = 32, DH = 32, DD = 32, DT = 10;
    std::cout << "CUDA: Generating 3D+T robust moving spheres intersection data..." << std::endl;

    // 1. Generate two moving sphere fields with different radii
    ftk::ndarray<double> s1({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<double> s2({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    
    for (int t = 0; t < DT; ++t) {
        double c1x = 16.0, c1y = 16.0, c1z = 16.0, r1 = 10.0;
        double c2x = 22.0, c2y = 16.0 + t * 0.5, c2z = 16.0, r2 = 8.0;

        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    double d1 = std::sqrt(std::pow(x-c1x, 2) + std::pow(y-c1y, 2) + std::pow(z-c1z, 2));
                    double d2 = std::sqrt(std::pow(x-c2x, 2) + std::pow(y-c2y, 2) + std::pow(z-c2z, 2));
                    s1.f(x, y, z, t) = d1 - r1;
                    s2.f(x, y, z, t) = d2 - r2;
                }
            }
        }
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"S1", s1}, {"S2", s2}};

    // 2. Create a 4D Spacetime Mesh
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );

    // 3. Set up the Intersection Predicate (m=2)
    IsosurfaceIntersectionPredicate<double> inter_pred;
    inter_pred.var_names[0] = "S1";
    inter_pred.var_names[1] = "S2";
    inter_pred.thresholds[0] = 0.0;
    inter_pred.thresholds[1] = 0.0;

    // 4. Initialize and run the Unified Simplicial Engine
    SimplicialEngine<double, IsosurfaceIntersectionPredicate<double>> engine(mesh, inter_pred);
    
    std::cout << "Tracking sphere intersections ON GPU (CUDA)..." << std::endl;
    engine.execute_cuda(data, {"S1", "S2"});

    // 5. Output results
    auto complex = engine.get_complex();
    std::cout << "Writing results to fiber_3d_cuda.vtu..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "fiber_3d_cuda.vtu", 2);

    std::cout << "Done. Found " << complex.vertices.size() << " feature elements on GPU." << std::endl;

    return 0;
}
