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
    const int DW = 16, DH = 16, DT = 5;
    std::cout << "CUDA: Generating 2D+T merger synthetic data (" << DW << "x" << DH << "x" << DT << ")..." << std::endl;

    // 1. Generate merger scalar data
    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        double time = (double)t / (DT - 1) * M_PI;
        auto s = ftk::synthetic_merger_2D<double>(DW, DH, time);
        for (int y = 0; y < DH; ++y)
            for (int x = 0; x < DW; ++x)
                scalar.f(x, y, t) = s.f(x, y);
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"Scalar", scalar}};

    // 2. Create a 3D Spacetime Mesh (2D spatial + 1D time)
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT});

    // 3. Set up the Contour Predicate (m=1)
    ContourPredicate<double> contour_pred;
    contour_pred.var_name = "Scalar";
    contour_pred.threshold = 0.5;

    // 4. Initialize and run the Unified Simplicial Engine
    SimplicialEngine<double, ContourPredicate<double>> engine(mesh, contour_pred);
    
    std::cout << "Tracking contours ON GPU (CUDA)..." << std::endl;
    engine.execute_cuda(data, {"Scalar"});

    // 5. Output results
    auto complex = engine.get_complex();
    std::cout << "Writing results to levelset_2d_cuda.vtu..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "levelset_2d_cuda.vtu", 2);

    std::cout << "Done. Found " << complex.vertices.size() << " feature elements on GPU." << std::endl;

    return 0;
}
