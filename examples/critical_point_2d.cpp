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
    const int DW = 32, DH = 32, DT = 10;
    std::cout << "Generating 2D+T woven synthetic data (" << DW << "x" << DH << "x" << DT << ")..." << std::endl;

    // 1. Generate woven scalar data
    ftk::ndarray<double> scalar = ftk::synthetic_woven_2Dt<double>(DW, DH, DT);

    // 2. Compute spatial gradient (U, V) using interior central differences
    ftk::ndarray<double> u({(size_t)DW, (size_t)DH, (size_t)DT}), v({(size_t)DW, (size_t)DH, (size_t)DT});
    u.fill(0.0); v.fill(0.0);
    
    for (int t = 0; t < DT; ++t) {
        for (int y = 1; y < DH - 1; ++y) {
            for (int x = 1; x < DW - 1; ++x) {
                u.f(x, y, t) = (scalar.f(x + 1, y, t) - scalar.f(x - 1, y, t)) / 2.0;
                v.f(x, y, t) = (scalar.f(x, y + 1, t) - scalar.f(x, y - 1, t)) / 2.0;
            }
        }
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"Woven", scalar}};

    // 3. Create a Spacetime Mesh on the CORE region with global context
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW-2, (uint64_t)DH-2, (uint64_t)DT}, // local_dims
        std::vector<uint64_t>{1, 1, 0},                                     // offset
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT}     // global_dims
    );

    // 4. Set up the Critical Point Predicate (m=2)
    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.scalar_var_name = "Woven";

    // 5. Initialize and run the Unified Simplicial Engine
    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    
    std::cout << "Tracking critical points strictly in the CORE region..." << std::endl;
    engine.execute(data);

    // 6. Output results
    auto complex = engine.get_complex();
    std::cout << "Writing results to critical_point_2d.vtu and critical_point_2d.vtp..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "critical_point_2d.vtu");
    write_complex_to_vtp(complex, *mesh, "critical_point_2d.vtp");

    std::cout << "Done. Found " << complex.vertices.size() << " feature elements." << std::endl;

    return 0;
}
