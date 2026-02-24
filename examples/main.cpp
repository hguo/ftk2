#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>

using namespace ftk2;

int main(int argc, char** argv) {
    // 1. Create a 3D Spacetime Mesh (2D spatial + 1D time)
    // Grid size: 10x10 spatial, 5 time steps
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{10, 10, 5});

    // 2. Create synthetic vector field data (U, V)
    // We want a critical point that moves from (3, 3) to (7, 7) over 5 time steps.
    ftk::ndarray<double> u({10, 10, 5, 2}), v({10, 10, 5, 2});
    std::cout << "Data rank=" << u.nd() << std::endl;
    for(int i=0; i<u.nd(); ++i) std::cout << "dim[" << i << "]=" << u.dimf(i) << ", prod[" << i << "]=" << u.get_lattice().prod_[i] << std::endl;

    std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}};

    // 3. Set up the Critical Point Predicate (m=2 for 2D spatial vector field)
    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";

    // 4. Initialize and run the Unified Simplicial Engine
    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    
    std::cout << "Running Simplicial Engine..." << std::endl;
    engine.execute(data, {"U", "V"});

    // 5. Output results
    auto complex = engine.get_complex();
    std::cout << "Processing complete. Found " << complex.vertices.size() << " feature elements." << std::endl;

    return 0;
}
