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
    const int DW = 16, DH = 16, DD = 16, DT = 5;
    std::cout << "Generating 3D+T moving sphere levelset data (" << DW << "x" << DH << "x" << DD << "x" << DT << ")..." << std::endl;

    // 1. Generate moving sphere scalar field
    ftk::ndarray<float> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        float cx = 16.0f + t * 0.2f, cy = 16.0f, cz = 16.0f, r = 8.0f;
        for (int z = 0; z < DD; ++z)
            for (int y = 0; y < DH; ++y)
                for (int x = 0; x < DW; ++x) {
                    float d = std::sqrt(std::pow(x-cx, 2) + std::pow(y-cy, 2) + std::pow(z-cz, 2));
                    scalar.f(x, y, z, t) = d - r;
                }
    }

    std::map<std::string, ftk::ndarray<float>> data = {{"S", scalar}};

    // 2. Create a 4D Spacetime Mesh
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});

    // 3. Set up the Levelset Predicate (m=1)
    ContourPredicate<float> levelset_pred;
    levelset_pred.var_name = "S";
    levelset_pred.threshold = 0.0f;

    // 4. Initialize and run the Unified Simplicial Engine
    // m=1 in 4D spacetime results in a 3D Volume track (k = 4-1 = 3)
    SimplicialEngine<float, ContourPredicate<float>> engine(mesh, levelset_pred);
    
    std::cout << "Tracking 3D levelset (results in a 3D volume manifold in spacetime)..." << std::endl;
    engine.execute(data, {"S"});

    // 5. Output results (Filter to dim 3 tetrahedra)
    auto complex = engine.get_complex();
    std::cout << "Writing results to levelset_3d.vtu..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "levelset_3d.vtu", 3);

    std::cout << "Done. Found " << complex.vertices.size() << " feature nodes." << std::endl;

    return 0;
}
