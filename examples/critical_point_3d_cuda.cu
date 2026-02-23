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
    const int DW = 24, DH = 24, DD = 24, DT = 8;
    std::cout << "Generating 3D+T Critical Point synthetic data (GPU)..." << std::endl;

    ftk::ndarray<float> u({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<float> v({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    ftk::ndarray<float> w({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});

    for (int t = 0; t < DT; ++t) {
        float cx = 16.0f + t * 0.1f, cy = 16.0f, cz = 16.0f;
        for (int z = 0; z < DD; ++z)
            for (int y = 0; y < DH; ++y)
                for (int x = 0; x < DW; ++x) {
                    u.f(x, y, z, t) = (float)x - cx;
                    v.f(x, y, z, t) = (float)y - cy;
                    w.f(x, y, z, t) = (float)z - cz;
                }
    }

    std::map<std::string, ftk::ndarray<float>> data = {{"U", u}, {"V", v}, {"W", w}};
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});

    CriticalPointPredicate<3, float> pred;
    pred.var_names[0] = "U";
    pred.var_names[1] = "V";
    pred.var_names[2] = "W";

    SimplicialEngine<float, CriticalPointPredicate<3, float>> engine(mesh, pred);
    engine.execute_cuda(data, {"U", "V", "W"});

    auto complex = engine.get_complex();
    write_complex_to_vtu(complex, *mesh, "cp_3d_cuda.vtu");
    std::cout << "GPU Done. Found " << complex.vertices.size() << " critical points." << std::endl;

    return 0;
}
