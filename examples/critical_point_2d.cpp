#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
#include <ndarray/ndarray_stream.hh>
#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <fstream>

#include <ftk2/utils/vtk.hpp>

using namespace ftk2;

int main(int argc, char** argv) {
    const int DW = 16, DH = 16, DT = 5;
    std::cout << "Generating 2D+T woven synthetic data (" << DW << "x" << DH << "x" << DT << ")..." << std::endl;

    // 1. Create YAML configuration for woven data
    {
        std::ofstream f("woven.yaml");
        f << "stream:\n";
        f << "  substreams:\n";
        f << "    - name: woven\n";
        f << "      format: synthetic\n";
        f << "      dimensions: [" << DW << ", " << DH << "]\n";
        f << "      timesteps: " << DT << "\n";
        f << "      delta: " << 1.0 / (DT - 1) << "\n"; // Match synthetic_woven_2Dt logic
        f << "      vars:\n";
        f << "        - name: scalar\n";
        f << "          dtype: float64\n";
        f.close();
    }

    // 2. Read stream
    ftk::stream<> stream;
    stream.parse_yaml("woven.yaml");

    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        auto group = stream.read(t);
        const auto& s_t = group->get_ref<double>("scalar");
        for (int y = 0; y < DH; ++y) {
            for (int x = 0; x < DW; ++x) {
                scalar.f(x, y, t) = s_t.f(x, y);
            }
        }
    }

    // 3. Compute spatial gradient (U, V) using interior central differences
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

    // 4. Create a Spacetime Mesh on the CORE region with global context
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW-2, (uint64_t)DH-2, (uint64_t)DT}, // local_dims
        std::vector<uint64_t>{1, 1, 0},                                     // offset
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT}     // global_dims
    );

    // 5. Set up the Critical Point Predicate (m=2)
    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.scalar_var_name = "Woven";

    // 6. Initialize and run the Unified Simplicial Engine
    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    
    std::cout << "Tracking critical points strictly in the CORE region..." << std::endl;
    engine.execute(data, {"U", "V", "Woven"});

    // 7. Output results
    auto complex = engine.get_complex();
    std::cout << "Writing results to critical_point_2d.vtu and critical_point_2d.vtp..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "critical_point_2d.vtu");
    write_complex_to_vtp(complex, *mesh, "critical_point_2d.vtp");

    std::cout << "Done. Found " << complex.vertices.size() << " feature elements." << std::endl;

    return 0;
}
