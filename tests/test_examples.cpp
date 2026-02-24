#include <iostream>
#include <fstream>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
#include <ndarray/ndarray_stream.hh>
#include <map>
#include <vector>
#include <string>
#include <cmath>

using namespace ftk2;

// Simple test framework macros
extern int total_tests;
extern int passed_tests;

#define ASSERT_TRUE(condition) \
    do { \
        total_tests++; \
        if ((condition)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        total_tests++; \
        if ((a) == (b)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

void test_critical_point_2d() {
    std::cout << "Testing critical_point_2d example..." << std::endl;
    const int DW = 16, DH = 16, DT = 5;
    
    // Create YAML configuration for woven data
    {
        std::ofstream f("woven_test.yaml");
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

    ftk::stream<> stream;
    stream.parse_yaml("woven_test.yaml");

    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DT});
    
    // Read stream into scalar array
    for (int t = 0; t < DT; ++t) {
        auto group = stream.read(t);
        const auto& s_t = group->get_ref<double>("scalar");
        for (int y = 0; y < DH; ++y) {
            for (int x = 0; x < DW; ++x) {
                scalar.f(x, y, t) = s_t.f(x, y);
            }
        }
    }

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

    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW-2, (uint64_t)DH-2, (uint64_t)DT}, 
        std::vector<uint64_t>{1, 1, 0},                                     
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT}     
    );

    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U";
    cp_pred.var_names[1] = "V";
    cp_pred.scalar_var_name = "Woven";

    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    engine.execute(data, {"U", "V", "Woven"});

    auto complex = engine.get_complex();
    ASSERT_EQ(complex.vertices.size(), 788);
}

void test_levelset_2d() {
    std::cout << "Testing levelset_2d example..." << std::endl;
    const int DW = 16, DH = 16, DT = 5;

    // Create YAML configuration for merger data
    {
        std::ofstream f("merger_test.yaml");
        f << "stream:\n";
        f << "  substreams:\n";
        f << "    - name: merger\n";
        f << "      format: synthetic\n";
        f << "      dimensions: [" << DW << ", " << DH << "]\n";
        f << "      timesteps: " << DT << "\n";
        f << "      delta: " << M_PI / (DT - 1) << "\n"; // Match levelset_2d logic
        f << "      vars:\n";
        f << "        - name: scalar\n";
        f << "          dtype: float64\n";
        f.close();
    }

    ftk::stream<> stream;
    stream.parse_yaml("merger_test.yaml");

    ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        auto group = stream.read(t);
        const auto& s_t = group->get_ref<double>("scalar");
        for (int y = 0; y < DH; ++y)
            for (int x = 0; x < DW; ++x)
                scalar.f(x, y, t) = s_t.f(x, y);
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"Scalar", scalar}};
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT});

    ContourPredicate<double> contour_pred;
    contour_pred.var_name = "Scalar";
    contour_pred.threshold = 0.5;

    SimplicialEngine<double, ContourPredicate<double>> engine(mesh, contour_pred);
    engine.execute(data, {"Scalar"});

    auto complex = engine.get_complex();
    ASSERT_EQ(complex.vertices.size(), 1206);
}

void test_critical_point_3d() {
    std::cout << "Testing critical_point_3d example..." << std::endl;
    const int DW = 24, DH = 24, DD = 24, DT = 8;

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

    ftk::ndarray<float> s({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT}); s.fill(0.0f);
    std::map<std::string, ftk::ndarray<float>> data = {{"U", u}, {"V", v}, {"W", w}, {"S", s}};
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});

    CriticalPointPredicate<3, float> pred;
    pred.var_names[0] = "U";
    pred.var_names[1] = "V";
    pred.var_names[2] = "W";

    SimplicialEngine<float, CriticalPointPredicate<3, float>> engine(mesh, pred);
    engine.execute(data, {"U", "V", "W", "S"});

    auto complex = engine.get_complex();
    ASSERT_EQ(complex.vertices.size(), 52);
}

void test_fiber_3d() {
    std::cout << "Testing fiber_3d example..." << std::endl;
    const int DW = 32, DH = 32, DD = 32, DT = 10;
    
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
    auto mesh = std::make_shared<RegularSimplicialMesh>(
        std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT}
    );

    FiberPredicate<double> inter_pred;
    inter_pred.var_names[0] = "S1";
    inter_pred.var_names[1] = "S2";
    inter_pred.thresholds[0] = 0.0;
    inter_pred.thresholds[1] = 0.0;

    SimplicialEngine<double, FiberPredicate<double>> engine(mesh, inter_pred);
    engine.execute(data, {"S1", "S2"});

    auto complex = engine.get_complex();
    ASSERT_EQ(complex.vertices.size(), 6078);
}

void test_levelset_3d() {
    std::cout << "Testing levelset_3d example..." << std::endl;
    const int DW = 32, DH = 32, DD = 32, DT = 10;
    ftk::ndarray<float> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
    for (int t = 0; t < DT; ++t) {
        float cx = 16.0f + t * 0.2f, cy = 16.0f, cz = 16.0f, r = 8.0f;
        for (int z = 0; z < DD; ++z) {
            for (int y = 0; y < DH; ++y) {
                for (int x = 0; x < DW; ++x) {
                    float d = std::sqrt(std::pow(x-cx, 2) + std::pow(y-cy, 2) + std::pow(z-cz, 2));
                    scalar.f(x, y, z, t) = d - r;
                }
            }
        }
    }

    std::map<std::string, ftk::ndarray<float>> data = {{"S", scalar}};
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});

    ContourPredicate<float> contour_pred;
    contour_pred.var_name = "S";
    contour_pred.threshold = 0.0f;

    SimplicialEngine<float, ContourPredicate<float>> engine(mesh, contour_pred);
    engine.execute(data, {"S"});

    auto complex = engine.get_complex();
    ASSERT_EQ(complex.vertices.size(), 66418);
}

void test_streaming() {
    std::cout << "Testing streaming execution..." << std::endl;
    const int DW = 16, DH = 16, DT = 5;
    
    // 1. One-shot execution
    ftk::ndarray<double> scalar = ftk::synthetic_woven_2Dt<double>(DW, DH, DT);
    std::map<std::string, ftk::ndarray<double>> data = {{"scalar", scalar}};
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT});
    ContourPredicate<double> pred;
    pred.var_name = "scalar";
    pred.threshold = 0.5;
    
    SimplicialEngine<double, ContourPredicate<double>> engine(mesh, pred);
    engine.execute(data);
    auto n1 = engine.get_complex().vertices.size();

    // 2. Streamed execution
    {
        std::ofstream f("streaming_test.yaml");
        f << "stream:\n";
        f << "  substreams:\n";
        f << "    - name: woven\n";
        f << "      format: synthetic\n";
        f << "      dimensions: [" << DW << ", " << DH << "]\n";
        f << "      timesteps: " << DT << "\n";
        f << "      delta: " << 1.0 / (DT - 1) << "\n";
        f << "      vars:\n";
        f << "        - name: scalar\n";
        f << "          dtype: float64\n";
        f.close();
    }
    ftk::stream<> stream;
    stream.parse_yaml("streaming_test.yaml");
    
    engine.execute_stream(stream);
    auto n2 = engine.get_complex().vertices.size();

    ASSERT_EQ(n1, n2);
}

void test_examples() {
    test_critical_point_2d();
    test_levelset_2d();
    test_critical_point_3d();
    test_levelset_3d();
    test_fiber_3d();
    test_streaming();
}
