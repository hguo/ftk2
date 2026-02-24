#include <iostream>
#include <atomic>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

using namespace ftk2;

// Simple test framework
extern int total_tests;
extern int passed_tests;

#define ASSERT_TRUE(cond) \
    total_tests++; \
    if ((cond)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_EQ(a, b) \
    total_tests++; \
    if ((a) == (b)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

void test_unstructured_io() {
    std::cout << "Testing unstructured mesh IO (1x1.vtu)..." << std::endl;
    std::string path = "../tests/data/1x1.vtu";
    auto mesh = read_vtu(path);
    ASSERT_TRUE(mesh != nullptr);
    if (mesh) {
        ASSERT_EQ(mesh->get_total_dimension(), 2);
        std::atomic<int> n_v(0), n_t(0);
        mesh->iterate_simplices(0, [&](const Simplex& s){ n_v++; });
        mesh->iterate_simplices(2, [&](const Simplex& s){ n_t++; });
        ASSERT_EQ(n_v.load(), 562);
        ASSERT_EQ(n_t.load(), 1042);
    }
}

void test_unstructured_2x1() {
    std::cout << "Testing unstructured mesh IO (2x1.vtu)..." << std::endl;
    std::string path = "../tests/data/2x1.vtu";
    auto mesh = read_vtu(path);
    ASSERT_TRUE(mesh != nullptr);
    if (mesh) {
        std::atomic<int> n_v(0), n_t(0);
        mesh->iterate_simplices(0, [&](const Simplex& s){ n_v++; });
        mesh->iterate_simplices(2, [&](const Simplex& s){ n_t++; });
        ASSERT_EQ(n_v.load(), 1110);
        ASSERT_EQ(n_t.load(), 2098);
    }
}

void test_unstructured_3d() {
    std::cout << "Testing unstructured mesh IO (3d.vtu)..." << std::endl;
    std::string path = "../tests/data/3d.vtu";
    auto mesh = read_vtu(path);
    ASSERT_TRUE(mesh != nullptr);
    if (mesh) {
        std::atomic<int> n_v(0), n_tetra(0);
        mesh->iterate_simplices(0, [&](const Simplex& s){ n_v++; });
        mesh->iterate_simplices(3, [&](const Simplex& s){ n_tetra++; });
        ASSERT_EQ(n_v.load(), 69943);
        ASSERT_EQ(n_tetra.load(), 390464);
    }
}

void test_unstructured_extrusion() {
    std::cout << "Testing unstructured mesh extrusion..." << std::endl;
    std::string path = "../tests/data/1x1.vtu";
    auto base_mesh = read_vtu(path);
    ASSERT_TRUE(base_mesh != nullptr);
    if (!base_mesh) return;

    auto extruded_mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, 1);
    ASSERT_EQ(extruded_mesh->get_total_dimension(), 3);

    std::atomic<int> n_v(0), n_tetra(0);
    extruded_mesh->iterate_simplices(0, [&](const Simplex& s){ n_v++; });
    extruded_mesh->iterate_simplices(3, [&](const Simplex& s){ n_tetra++; });
    
    ASSERT_EQ(n_v.load(), 1124); // 562 * 2
    ASSERT_EQ(n_tetra.load(), 3126); // 1042 * 3
}

void test_unstructured_float_precision() {
    std::cout << "Testing unstructured mesh with float precision..." << std::endl;
    std::string path = "../tests/data/1x1.vtu";
    auto base_mesh = read_vtu(path);
    auto mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, 1);

    // Get correct vertex count
    std::atomic<int> n_v(0);
    mesh->iterate_simplices(0, [&](const Simplex& s){ n_v++; });

    // Generate float data
    ftk::ndarray<float> scalar;
    scalar.reshapef({(size_t)n_v.load()});
    scalar.fill(1.0f);
    scalar[0] = -1.0f; // Cross zero
    
    std::map<std::string, ftk::ndarray<float>> data = {{"S", scalar}};
    
    ContourPredicate<float> pred;
    pred.var_name = "S";
    pred.threshold = 0.0f;
    pred.sos_q = 1e5; // Test configurable quantization

    SimplicialEngine<float, ContourPredicate<float>> engine(mesh, pred);
    engine.execute(data);
    
    auto complex = engine.get_complex();
    ASSERT_TRUE(complex.vertices.size() > 0);
}

void test_unstructured_critical_point_tracking() {
    std::cout << "Testing unstructured 2D critical point tracking..." << std::endl;
    std::string vtu_path = "../tests/data/1x1.vtu";
    const int n_timesteps = 10;

    auto base_mesh = read_vtu(vtu_path);
    ASSERT_TRUE(base_mesh != nullptr);
    if (!base_mesh) return;

    auto mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, n_timesteps - 1);

    std::atomic<int> n_total_verts(0);
    mesh->iterate_simplices(0, [&](const Simplex& s) { n_total_verts++; });

    ftk::ndarray<double> u, v;
    u.reshapef({(size_t)n_total_verts.load()});
    v.reshapef({(size_t)n_total_verts.load()});
    
    for (int i = 0; i < n_total_verts.load(); ++i) {
        auto coords = mesh->get_vertex_coordinates(i);
        double x = coords[0], y = coords[1], t = coords.back();
        double phase = t * (2.0 * M_PI / (n_timesteps - 1));
        double cx = 0.5 + 0.2 * std::cos(phase);
        double cy = 0.5 + 0.2 * std::sin(phase);
        u[i] = x - cx;
        v[i] = y - cy;
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}};
    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U"; cp_pred.var_names[1] = "V";

    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    engine.execute(data);

    auto complex = engine.get_complex();
    
    // We expect some interior nodes and connected cells (tracks)
    ASSERT_TRUE(complex.vertices.size() > 0);
    bool has_cells = false;
    for (const auto& conn : complex.connectivity) if (conn.dimension == 1 && !conn.indices.empty()) has_cells = true;
    ASSERT_TRUE(has_cells);
}

void test_unstructured() {
    test_unstructured_io();
    test_unstructured_2x1();
    test_unstructured_3d();
    test_unstructured_extrusion();
    test_unstructured_float_precision();
    test_unstructured_critical_point_tracking();
}
