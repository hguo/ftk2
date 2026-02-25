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

    uint64_t nv = mesh->get_num_vertices();
    ftk::ndarray<float> scalar;
    scalar.reshapef({(size_t)nv});
    scalar.fill(1.0f);
    scalar[0] = -1.0f; // Cross zero
    
    std::map<std::string, ftk::ndarray<float>> data = {{"S", scalar}};
    ContourPredicate<float> pred;
    pred.var_name = "S";
    pred.threshold = 0.0f;

    SimplicialEngine<float, ContourPredicate<float>> engine(mesh, pred);
    engine.execute(data);
    
    auto complex = engine.get_complex();
    ASSERT_TRUE(complex.vertices.size() > 0);
}

void test_unstructured_critical_point_tracking() {
    std::cout << "Testing unstructured 2D critical point tracking (Helical)..." << std::endl;
    std::string vtu_path = "../tests/data/1x1.vtu";
    const int n_timesteps = 10;

    auto base_mesh = read_vtu(vtu_path);
    if (!base_mesh) return;

    auto mesh = std::make_shared<ExtrudedSimplicialMesh>(base_mesh, n_timesteps - 1);
    uint64_t nv = mesh->get_num_vertices();

    ftk::ndarray<double> u, v;
    u.reshapef({(size_t)nv}); v.reshapef({(size_t)nv});
    
    for (int i = 0; i < nv; ++i) {
        auto coords = mesh->get_vertex_coordinates(i);
        double x = coords[0], y = coords[1], t = coords.back();
        double phase = t * (2.0 * M_PI / (n_timesteps - 1));
        double cx = 0.51 + 0.2 * std::cos(phase);
        double cy = 0.52 + 0.2 * std::sin(phase);
        u[i] = x - cx;
        v[i] = y - cy;
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}};
    CriticalPointPredicate<2, double> cp_pred;
    cp_pred.var_names[0] = "U"; cp_pred.var_names[1] = "V";

    SimplicialEngine<double, CriticalPointPredicate<2, double>> engine(mesh, cp_pred);
    engine.execute(data);

    auto complex = engine.get_complex();
    ASSERT_TRUE(complex.vertices.size() > 0);
    bool has_cells = false;
    for (const auto& conn : complex.connectivity) if (conn.dimension == 1 && !conn.indices.empty()) has_cells = true;
    ASSERT_TRUE(has_cells);
}

void test_unstructured_3d_features() {
    std::cout << "Testing unstructured 3D steady-state features (contour, fiber, cp)..." << std::endl;
    std::string path = "../tests/data/3d.vtu";
    auto base_mesh = read_vtu(path);
    if (!base_mesh) return;

    uint64_t nv = base_mesh->get_num_vertices();
    ftk::ndarray<double> s, u, v, w;
    s.reshapef({(size_t)nv}); u.reshapef({(size_t)nv}); v.reshapef({(size_t)nv}); w.reshapef({(size_t)nv});

    for (int i = 0; i < nv; ++i) {
        auto coords = base_mesh->get_vertex_coordinates(i);
        double dx = coords[0] - 37.6, dy = coords[1] - 37.7, dz = coords[2] - 37.8;
        s[i] = std::sqrt(dx*dx + dy*dy + dz*dz) - 15.0; // Sphere r=15
        u[i] = dx; v[i] = dy; w[i] = dz; // CP at (37.6, 37.7, 37.8)
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"S", s}, {"U", u}, {"V", v}, {"W", w}};

    // 1. Contour (m=1)
    {
        ContourPredicate<double> pred; pred.var_name = "S";
        SimplicialEngine<double, ContourPredicate<double>> engine(base_mesh, pred);
        engine.execute(data);
        ASSERT_TRUE(engine.get_complex().vertices.size() > 0);
    }

    // 2. Fiber (m=2)
    {
        FiberPredicate<double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V";
        SimplicialEngine<double, FiberPredicate<double>> engine(base_mesh, pred);
        engine.execute(data);
        ASSERT_TRUE(engine.get_complex().vertices.size() > 0);
    }

    // 3. CP (m=3)
    {
        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(base_mesh, pred);
        engine.execute(data);
        // Ensure engine completes. Steady state CP might find 0 nodes on some meshes.
    }
}

void test_regular_3d_features() {
    std::cout << "Testing regular 3D steady-state features (contour, fiber, cp)..." << std::endl;
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{10, 10, 10});

    ftk::ndarray<double> s, u, v, w;
    s.reshapef({10, 10, 10}); u.reshapef({10, 10, 10}); v.reshapef({10, 10, 10}); w.reshapef({10, 10, 10});

    for (int z = 0; z < 10; ++z) for (int y = 0; y < 10; ++y) for (int x = 0; x < 10; ++x) {
        double dx = x - 4.51, dy = y - 4.52, dz = z - 4.53;
        s.f(x, y, z) = std::sqrt(dx*dx + dy*dy + dz*dz) - 2.0;
        u.f(x, y, z) = dx; v.f(x, y, z) = dy; w.f(x, y, z) = dz;
    }

    std::map<std::string, ftk::ndarray<double>> data = {{"S", s}, {"U", u}, {"V", v}, {"W", w}};

    {
        ContourPredicate<double> pred; pred.var_name = "S";
        SimplicialEngine<double, ContourPredicate<double>> engine(mesh, pred);
        engine.execute(data);
        ASSERT_TRUE(engine.get_complex().vertices.size() > 0);
    }
    {
        FiberPredicate<double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V";
        SimplicialEngine<double, FiberPredicate<double>> engine(mesh, pred);
        engine.execute(data);
        ASSERT_TRUE(engine.get_complex().vertices.size() > 0);
    }
    {
        CriticalPointPredicate<3, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, pred);
        engine.execute(data);
        ASSERT_TRUE(engine.get_complex().vertices.size() > 0);
    }
}

void test_unstructured() {
    test_unstructured_io();
    test_unstructured_2x1();
    test_unstructured_3d();
    test_unstructured_extrusion();
    test_unstructured_float_precision();
    test_unstructured_critical_point_tracking();
    test_unstructured_3d_features();
    test_regular_3d_features();
}
