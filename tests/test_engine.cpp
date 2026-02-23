#include <iostream>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>

using namespace ftk2;

// External simple test framework macros
#define ASSERT_TRUE(...) \
    do { \
        extern int total_tests; \
        extern int passed_tests; \
        total_tests++; \
        if ((__VA_ARGS__)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        extern int total_tests; \
        extern int passed_tests; \
        total_tests++; \
        if ((a) == (b)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

void test_engine() {
    std::cout << "Testing Simplicial Engine..." << std::endl;

    // 1. Simple 2D tracking (Critical point moving in 1D+T)
    {
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{10, 2}); 
        ftk::ndarray<double> u({10, 2});
        for (int x = 0; x < 10; ++x) u.f(x, 0) = (double)x - 3.5;
        for (int x = 0; x < 10; ++x) u.f(x, 1) = (double)x - 4.5;
        std::map<std::string, ftk::ndarray<double>> data = {{"U", u}};
        CriticalPointPredicate<1, double> pred;
        pred.var_names[0] = "U";
        SimplicialEngine<double, CriticalPointPredicate<1, double>> engine(mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        ASSERT_TRUE(complex.vertices.size() > 0);
        ASSERT_EQ(complex.connectivity.size(), 2); 
        ASSERT_EQ(complex.connectivity[0].dimension, 1);
    }

    // 2. 2D Surface Track (Contour in 2D+T)
    {
        // 3x3x2 grid
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{3, 3, 2});
        ftk::ndarray<double> s({3, 3, 2});
        s.fill(1.0);
        s.f(1, 1, 0) = -1.0; // Zero around center at t=0
        s.f(1, 1, 1) = -1.0; // Zero around center at t=1
        
        std::map<std::string, ftk::ndarray<double>> data = {{"S", s}};
        ContourPredicate<double> pred;
        pred.var_name = "S";
        pred.threshold = 0.0;
        
        SimplicialEngine<double, ContourPredicate<double>> engine(mesh, pred);
        engine.execute(data);
        auto complex = engine.get_complex();
        
        ASSERT_TRUE(complex.vertices.size() > 0);
        ASSERT_EQ(complex.connectivity[1].dimension, 2);
        ASSERT_TRUE(complex.connectivity[1].indices.size() > 0); // Should find triangles
    }
}
