#include <iostream>
#include <ftk2/core/mesh.hpp>
#include <vector>

using namespace ftk2;

// External simple test framework macros from test_main.cpp
#define ASSERT_TRUE(...) \
    extern int total_tests; \
    extern int passed_tests; \
    total_tests++; \
    if ((__VA_ARGS__)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_EQ(a, b) \
    extern int total_tests; \
    extern int passed_tests; \
    total_tests++; \
    if ((a) == (b)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

void test_mesh() {
    std::cout << "Testing Mesh..." << std::endl;

    // D1
    {
        RegularSimplicialMesh mesh({10});
        int count = 0;
        mesh.iterate_simplices(1, [&](const Simplex& s) {
            ASSERT_EQ(s.dimension, 1);
            ASSERT_EQ(s.vertices[1] - s.vertices[0], 1);
            count++;
        });
        ASSERT_EQ(count, 9);
    }

    // D2
    {
        RegularSimplicialMesh mesh({3, 3});
        int count = 0;
        mesh.iterate_simplices(2, [&](const Simplex& s) {
            ASSERT_EQ(s.dimension, 2);
            count++;
        });
        ASSERT_EQ(count, 8);
    }

    // D3
    {
        RegularSimplicialMesh mesh({2, 2, 2});
        int count = 0;
        mesh.iterate_simplices(3, [&](const Simplex& s) {
            ASSERT_EQ(s.dimension, 3);
            count++;
        });
        ASSERT_EQ(count, 6);
    }
}
