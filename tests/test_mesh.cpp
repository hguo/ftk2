#include <iostream>
#include <ftk2/core/mesh.hpp>
#include <vector>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;

#define ASSERT_TRUE(...) \
    total_tests++; \
    if ((__VA_ARGS__)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_EQ(a, b) \
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

int main() {
    std::cout << "Running Mesh tests..." << std::endl;
    test_mesh();
    std::cout << "Summary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
