#include <iostream>
#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/zero_crossing.hpp>

using namespace ftk2;

extern int total_tests;
extern int passed_tests;

#define ASSERT_TRUE(cond) \
    do { \
        total_tests++; \
        if ((cond)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: " << #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
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

void test_3d_cp_system() {
    std::cout << "Testing single-tetrahedron 3D CP extraction..." << std::endl;
    // Tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    // Field: u=x-0.25, v=y-0.25, w=z-0.25. Zero at (0.25, 0.25, 0.25)
    double values[4][3] = {
        {-0.25, -0.25, -0.25}, // v0: (0,0,0)
        { 0.75, -0.25, -0.25}, // v1: (1,0,0)
        {-0.25,  0.75, -0.25}, // v2: (0,1,0)
        {-0.25, -0.25,  0.75}  // v3: (0,0,1)
    };
    uint64_t indices[4] = {0, 1, 2, 3};
    
    bool inside = sos::origin_inside<3, double>::check(values, indices);
    ASSERT_TRUE(inside);
    
    if (inside) {
        double lambda[4];
        bool solved = ZeroCrossingSolver<3, double>::solve(values, lambda);
        ASSERT_TRUE(solved);
        if (solved) {
            std::cout << "  Lambda: " << lambda[0] << ", " << lambda[1] << ", " << lambda[2] << ", " << lambda[3] << std::endl;
            ASSERT_EQ(lambda[0], 0.25);
            ASSERT_EQ(lambda[1], 0.25);
            ASSERT_EQ(lambda[2], 0.25);
            ASSERT_EQ(lambda[3], 0.25);
        }
    }
}
