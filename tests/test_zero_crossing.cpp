#include <iostream>
#include <ftk2/core/zero_crossing.hpp>

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

#define ASSERT_NEAR(a, b, eps) \
    extern int total_tests; \
    extern int passed_tests; \
    total_tests++; \
    if (std::abs((a) - (b)) < (eps)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_NEAR(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

void test_zero_crossing() {
    std::cout << "Testing Zero Crossing Solver..." << std::endl;

    // K=1
    {
        double values[2][1] = {{-1.0}, {1.0}};
        double lambda[2];
        ASSERT_TRUE(ZeroCrossingSolver<1, double>::solve(values, lambda));
        ASSERT_NEAR(lambda[0], 0.5, 1e-9);
        ASSERT_NEAR(lambda[1], 0.5, 1e-9);
    }

    // K=2
    {
        double values[3][2] = {
            {-0.25, -0.25}, 
            { 0.75, -0.25}, 
            {-0.25,  0.75}
        };
        double lambda[3];
        ASSERT_TRUE(ZeroCrossingSolver<2, double>::solve(values, lambda));
        ASSERT_NEAR(lambda[0], 0.5, 1e-9);
        ASSERT_NEAR(lambda[1], 0.25, 1e-9);
        ASSERT_NEAR(lambda[2], 0.25, 1e-9);
    }

    // K=3
    {
        double values[4][3] = {
            {-0.2, -0.2, -0.2}, 
            { 0.8, -0.2, -0.2}, 
            {-0.2,  0.8, -0.2}, 
            {-0.2, -0.2,  0.8}
        };
        double lambda[4];
        ASSERT_TRUE(ZeroCrossingSolver<3, double>::solve(values, lambda));
        ASSERT_NEAR(lambda[0], 0.4, 1e-9);
        ASSERT_NEAR(lambda[1], 0.2, 1e-9);
        ASSERT_NEAR(lambda[2], 0.2, 1e-9);
        ASSERT_NEAR(lambda[3], 0.2, 1e-9);
    }
}
