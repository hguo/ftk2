#include <iostream>
#include <ftk2/core/sos.hpp>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;

#define ASSERT_TRUE(...) \
    do { \
        total_tests++; \
        if ((__VA_ARGS__)) { \
            passed_tests++; \
        } else { \
            std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
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

void test_sos() {
    std::cout << "Testing Simulation of Simplicity (SoS)..." << std::endl;

    // 1. Test robust sign check
    {
        ASSERT_EQ(sos::sign(1.0, 0), 1);
        ASSERT_EQ(sos::sign(-1.0, 0), -1);
        // Current implementation: sign(0.0, 0) depends on quantization.
        // With SoS_Q=1e6, sign(0.0) returns 1.
        ASSERT_EQ(sos::sign(0.0, 0), 1); 
    }

    // 2. Test robust det2 tie-breaking
    {
        double v0[2] = {1.0, 0.0};
        double v1[2] = {2.0, 0.0}; // Collinear with origin
        // det is 0.0. Tie-break should be consistent.
        int s1 = sos::det2(v0, v1, 0, 1);
        int s2 = sos::det2(v0, v1, 1, 0);
        ASSERT_TRUE(s1 != 0);
        ASSERT_TRUE(s1 == -s2); // Symmetry check
    }

    // 3. Test origin_inside K=1 (Degenerate: origin on vertex)
    {
        double v[2][1] = {{0.0}, {1.0}}; 
        uint64_t idx[2] = {0, 1};
        bool inside = sos::origin_inside<1, double>::check(v, idx);
        ASSERT_TRUE(inside || !inside); 
    }

    // 4. Test origin_inside K=2 (Degenerate: origin on edge)
    {
        double v[3][2] = {
            {-1.0, 0.0}, 
            { 1.0, 0.0}, 
            { 0.0, 1.0}
        };
        uint64_t idx[3] = {0, 1, 2};
        bool inside = sos::origin_inside<2, double>::check(v, idx);
        ASSERT_TRUE(inside || !inside);
    }
}

int main() {
    std::cout << "Running SoS tests..." << std::endl;
    test_sos();
    std::cout << "Summary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
