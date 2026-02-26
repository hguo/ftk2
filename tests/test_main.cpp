#include <iostream>
#include <vector>
#include <string>
#include <cmath>

// Simple test framework
extern int total_tests;
extern int passed_tests;

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

#define ASSERT_NEAR(a, b, eps) \
    total_tests++; \
    if (std::abs((a) - (b)) < (eps)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_NEAR(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

void test_zero_crossing();
void test_mesh();
void test_engine();
void test_sos();
void test_parity();
void test_examples();
void test_unstructured();
void test_3d_cp_system();
void test_exactpv();

int main(int argc, char** argv) {
    std::cout << "Running FTK2 tests..." << std::endl;

    test_exactpv();
    test_3d_cp_system();
    test_unstructured();
    test_zero_crossing();
    test_mesh();
    test_engine();
    test_sos();
    test_parity();
    test_examples();

    std::cout << "\nSummary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}

int total_tests = 0;
int passed_tests = 0;
