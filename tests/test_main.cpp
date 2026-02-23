#include <iostream>
#include <vector>
#include <string>

// Simple test framework
int total_tests = 0;
int passed_tests = 0;

void test_zero_crossing();
void test_mesh();
void test_engine();

int main(int argc, char** argv) {
    std::cout << "Running FTK2 tests..." << std::endl;
    
    test_zero_crossing();
    test_mesh();
    test_engine();
    
    std::cout << "\nSummary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}

#define ASSERT_TRUE(...) \
    total_tests++; \
    if ((__VA_ARGS__)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }

#define ASSERT_NEAR(a, b, eps) \
    total_tests++; \
    if (std::abs((a) - (b)) < (eps)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_NEAR(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    }
