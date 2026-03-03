#include <iostream>
#include <ftk2/core/mesh.hpp>
#include <vector>
#include <set>
#include <unordered_set>

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

// Brute-force cofaces: iterate ALL (k+1)-simplices and check containment
void brute_force_cofaces(const RegularSimplicialMesh& mesh, const Simplex& s,
                          std::function<void(const Simplex&)> callback) {
    int target_dim = s.dimension + 1;
    if (target_dim > mesh.get_total_dimension()) return;
    mesh.iterate_simplices(target_dim, [&](const Simplex& candidate) {
        bool contains_all = true;
        for (int i = 0; i <= s.dimension; ++i) {
            bool found = false;
            for (int j = 0; j <= target_dim; ++j) {
                if (candidate.vertices[j] == s.vertices[i]) { found = true; break; }
            }
            if (!found) { contains_all = false; break; }
        }
        if (contains_all) callback(candidate);
    });
}

// Cross-check: LUT cofaces == brute-force cofaces for every simplex in the mesh
void test_cofaces_cross_check(const std::vector<uint64_t>& dims, int max_k) {
    RegularSimplicialMesh mesh(dims);
    int d = mesh.get_total_dimension();
    std::cout << "  Cross-checking cofaces for d=" << d << " mesh "
              << dims[0];
    for (size_t i = 1; i < dims.size(); i++) std::cout << "x" << dims[i];
    std::cout << "..." << std::endl;

    for (int k = 0; k < d && k <= max_k; ++k) {
        int simplex_count = 0;
        int total_cofaces_lut = 0;
        int total_cofaces_bf = 0;
        bool all_match = true;

        mesh.iterate_simplices(k, [&](const Simplex& s) {
            simplex_count++;

            // LUT cofaces
            std::set<Simplex> lut_cofaces;
            mesh.cofaces(s, [&](const Simplex& cf) {
                lut_cofaces.insert(cf);
            });
            total_cofaces_lut += lut_cofaces.size();

            // Brute-force cofaces
            std::set<Simplex> bf_cofaces;
            brute_force_cofaces(mesh, s, [&](const Simplex& cf) {
                bf_cofaces.insert(cf);
            });
            total_cofaces_bf += bf_cofaces.size();

            if (lut_cofaces != bf_cofaces) {
                all_match = false;
                std::cerr << "MISMATCH for k=" << k << " simplex [";
                for (int i = 0; i <= s.dimension; i++) {
                    if (i) std::cerr << ",";
                    std::cerr << s.vertices[i];
                }
                std::cerr << "]: LUT=" << lut_cofaces.size()
                          << " BF=" << bf_cofaces.size() << std::endl;
            }
        });

        ASSERT_TRUE(all_match);
        ASSERT_EQ(total_cofaces_lut, total_cofaces_bf);
        std::cout << "    k=" << k << ": " << simplex_count << " simplices, "
                  << total_cofaces_lut << " total cofaces — OK" << std::endl;
    }
}

void test_cofaces() {
    std::cout << "Testing cofaces LUT..." << std::endl;

    // 2D meshes
    test_cofaces_cross_check({3, 3}, 1);
    test_cofaces_cross_check({4, 5}, 1);

    // 3D meshes
    test_cofaces_cross_check({3, 3, 3}, 2);
    test_cofaces_cross_check({2, 3, 4}, 2);

    // 4D mesh (small, since brute-force is expensive)
    test_cofaces_cross_check({2, 2, 2, 2}, 3);
}

int main() {
    std::cout << "Running Mesh tests..." << std::endl;
    test_mesh();
    test_cofaces();
    std::cout << "Summary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
