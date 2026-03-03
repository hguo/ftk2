#include <iostream>
#include <chrono>
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

void bench_cofaces(const std::vector<uint64_t>& dims, int k) {
    RegularSimplicialMesh mesh(dims);
    int d = mesh.get_total_dimension();

    // Collect all k-simplices
    std::vector<Simplex> simplices;
    mesh.iterate_simplices(k, [&](const Simplex& s) { simplices.push_back(s); });

    int n_queries = simplices.size();

    // Benchmark LUT
    int lut_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& s : simplices)
        mesh.cofaces(s, [&](const Simplex&) { lut_count++; });
    auto t1 = std::chrono::high_resolution_clock::now();
    double lut_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Benchmark brute-force
    int bf_count = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (auto& s : simplices)
        brute_force_cofaces(mesh, s, [&](const Simplex&) { bf_count++; });
    auto t3 = std::chrono::high_resolution_clock::now();
    double bf_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    std::cout << "  d=" << d << " mesh ";
    for (size_t i = 0; i < dims.size(); i++) {
        if (i) std::cout << "x";
        std::cout << dims[i];
    }
    std::cout << ", k=" << k << ": " << n_queries << " queries"
              << "  LUT=" << lut_us / 1000.0 << "ms"
              << "  BF=" << bf_us / 1000.0 << "ms"
              << "  speedup=" << bf_us / lut_us << "x" << std::endl;
}

// Brute-force cofaces for extruded mesh: iterate ALL (k+1)-simplices
void brute_force_extruded_cofaces(const ExtrudedSimplicialMesh& mesh, const Simplex& s,
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

void test_extruded_cofaces_cross_check(const std::vector<uint64_t>& base_dims,
                                        uint64_t n_layers, int max_k) {
    auto base = std::make_shared<RegularSimplicialMesh>(base_dims);
    ExtrudedSimplicialMesh mesh(base, n_layers);
    int d = mesh.get_total_dimension();
    std::cout << "  Cross-checking extruded cofaces for " << base_dims[0];
    for (size_t i = 1; i < base_dims.size(); i++) std::cout << "x" << base_dims[i];
    std::cout << "+" << n_layers << "t (d=" << d << ")..." << std::endl;

    for (int k = 0; k < d && k <= max_k; ++k) {
        int simplex_count = 0;
        int total_cofaces_new = 0;
        int total_cofaces_bf = 0;
        bool all_match = true;

        mesh.iterate_simplices(k, [&](const Simplex& s) {
            simplex_count++;

            std::set<Simplex> new_cofaces;
            mesh.cofaces(s, [&](const Simplex& cf) { new_cofaces.insert(cf); });
            total_cofaces_new += new_cofaces.size();

            std::set<Simplex> bf_cofaces;
            brute_force_extruded_cofaces(mesh, s, [&](const Simplex& cf) { bf_cofaces.insert(cf); });
            total_cofaces_bf += bf_cofaces.size();

            if (new_cofaces != bf_cofaces) {
                all_match = false;
                std::cerr << "MISMATCH for k=" << k << " simplex [";
                for (int i = 0; i <= s.dimension; i++) {
                    if (i) std::cerr << ",";
                    std::cerr << s.vertices[i];
                }
                std::cerr << "]: new=" << new_cofaces.size()
                          << " BF=" << bf_cofaces.size() << std::endl;
                // Print missing/extra cofaces
                for (auto& cf : bf_cofaces) {
                    if (new_cofaces.find(cf) == new_cofaces.end()) {
                        std::cerr << "  MISSING: [";
                        for (int i = 0; i <= cf.dimension; i++) {
                            if (i) std::cerr << ",";
                            std::cerr << cf.vertices[i];
                        }
                        std::cerr << "]" << std::endl;
                    }
                }
                for (auto& cf : new_cofaces) {
                    if (bf_cofaces.find(cf) == bf_cofaces.end()) {
                        std::cerr << "  EXTRA: [";
                        for (int i = 0; i <= cf.dimension; i++) {
                            if (i) std::cerr << ",";
                            std::cerr << cf.vertices[i];
                        }
                        std::cerr << "]" << std::endl;
                    }
                }
            }
        });

        ASSERT_TRUE(all_match);
        ASSERT_EQ(total_cofaces_new, total_cofaces_bf);
        std::cout << "    k=" << k << ": " << simplex_count << " simplices, "
                  << total_cofaces_new << " total cofaces — "
                  << (all_match ? "OK" : "FAIL") << std::endl;
    }
}

void test_extruded_cofaces() {
    std::cout << "Testing extruded mesh cofaces..." << std::endl;

    // 1D base + time (2D total)
    test_extruded_cofaces_cross_check({5}, 3, 1);

    // 2D base + time (3D total)
    test_extruded_cofaces_cross_check({3, 3}, 2, 2);
    test_extruded_cofaces_cross_check({4, 3}, 3, 2);

    // 3D base + time (4D total)
    test_extruded_cofaces_cross_check({2, 2, 2}, 2, 3);
    test_extruded_cofaces_cross_check({3, 3, 3}, 1, 2);
}

void bench_extruded_cofaces(const std::vector<uint64_t>& base_dims,
                             uint64_t n_layers, int k) {
    auto base = std::make_shared<RegularSimplicialMesh>(base_dims);
    ExtrudedSimplicialMesh mesh(base, n_layers);

    std::vector<Simplex> simplices;
    mesh.iterate_simplices(k, [&](const Simplex& s) { simplices.push_back(s); });

    // Benchmark combinatorial cofaces
    int new_count = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (auto& s : simplices)
        mesh.cofaces(s, [&](const Simplex&) { new_count++; });
    auto t1 = std::chrono::high_resolution_clock::now();
    double new_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Benchmark brute-force
    int bf_count = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (auto& s : simplices)
        brute_force_extruded_cofaces(mesh, s, [&](const Simplex&) { bf_count++; });
    auto t3 = std::chrono::high_resolution_clock::now();
    double bf_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    std::cout << "  base=";
    for (size_t i = 0; i < base_dims.size(); i++) {
        if (i) std::cout << "x";
        std::cout << base_dims[i];
    }
    std::cout << " layers=" << n_layers << ", k=" << k
              << ": " << simplices.size() << " queries"
              << "  new=" << new_us / 1000.0 << "ms"
              << "  BF=" << bf_us / 1000.0 << "ms"
              << "  speedup=" << bf_us / new_us << "x" << std::endl;
}

void test_extruded_cofaces_benchmark() {
    std::cout << "\nExtruded coface benchmark (combinatorial vs brute-force):" << std::endl;

    // 1D+t
    bench_extruded_cofaces({10}, 5, 0);
    bench_extruded_cofaces({10}, 5, 1);

    // 2D+t
    bench_extruded_cofaces({5, 5}, 3, 1);
    bench_extruded_cofaces({10, 10}, 3, 1);

    // 3D+t
    bench_extruded_cofaces({3, 3, 3}, 2, 2);
    bench_extruded_cofaces({5, 5, 5}, 2, 2);
}

void test_cofaces_benchmark() {
    std::cout << "\nCoface benchmark (LUT vs brute-force):" << std::endl;

    // 2D — increasing mesh size shows O(N^d) vs O(1) scaling
    bench_cofaces({10, 10}, 1);
    bench_cofaces({20, 20}, 1);
    bench_cofaces({50, 50}, 1);

    // 3D — triangle→tet cofaces
    bench_cofaces({5, 5, 5}, 2);
    bench_cofaces({10, 10, 10}, 2);

    // 3D — vertex→edge cofaces
    bench_cofaces({5, 5, 5}, 0);
    bench_cofaces({10, 10, 10}, 0);

    // LUT-only on larger meshes (brute-force too slow)
    std::cout << "\n  LUT-only (larger meshes):" << std::endl;
    for (auto& dims : std::vector<std::vector<uint64_t>>{{50,50,50}, {100,100,100}}) {
        RegularSimplicialMesh mesh(dims);
        std::vector<Simplex> tris;
        mesh.iterate_simplices(2, [&](const Simplex& s) { tris.push_back(s); });
        int count = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (auto& s : tris)
            mesh.cofaces(s, [&](const Simplex&) { count++; });
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  " << dims[0] << "^3: " << tris.size() << " triangles, "
                  << count << " cofaces in " << ms << "ms ("
                  << (ms * 1e6 / tris.size()) << " ns/query)" << std::endl;
    }
}

int main() {
    std::cout << "Running Mesh tests..." << std::endl;
    test_mesh();
    test_cofaces();
    test_extruded_cofaces();
    test_cofaces_benchmark();
    test_extruded_cofaces_benchmark();
    std::cout << "\nSummary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
