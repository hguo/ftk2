// Generator tool: computes coface lookup tables from existing Freudenthal
// iteration LUTs and prints a constexpr header file.
//
// Usage:
//   g++ -std=c++17 -O2 -I include tools/gen_coface_lut.cpp -o build/gen_coface_lut
//   ./build/gen_coface_lut > include/ftk2/core/coface_lut.hpp

#include <ftk2/core/mesh.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <cassert>

using namespace ftk2;

struct CofaceEntry {
    int coface_type;
    int delta[4]; // corner offset, padded to 4
};

// For k=0: find all edges containing vertex (0,...,0).
// An edge of type t at corner delta contains vertex (0,...,0) if
// delta[j] + lut[t][vi][j] == 0 for some vertex vi.
std::vector<CofaceEntry> find_vertex_cofaces(int d) {
    int n_edge_types = cpu_get_num_simplex_types(d, 1);
    std::vector<CofaceEntry> result;
    for (int type = 0; type < n_edge_types; type++) {
        for (int vi = 0; vi < 2; vi++) {
            CofaceEntry e;
            e.coface_type = type;
            for (int j = 0; j < 4; j++)
                e.delta[j] = (j < d) ? -cpu_lut_dispatch(d, 1, type, vi, j) : 0;
            result.push_back(e);
        }
    }
    return result;
}

// For k >= 1: find all (k+1)-simplices containing the k-simplex of given type
// placed at corner (0,...,0).
std::vector<CofaceEntry> find_simplex_cofaces(int d, int k, int type) {
    int n_coface_types = cpu_get_num_simplex_types(d, k + 1);
    if (n_coface_types == 0) return {};

    // Vertices of the k-simplex at corner (0,...,0)
    int sv[5][4];
    for (int vi = 0; vi <= k; vi++)
        for (int j = 0; j < 4; j++)
            sv[vi][j] = (j < d) ? cpu_lut_dispatch(d, k, type, vi, j) : 0;

    std::vector<CofaceEntry> result;

    int total_deltas = 1;
    for (int i = 0; i < d; i++) total_deltas *= 3;

    for (int ct = 0; ct < n_coface_types; ct++) {
        // Coface vertices at corner (0,...,0)
        int cv[5][4];
        for (int vi = 0; vi <= k + 1; vi++)
            for (int j = 0; j < 4; j++)
                cv[vi][j] = (j < d) ? cpu_lut_dispatch(d, k + 1, ct, vi, j) : 0;

        for (int di = 0; di < total_deltas; di++) {
            int delta[4] = {0, 0, 0, 0};
            int tmp = di;
            for (int j = 0; j < d; j++) {
                delta[j] = (tmp % 3) - 1;
                tmp /= 3;
            }

            // Check if all simplex vertices ⊂ coface vertices shifted by delta
            bool all_found = true;
            for (int vi = 0; vi <= k && all_found; vi++) {
                bool found = false;
                for (int cvi = 0; cvi <= k + 1 && !found; cvi++) {
                    bool match = true;
                    for (int j = 0; j < d && match; j++) {
                        if (sv[vi][j] != delta[j] + cv[cvi][j])
                            match = false;
                    }
                    if (match) found = true;
                }
                if (!found) all_found = false;
            }

            if (all_found) {
                CofaceEntry e;
                e.coface_type = ct;
                for (int j = 0; j < 4; j++) e.delta[j] = delta[j];
                result.push_back(e);
            }
        }
    }

    return result;
}

// Print a single array entry {coface_type, delta[0], ..., delta[d-1]}
void print_entry(const CofaceEntry& e, int d) {
    std::cout << "{" << e.coface_type;
    for (int j = 0; j < d; j++)
        std::cout << "," << e.delta[j];
    std::cout << "}";
}

void print_vertex_cofaces(int d) {
    auto cofaces = find_vertex_cofaces(d);
    int count = cofaces.size();

    std::cout << "// d=" << d << ", k=0: vertex -> edge cofaces\n";
    std::cout << "inline constexpr int cpu_coface_count_" << d << "_0 = " << count << ";\n";
    std::cout << "inline constexpr int cpu_coface_lut_" << d << "_0[" << count << "][" << (1 + d) << "] = {\n";
    for (int i = 0; i < count; i++) {
        std::cout << "    ";
        print_entry(cofaces[i], d);
        if (i + 1 < count) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "};\n\n";
}

void print_simplex_cofaces(int d, int k) {
    int n_types = cpu_get_num_simplex_types(d, k);
    assert(n_types > 0);

    // Compute cofaces for each type
    std::vector<std::vector<CofaceEntry>> all_cofaces(n_types);
    int max_count = 0;
    for (int t = 0; t < n_types; t++) {
        all_cofaces[t] = find_simplex_cofaces(d, k, t);
        if ((int)all_cofaces[t].size() > max_count)
            max_count = all_cofaces[t].size();
    }

    std::cout << "// d=" << d << ", k=" << k << ": " << k << "-simplex -> "
              << (k + 1) << "-simplex cofaces\n";
    std::cout << "inline constexpr int cpu_coface_max_" << d << "_" << k << " = " << max_count << ";\n";
    std::cout << "inline constexpr int cpu_coface_count_" << d << "_" << k << "[" << n_types << "] = {";
    for (int t = 0; t < n_types; t++) {
        if (t > 0) std::cout << ",";
        std::cout << (int)all_cofaces[t].size();
    }
    std::cout << "};\n";

    std::cout << "inline constexpr int cpu_coface_lut_" << d << "_" << k
              << "[" << n_types << "][" << max_count << "][" << (1 + d) << "] = {\n";
    for (int t = 0; t < n_types; t++) {
        std::cout << "    {";
        for (int i = 0; i < max_count; i++) {
            if (i > 0) std::cout << ", ";
            if (i < (int)all_cofaces[t].size())
                print_entry(all_cofaces[t][i], d);
            else {
                // Pad with zeros
                std::cout << "{0";
                for (int j = 0; j < d; j++) std::cout << ",0";
                std::cout << "}";
            }
        }
        std::cout << "}";
        if (t + 1 < n_types) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "};\n\n";
}

void print_dispatch_functions() {
    std::cout << R"(// Dispatch: number of coface entries for a given (d, k, type).
// For k=0, type is ignored (all vertices are equivalent).
inline int cpu_coface_count_dispatch(int d, int k, int type) {
    if (k == 0) {
        if (d == 2) return cpu_coface_count_2_0;
        if (d == 3) return cpu_coface_count_3_0;
        if (d == 4) return cpu_coface_count_4_0;
    }
    if (d == 2 && k == 1) return cpu_coface_count_2_1[type];
    if (d == 3 && k == 1) return cpu_coface_count_3_1[type];
    if (d == 3 && k == 2) return cpu_coface_count_3_2[type];
    if (d == 4 && k == 1) return cpu_coface_count_4_1[type];
    if (d == 4 && k == 2) return cpu_coface_count_4_2[type];
    if (d == 4 && k == 3) return cpu_coface_count_4_3[type];
    return 0;
}

// Dispatch: coface LUT entry field for (d, k, type, coface_index, field).
// field layout: [0]=coface_type, [1..d]=delta coordinates.
inline int cpu_coface_lut_dispatch(int d, int k, int type, int ci, int field) {
    if (k == 0) {
        if (d == 2) return cpu_coface_lut_2_0[ci][field];
        if (d == 3) return cpu_coface_lut_3_0[ci][field];
        if (d == 4) return cpu_coface_lut_4_0[ci][field];
    }
    if (d == 2 && k == 1) return cpu_coface_lut_2_1[type][ci][field];
    if (d == 3 && k == 1) return cpu_coface_lut_3_1[type][ci][field];
    if (d == 3 && k == 2) return cpu_coface_lut_3_2[type][ci][field];
    if (d == 4 && k == 1) return cpu_coface_lut_4_1[type][ci][field];
    if (d == 4 && k == 2) return cpu_coface_lut_4_2[type][ci][field];
    if (d == 4 && k == 3) return cpu_coface_lut_4_3[type][ci][field];
    return 0;
}
)";
}

int main() {
    std::cout << "// Auto-generated by tools/gen_coface_lut.cpp — do not edit.\n";
    std::cout << "#pragma once\n\n";
    std::cout << "namespace ftk2 {\n\n";

    // d=2: k=0 (vertex→edge), k=1 (edge→triangle)
    print_vertex_cofaces(2);
    print_simplex_cofaces(2, 1);

    // d=3: k=0..2
    print_vertex_cofaces(3);
    print_simplex_cofaces(3, 1);
    print_simplex_cofaces(3, 2);

    // d=4: k=0..3
    print_vertex_cofaces(4);
    print_simplex_cofaces(4, 1);
    print_simplex_cofaces(4, 2);
    print_simplex_cofaces(4, 3);

    print_dispatch_functions();

    std::cout << "} // namespace ftk2\n";

    return 0;
}
