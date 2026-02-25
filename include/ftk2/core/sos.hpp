#pragma once

#include <ftk2/core/mesh.hpp>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>

namespace ftk2 {
namespace sos {

template <typename T>
FTK_HOST_DEVICE
inline int sign(T v, uint64_t index, double Q = 1000000.0) {
    __int128 qv = static_cast<__int128>(static_cast<double>(v) * Q + (v > 0 ? 0.5 : -0.5));
    if (qv > 0) return 1;
    if (qv < 0) return -1;
    // Tie-breaking: Use index parity to ensure consistent sign for zero values
    return (index % 2 == 0) ? 1 : -1;
}

template <typename T>
FTK_HOST_DEVICE
inline int det2(const T v0[2], const T v1[2], uint64_t i0, uint64_t i1, double Q = 1000000.0) {
    auto q = [Q](T val) { return static_cast<__int128>(static_cast<double>(val) * Q + (val > 0 ? 0.5 : -0.5)); };
    __int128 a = q(v0[0]), b = q(v0[1]);
    __int128 c = q(v1[0]), d = q(v1[1]);
    __int128 det = a * d - b * c;
    if (det > 0) return 1;
    if (det < 0) return -1;
    // Lexicographical tie-breaking for consistency
    if (i0 < i1) return 1;
    if (i0 > i1) return -1;
    return 0;
}

template <typename T>
FTK_HOST_DEVICE
inline int det3(const T v0[3], const T v1[3], const T v2[3], uint64_t i0, uint64_t i1, uint64_t i2, double Q = 1000000.0) {
    auto q = [Q](T val) { return static_cast<__int128>(static_cast<double>(val) * Q + (val > 0 ? 0.5 : -0.5)); };
    __int128 a = q(v0[0]), b = q(v0[1]), c = q(v0[2]);
    __int128 d = q(v1[0]), e = q(v1[1]), f = q(v1[2]);
    __int128 g = q(v2[0]), h = q(v2[1]), i = q(v2[2]);
    __int128 det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    
    if (det > 0) return 1;
    if (det < 0) return -1;
    
    // Manual sort for device compatibility
    uint64_t i_min = i0, i_mid = i1, i_max = i2;
    if (i_min > i_mid) { uint64_t tmp = i_min; i_min = i_mid; i_mid = tmp; }
    if (i_mid > i_max) { uint64_t tmp = i_mid; i_mid = i_max; i_max = tmp; }
    if (i_min > i_mid) { uint64_t tmp = i_min; i_min = i_mid; i_mid = tmp; }
    
    uint64_t h_val = i_min ^ (i_mid << 1) ^ (i_max << 2);
    return (h_val % 2 == 0) ? 1 : -1;
}

template <int K, typename T>
class origin_inside {
public:
    FTK_HOST_DEVICE
    static bool check(const T values[K+1][K], const uint64_t indices[K+1], double Q = 1000000.0) {
        return false;
    }
};

template <typename T>
class origin_inside<1, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[2][1], const uint64_t idx[2], double Q = 1000000.0) {
        int s0 = sign(v[0][0], idx[0], Q);
        int s1 = sign(v[1][0], idx[1], Q);
        return (s0 != s1);
    }
};

template <typename T>
class origin_inside<2, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[3][2], const uint64_t idx[3], double Q = 1000000.0) {
        int s0 = det2(v[0], v[1], idx[0], idx[1], Q);
        int s1 = det2(v[1], v[2], idx[1], idx[2], Q);
        int s2 = det2(v[2], v[0], idx[2], idx[0], Q);
        return (s0 == s1 && s1 == s2);
    }
};

template <typename T>
class origin_inside<3, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[4][3], const uint64_t idx[4], double Q = 1000000.0) {
        // Origin O is inside tetrahedron (v0, v1, v2, v3) iff the 4 oriented volumes 
        // Vi = det(O, face_i) have the same sign.
        // We use consistent orientations for each face:
        // d0 = det(v1, v3, v2)
        // d1 = det(v0, v2, v3)
        // d2 = det(v0, v3, v1)
        // d3 = det(v0, v1, v2)
        
        int d0 = det3(v[1], v[3], v[2], idx[1], idx[3], idx[2], Q);
        int d1 = det3(v[0], v[2], v[3], idx[0], idx[2], idx[3], Q);
        int d2 = det3(v[0], v[3], v[1], idx[0], idx[3], idx[1], Q);
        int d3 = det3(v[0], v[1], v[2], idx[0], idx[1], idx[2], Q);
        
        return (d0 == d1 && d1 == d2 && d2 == d3);
    }
};

} // namespace sos
} // namespace ftk2
