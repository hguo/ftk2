#pragma once

#include <ftk2/core/mesh.hpp>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>

namespace ftk2 {
namespace sos {

static constexpr double SoS_Q = 1000000.0;

template <typename T>
FTK_HOST_DEVICE
inline int sign(T v, uint64_t index) {
    long long qv = static_cast<long long>(v * SoS_Q + (v > 0 ? 0.5 : -0.5));
    if (qv > 0) return 1;
    if (qv < 0) return -1;
    // Tie-breaking: Use index parity to ensure consistent sign for zero values
    return (index % 2 == 0) ? 1 : -1;
}

template <typename T>
FTK_HOST_DEVICE
inline int det2(const T v0[2], const T v1[2], uint64_t i0, uint64_t i1) {
    auto q = [](T val) { return static_cast<__int128>(val * SoS_Q + (val > 0 ? 0.5 : -0.5)); };
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
inline int det3(const T v0[3], const T v1[3], const T v2[3], uint64_t i0, uint64_t i1, uint64_t i2) {
    auto q = [](T val) { return static_cast<__int128>(val * SoS_Q + (val > 0 ? 0.5 : -0.5)); };
    __int128 a = q(v0[0]), b = q(v0[1]), c = q(v0[2]);
    __int128 d = q(v1[0]), e = q(v1[1]), f = q(v1[2]);
    __int128 g = q(v2[0]), h = q(v2[1]), i = q(v2[2]);
    __int128 det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    if (det > 0) return 1;
    if (det < 0) return -1;
    if (i0 < i1 && i1 < i2) return 1;
    if (i0 > i1 && i1 > i2) return -1;
    return (i0 < i2) ? 1 : -1;
}

template <int K, typename T>
class origin_inside {
public:
    FTK_HOST_DEVICE
    static bool check(const T values[K+1][K], const uint64_t indices[K+1]) {
        return false;
    }
};

template <typename T>
class origin_inside<1, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[2][1], const uint64_t idx[2]) {
        int s0 = sign(v[0][0], idx[0]);
        int s1 = sign(v[1][0], idx[1]);
        return (s0 != s1);
    }
};

template <typename T>
class origin_inside<2, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[3][2], const uint64_t idx[3]) {
        int s0 = det2(v[0], v[1], idx[0], idx[1]);
        int s1 = det2(v[1], v[2], idx[1], idx[2]);
        int s2 = det2(v[2], v[0], idx[2], idx[0]);
        return (s0 == s1 && s1 == s2);
    }
};

template <typename T>
class origin_inside<3, T> {
public:
    FTK_HOST_DEVICE
    static bool check(const T v[4][3], const uint64_t idx[4]) {
        int s[4];
        s[0] = det3(v[0], v[1], v[2], idx[0], idx[1], idx[2]);
        s[1] = det3(v[1], v[2], v[3], idx[1], idx[2], idx[3]);
        s[2] = det3(v[2], v[3], v[0], idx[2], idx[3], idx[0]);
        s[3] = det3(v[3], v[0], v[1], idx[3], idx[0], idx[1]);
        
        // In 3D, for origin to be inside, signs must alternate:
        // s1 == s3 and s0 == s2 and s0 != s1
        return (s[1] == s[3] && s[0] == s[2] && s[0] != s[1]);
    }
};

} // namespace sos
} // namespace ftk2
