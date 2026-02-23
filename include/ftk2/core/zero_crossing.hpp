#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <array>
#include <cmath>

namespace ftk2 {

/**
 * @brief Mathematical solver for finding zero-crossings in K-simplices.
 */
template <int K, typename T = double>
class ZeroCrossingSolver {
public:
    /**
     * @brief Solve for zero-crossing within a K-simplex.
     */
    FTK_HOST_DEVICE
    static bool solve(const T values[K+1][K], T lambda[K+1]) {
        T A[K][K];
        T b[K];
        
        for (int j = 0; j < K; ++j) {
            for (int i = 0; i < K; ++i) {
                A[j][i] = values[i][j] - values[K][j];
            }
            b[j] = -values[K][j];
        }
        
        T lambda_sub[K];
        if (!solve_linear_system(A, b, lambda_sub)) return false;
        
        T sum_lambda = 0;
        for (int i = 0; i < K; ++i) {
            lambda[i] = lambda_sub[i];
            sum_lambda += lambda[i];
        }
        lambda[K] = T(1.0) - sum_lambda;
        
        const T eps = T(1e-9);
        for (int i = 0; i <= K; ++i) {
            if (lambda[i] < -eps || lambda[i] > T(1.0) + eps) return false;
        }
        
        return true;
    }

private:
    FTK_HOST_DEVICE
    static bool solve_linear_system(T A[K][K], const T b[K], T x[K]) {
        T temp_A[K][K];
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) temp_A[i][j] = A[i][j];
            x[i] = b[i];
        }

        for (int i = 0; i < K; ++i) {
            int pivot = i;
            for (int j = i + 1; j < K; ++j) {
                if (std::abs(temp_A[j][i]) > std::abs(temp_A[pivot][i])) pivot = j;
            }
            
            // Swap rows
            for (int l = 0; l < K; ++l) {
                T tmp = temp_A[i][l];
                temp_A[i][l] = temp_A[pivot][l];
                temp_A[pivot][l] = tmp;
            }
            T tmp_b = x[i]; x[i] = x[pivot]; x[pivot] = tmp_b;

            if (std::abs(temp_A[i][i]) < T(1e-12)) return false;

            for (int j = i + 1; j < K; ++j) {
                T factor = temp_A[j][i] / temp_A[i][i];
                x[j] -= factor * x[i];
                for (int l = i + 1; l < K; ++l) {
                    temp_A[j][l] -= factor * temp_A[i][l];
                }
            }
        }

        for (int i = K - 1; i >= 0; --i) {
            T sum = 0;
            for (int j = i + 1; j < K; ++j) sum += temp_A[i][j] * x[j];
            x[i] = (x[i] - sum) / temp_A[i][i];
        }
        
        return true;
    }
};

template <typename T>
class ZeroCrossingSolver<1, T> {
public:
    FTK_HOST_DEVICE
    static bool solve(const T values[2][1], T lambda[2]) {
        T d = values[0][0] - values[1][0];
        if (std::abs(d) < T(1e-12)) return false;
        lambda[0] = -values[1][0] / d;
        lambda[1] = T(1.0) - lambda[0];
        const T eps = T(1e-9);
        return (lambda[0] >= -eps && lambda[0] <= T(1.0) + eps);
    }
};

} // namespace ftk2
