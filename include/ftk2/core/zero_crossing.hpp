#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <ndarray/ndarray.hh>
#include <array>
#include <cmath>

namespace ftk2 {

/**
 * @brief Mathematical solver for finding zero-crossings in K-simplices.
 * 
 * Given values at K+1 vertices, solves f(x) = 0 using linear interpolation.
 */
template <int K, typename T = double>
class ZeroCrossingSolver {
public:
    /**
     * @brief Solve for zero-crossing within a K-simplex.
     * 
     * @param values An array of (K+1) vectors, each of dimension K.
     *               values[i][j] is the j-th component of the field at vertex i.
     * @param lambda Output barycentric coordinates (K+1).
     * @return true if a zero-crossing exists inside the simplex.
     */
    FTK_HOST_DEVICE
    static bool solve(const T values[K+1][K], T lambda[K+1]) {
        // We solve the system: sum_{i=0}^K lambda_i * values_i = 0
        // with the constraint: sum_{i=0}^K lambda_i = 1
        
        // This transforms to a K x K system:
        // sum_{i=0}^{K-1} lambda_i * (values_i - values_K) = -values_K
        
        T A[K][K];
        T b[K];
        
        for (int j = 0; j < K; ++j) {
            for (int i = 0; i < K; ++i) {
                A[j][i] = values[i][j] - values[K][j];
            }
            b[j] = -values[K][j];
        }
        
        // Solve the system A * lambda_sub = b
        T lambda_sub[K];
        if (!solve_linear_system(A, b, lambda_sub)) {
            return false;
        }
        
        // Compute the last barycentric coordinate
        T sum_lambda = 0;
        for (int i = 0; i < K; ++i) {
            lambda[i] = lambda_sub[i];
            sum_lambda += lambda[i];
        }
        lambda[K] = T(1.0) - sum_lambda;
        
        // Check if the zero is inside the simplex
        const T eps = T(1e-9);
        for (int i = 0; i <= K; ++i) {
            if (lambda[i] < -eps || lambda[i] > T(1.0) + eps) {
                return false;
            }
        }
        
        return true;
    }

private:
    /**
     * @brief Simple Gaussian elimination with partial pivoting for small K.
     */
    FTK_HOST_DEVICE
    static bool solve_linear_system(T A[K][K], const T b[K], T x[K]) {
        T temp_A[K][K];
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) temp_A[i][j] = A[i][j];
            x[i] = b[i];
        }

        for (int i = 0; i < K; ++i) {
            // Pivoting
            int pivot = i;
            for (int j = i + 1; j < K; ++j) {
                if (std::abs(temp_A[j][i]) > std::abs(temp_A[pivot][i])) pivot = j;
            }
            
            std::swap(temp_A[i], temp_A[pivot]);
            std::swap(x[i], x[pivot]);

            if (std::abs(temp_A[i][i]) < T(1e-12)) return false;

            for (int j = i + 1; j < K; ++j) {
                T factor = temp_A[j][i] / temp_A[i][i];
                x[j] -= factor * x[i];
                for (int l = i + 1; l < K; ++l) {
                    temp_A[j][l] -= factor * temp_A[i][l];
                }
            }
        }

        // Back substitution
        for (int i = K - 1; i >= 0; --i) {
            T sum = 0;
            for (int j = i + 1; j < K; ++j) sum += temp_A[i][j] * x[j];
            x[i] = (x[i] - sum) / temp_A[i][i];
        }
        
        return true;
    }
};

/**
 * @brief Specialization for K=1 (Isosurface/Contour).
 */
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
