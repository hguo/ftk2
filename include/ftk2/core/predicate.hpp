#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/zero_crossing.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/feature.hpp>
#include <ftk2/numeric/parallel_vector_solver.hpp>
#include <ndarray/ndarray.hh>
#include <vector>
#include <string>
#include <map>

#ifdef __CUDACC__
#include <ftk2/core/device_mesh.hpp>
#endif

namespace ftk2 {

/**
 * @brief Base class for feature predicates.
 */
template <int M, typename T = double>
struct Predicate {
    static constexpr int codimension = M;
    double sos_q = 1000000.0;
};

// Forward declaration for device view
template <typename T> struct CudaDataView;

/**
 * @brief Attribute configuration for recording at feature locations
 */
struct AttributeSpec {
    std::string name;           // Attribute name
    std::string source;         // Source data array name
    std::string type = "scalar"; // scalar, magnitude, component_X
    int component = -1;         // For multi-component: which component
    int slot = -1;              // Which attributes[] slot to use (0-15)
};

/**
 * @brief Predicate for finding critical points (zeros of a vector field).
 *
 * Supports two data formats:
 * 1. Multi-component array: Single array with shape [M, spatial..., time]
 * 2. Separate arrays: M separate arrays (legacy, deprecated)
 */
template <int M, typename T = double>
struct CriticalPointPredicate : public Predicate<M, T> {
    // Multi-component mode (preferred)
    std::string vector_var_name;  // Name of multi-component vector array

    // Legacy mode (deprecated - separate scalar arrays)
    std::string var_names[M];

    // Optional scalar field for classification
    std::string scalar_var_name;

    // Data format flag
    bool use_multicomponent = true;  // Default to multi-component

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[M+1][M], FeatureElement& el, 
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[M + 1];
        for(int i=0; i<=M; ++i) indices[i] = s.vertices[i];
        if (!sos::origin_inside<M, T>::check(values, indices, this->sos_q)) return false;
        
        T lambda[M + 1];
        if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) return false; 
        
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];

        el.type = 0; el.scalar = 0.0f;
        if (!scalar_var_name.empty() && arrays.size() > M && mesh) {
            T s_val = 0;
            for (int i = 0; i <= M; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                const auto& arr = *arrays[M];
                if (coords.size() == 1) s_val += lambda[i] * arr.f(coords[0]);
                else if (coords.size() == 2) s_val += lambda[i] * arr.f(coords[0], coords[1]);
                else if (coords.size() == 3) s_val += lambda[i] * arr.f(coords[0], coords[1], coords[2]);
                else if (coords.size() == 4) s_val += lambda[i] * arr.f(coords[0], coords[1], coords[2], coords[3]);
            }
            el.scalar = (float)s_val;
        }

        // Initialize attributes to zero
        // (Actual attribute interpolation is handled by the engine after extract_it succeeds)
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;

        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = M;
        bool has_scalar;
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[M + 1][M]; uint64_t indices[M + 1];
            for (int i = 0; i <= M; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                for (int j = 0; j < M && j < n_vars; ++j) values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]);
            }
            if (!sos::origin_inside<M, T>::check(values, indices, sos_q)) return false;
            T lambda[M + 1]; if (!ZeroCrossingSolver<M, T>::solve(values, lambda)) { for (int i = 0; i <= M; ++i) lambda[i] = (T)1.0 / (M + 1); }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= M; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = 0.0f;
            if (has_scalar && n_vars > M) {
                T s_val = 0; for (int i = 0; i <= M; ++i) { uint64_t coords[4] = {0}; mesh.id_to_coords(s.vertices[i], coords); s_val += lambda[i] * data[M].f(coords[0], coords[1], coords[2], coords[3]); }
                el.scalar = (float)s_val;
            }
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true; 
        }
    };
    Device get_device() const { return {!scalar_var_name.empty(), this->sos_q}; }
#endif
};

/**
 * @brief Predicate for finding contours (isosurfaces of a scalar field).
 */
template <typename T = double>
struct ContourPredicate : public Predicate<1, T> {
    std::string var_name;
    T threshold = (T)0.0;

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[2][1], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[2] = {s.vertices[0], s.vertices[1]};
        T adjusted_values[2][1] = {{values[0][0] - threshold}, {values[1][0] - threshold}};
        if (!sos::origin_inside<1, T>::check(adjusted_values, indices, this->sos_q)) return false;
        T lambda[2]; if (!ZeroCrossingSolver<1, T>::solve(adjusted_values, lambda)) { for (int i = 0; i <= 1; ++i) lambda[i] = (T)0.5; }
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        el.scalar = (float)threshold;
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 1;
        T threshold;
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[2][1]; uint64_t indices[2];
            for (int i = 0; i <= 1; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                values[i][0] = data[0].f(coords[0], coords[1], coords[2], coords[3]) - threshold;
            }
            if (!sos::origin_inside<1, T>::check(values, indices, sos_q)) return false;
            T lambda[2]; if (!ZeroCrossingSolver<1, T>::solve(values, lambda)) { for (int i = 0; i <= 1; ++i) lambda[i] = (T)0.5; }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= 1; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = (float)threshold;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }
    };
    Device get_device() const { return {threshold, this->sos_q}; }
#endif
};

/**
 * @brief Predicate for finding TDGL magnetic vortices in superconductor simulations.
 *
 * Detects topological defects (vortex cores) in complex order parameter fields.
 * Vortices are characterized by:
 * - Phase winding around the core (winding number ±1, ±2, ...)
 * - Amplitude drop to zero at the core
 * - Gauge-invariant detection using magnetic vector potential
 *
 * Input data format:
 * - Complex field: ψ = ρ exp(iθ) where ρ is amplitude, θ is phase
 * - Requires: re (real), im (imaginary), or rho (amplitude) + phi (phase)
 * - Optional: magnetic vector potential A for gauge transformation
 */
template <typename T = double>
struct TDGLVortexPredicate : public Predicate<2, T> {
    // Input field names
    std::string re_name = "re";   // Real part of order parameter
    std::string im_name = "im";   // Imaginary part of order parameter
    std::string rho_name = "rho"; // Amplitude (optional, computed from re/im if not provided)
    std::string phi_name = "phi"; // Phase (optional, computed from re/im if not provided)

    // Magnetic vector potential (optional, for gauge transformation)
    std::string Ax_name = "";  // x-component
    std::string Ay_name = "";  // y-component
    std::string Az_name = "";  // z-component (3D only)

    // Minimum winding number to detect
    int min_winding = 1;

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[3][2], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const
    {
        // values[i][0] = re, values[i][1] = im for vertex i
        uint64_t indices[3] = {s.vertices[0], s.vertices[1], s.vertices[2]};

        // Compute phase from complex components
        T phi[3], rho[3];
        for (int i = 0; i < 3; ++i) {
            T re = values[i][0];
            T im = values[i][1];
            rho[i] = std::sqrt(re * re + im * im);
            phi[i] = std::atan2(im, re);
        }

        // Compute phase differences around triangle with gauge transformation
        T delta[3], phase_shift = 0;
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;

            // Phase difference
            T dphi = phi[j] - phi[i];

            // TODO: Add magnetic potential contribution for gauge transformation
            // For now, use simple phase difference
            // dphi -= line_integral(A, X[i], X[j]);

            // Wrap to [-π, π]
            delta[i] = std::remainder(dphi, 2.0 * M_PI);
            phase_shift -= delta[i];
        }

        // Compute winding number
        T winding = phase_shift / (2.0 * M_PI);
        int winding_int = std::round(winding);

        // Check if winding number is significant
        if (std::abs(winding_int) < min_winding) return false;

        // Use SoS to break ties (consistent orientation)
        if (!sos::sign(winding, indices[0], this->sos_q)) return false;

        // Solve for barycentric coordinates (center of triangle for now)
        T lambda[3] = {(T)1.0 / 3.0, (T)1.0 / 3.0, (T)1.0 / 3.0};

        el = FeatureElement();
        el.simplex = s;
        el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];

        // Store winding number as type
        el.type = winding_int;

        // Store interpolated amplitude as scalar
        el.scalar = (float)((rho[0] + rho[1] + rho[2]) / 3.0);

        // Initialize attributes
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;

        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 2;
        int min_winding;
        double sos_q;

        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[3][2]; uint64_t indices[3];
            for (int i = 0; i <= 2; ++i) {
                indices[i] = s.vertices[i];
                uint64_t coords[4] = {0};
                mesh.id_to_coords(indices[i], coords);

                // data[0] = re, data[1] = im
                for (int j = 0; j < 2 && j < n_vars; ++j) {
                    values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]);
                }
            }

            // Compute phases and phase shift
            T phi[3], rho[3], phase_shift = 0;
            for (int i = 0; i < 3; ++i) {
                T re = values[i][0], im = values[i][1];
                rho[i] = sqrt(re * re + im * im);
                phi[i] = atan2(im, re);
            }

            for (int i = 0; i < 3; ++i) {
                int j = (i + 1) % 3;
                T delta = remainder(phi[j] - phi[i], 2.0 * M_PI);
                phase_shift -= delta;
            }

            T winding = phase_shift / (2.0 * M_PI);
            int winding_int = round(winding);

            if (abs(winding_int) < min_winding) return false;
            if (!sos::sign(winding, indices[0], sos_q)) return false;

            el = FeatureElement();
            el.simplex = s;
            el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = 1.0f / 3.0f;
            el.type = winding_int;
            el.scalar = (float)((rho[0] + rho[1] + rho[2]) / 3.0f);
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;

            return true;
        }
    };

    Device get_device() const { return {min_winding, this->sos_q}; }
#endif
};

/**
 * @brief Predicate for finding fibers (intersections of two isosurfaces from two scalar fields).
 */
template <typename T = double>
struct FiberPredicate : public Predicate<2, T> {
    std::string var_names[2];
    T thresholds[2] = {(T)0.0, (T)0.0};

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    bool extract_it(const Simplex& s, const T values[3][2], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const 
    {
        uint64_t indices[3] = {s.vertices[0], s.vertices[1], s.vertices[2]};
        T adj[3][2]; for(int i=0; i<3; ++i) for(int j=0; j<2; ++j) adj[i][j] = values[i][j] - thresholds[j];
        if (!sos::origin_inside<2, T>::check(adj, indices, this->sos_q)) return false;
        T lambda[3]; if (!ZeroCrossingSolver<2, T>::solve(adj, lambda)) { for (int i = 0; i <= 2; ++i) lambda[i] = (T)1.0 / 3.0; }
        
        el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
        for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
        el.scalar = 0.0f;
        for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
        return true;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 2;
        T thresholds[2];
        double sos_q;
        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars, const DeviceMesh& mesh, FeatureElement& el) const {
            T values[3][2]; uint64_t indices[3];
            for (int i = 0; i <= 2; ++i) {
                indices[i] = s.vertices[i]; uint64_t coords[4] = {0}; mesh.id_to_coords(indices[i], coords);
                for (int j = 0; j < 2 && j < n_vars; ++j) values[i][j] = data[j].f(coords[0], coords[1], coords[2], coords[3]) - thresholds[j];
            }
            if (!sos::origin_inside<2, T>::check(values, indices, sos_q)) return false;
            T lambda[3]; if (!ZeroCrossingSolver<2, T>::solve(values, lambda)) { for (int i = 0; i <= 2; ++i) lambda[i] = (T)1.0 / 3.0; }
            el = FeatureElement(); // Full zero-initialization
        el.simplex = s; el.geometry_type = FeatureGeometryType::Point;
            for (int i = 0; i <= 2; ++i) el.barycentric_coords[0][i] = (float)lambda[i];
            el.type = 0; el.scalar = 0.0f;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }
    };
    Device get_device() const { return {{thresholds[0], thresholds[1]}, this->sos_q}; }
#endif
};

/**
 * @brief Predicate for exact parallel vector (ExactPV) tracking
 *
 * Finds locations where two vector fields v and w are parallel (v × w = 0).
 * Uses analytical cubic rational solver for exact detection.
 *
 * Codimension M=2:
 * - 3D spatial: Parallel vectors form curves (1D manifolds in 3D space)
 * - 4D spacetime: Parallel vectors form surfaces (2D manifolds in 4D space)
 */
template <typename T = double>
struct ExactPVPredicate : public Predicate<6, T> {  // 6 components: 3 for u, 3 for v
    static constexpr int codimension = 2;

    // Multi-component mode (preferred): single array with 6 components [ux, uy, uz, vx, vy, vz]
    std::string vector_var_name = "uv";  // Combined vector field

    // Legacy mode: separate vector fields
    std::string vector_u_name = "u";  // First vector field (3D)
    std::string vector_v_name = "v";  // Second vector field (3D)
    bool use_multicomponent = true;  // Default to multi-component

    // Attributes to record at feature locations
    std::vector<AttributeSpec> attributes;

    // Storage for parametric curves (3D spatial)
    // Note: These are mutable because they are filled during extraction
    mutable std::vector<PVCurveSegment> curve_segments;

    // Storage for parametric surfaces (4D spacetime)
    mutable std::vector<PVSurfacePatch> surface_patches;

    /**
     * @brief Extract parallel vector features from a simplex
     *
     * For 3D spatial (triangles): Detects puncture points (up to 3 per triangle)
     * For 4D spacetime: Extracts surface patches
     *
     * Note: This predicate stores parametric representations (curves/surfaces)
     * for later adaptive sampling and visualization.
     */
    bool extract_simplex(const Simplex& s,
                        const std::map<std::string, ftk::ndarray<T>>& data,
                        std::vector<FeatureElement>& elements) const;

    // CPU implementation for triangles (3 vertices, 2 vector fields of 3 components each)
    bool extract_it(const Simplex& s, const T values[3][6], FeatureElement& el,
                   const std::vector<const ftk::ndarray<T>*>& arrays = {}, const Mesh* mesh = nullptr) const
    {
        // values layout: [3 vertices][6 components: ux, uy, uz, vx, vy, vz]
        T V[3][3], W[3][3];
        for (int i = 0; i < 3; ++i) {
            V[i][0] = values[i][0];  // u_x
            V[i][1] = values[i][1];  // u_y
            V[i][2] = values[i][2];  // u_z
            W[i][0] = values[i][3];  // v_x
            W[i][1] = values[i][4];  // v_y
            W[i][2] = values[i][5];  // v_z
        }

        // Solve for parallel vector puncture points with SoS perturbation.
        // Pass global vertex indices so the solver can apply index-based
        // symbolic perturbation, removing the need for user-chosen field offsets.
        std::vector<PuncturePoint> punctures;
        int n = solve_pv_triangle(V, W, punctures, s.vertices);

        // Handle degenerate case (entire triangle is PV surface)
        if (n == std::numeric_limits<int>::max()) {
            // Store a single point at barycenter for degenerate case
            el = FeatureElement();
            el.simplex = s;
            el.geometry_type = FeatureGeometryType::Point;
            el.barycentric_coords[0][0] = 1.0f / 3.0f;
            el.barycentric_coords[0][1] = 1.0f / 3.0f;
            el.barycentric_coords[0][2] = 1.0f / 3.0f;
            el.type = -1;  // Mark as degenerate
            el.scalar = 0.0f;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }

        // Return first puncture if any exist.
        // NOTE: Use extract_all() to obtain ALL punctures when n > 1.
        if (n > 0) {
            el = FeatureElement();
            el.simplex = s;
            el.geometry_type = FeatureGeometryType::Point;
            el.barycentric_coords[0][0] = (float)punctures[0].barycentric[0];
            el.barycentric_coords[0][1] = (float)punctures[0].barycentric[1];
            el.barycentric_coords[0][2] = (float)punctures[0].barycentric[2];
            el.type = 0;
            el.scalar = (float)punctures[0].lambda;  // Store lambda value
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            return true;
        }

        return false;
    }

    // Extract ALL parallel-vector punctures on this triangle (up to 3).
    // Unlike extract_it(), this method appends EVERY puncture found to `els`
    // rather than only the first. The SimplicialEngine uses this to ensure
    // no punctures are dropped when a triangle contains multiple ones.
    int extract_all(const Simplex& s, const T values[3][6],
                    std::vector<FeatureElement>& els) const
    {
        T V[3][3], W[3][3];
        for (int i = 0; i < 3; ++i) {
            V[i][0] = values[i][0]; V[i][1] = values[i][1]; V[i][2] = values[i][2];
            W[i][0] = values[i][3]; W[i][1] = values[i][4]; W[i][2] = values[i][5];
        }
        std::vector<PuncturePoint> punctures;
        int n = solve_pv_triangle(V, W, punctures, s.vertices);

        if (n == std::numeric_limits<int>::max()) {
            // Degenerate: entire triangle is PV surface — store single barycenter point
            FeatureElement el;
            el.simplex = s;
            el.geometry_type = FeatureGeometryType::Point;
            el.barycentric_coords[0][0] = el.barycentric_coords[0][1] = el.barycentric_coords[0][2] = 1.0f / 3.0f;
            el.type = -1;  // degenerate marker
            el.scalar = 0.0f;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            els.push_back(el);
            return 1;
        }

        for (int k = 0; k < n; ++k) {
            FeatureElement el;
            el.simplex = s;
            el.geometry_type = FeatureGeometryType::Point;
            el.barycentric_coords[0][0] = (float)punctures[k].barycentric[0];
            el.barycentric_coords[0][1] = (float)punctures[k].barycentric[1];
            el.barycentric_coords[0][2] = (float)punctures[k].barycentric[2];
            el.type = 0;
            el.scalar = (float)punctures[k].lambda;
            for (int i = 0; i < 16; ++i) el.attributes[i] = 0.0f;
            els.push_back(el);
        }
        return n;
    }

    // Extraction for tetrahedra (4 vertices, 2 vector fields)
    bool extract_tetrahedron(const Simplex& s, const T values[4][6],
                            PVCurveSegment& segment) const
    {
        // values layout: [4 vertices][6 components: ux, uy, uz, vx, vy, vz]
        T V[4][3], W[4][3];
        for (int i = 0; i < 4; ++i) {
            V[i][0] = values[i][0];  // u_x
            V[i][1] = values[i][1];  // u_y
            V[i][2] = values[i][2];  // u_z
            W[i][0] = values[i][3];  // v_x
            W[i][1] = values[i][4];  // v_y
            W[i][2] = values[i][5];  // v_z
        }

        // Solve for parametric PV curve
        bool found = solve_pv_tetrahedron(V, W, segment);
        if (found) {
            // Compute a hash for the simplex as an ID
            SimplexHash hasher;
            segment.simplex_id = (int)(hasher(s) % INT_MAX);
        }
        return found;
    }

    // Extract curves from all tetrahedra in a mesh
    void extract_curves_from_tets(const Mesh* mesh, const std::map<std::string, ftk::ndarray<T>>& data) {
        curve_segments.clear();

        if (!use_multicomponent || vector_var_name.empty()) {
            std::cerr << "ExactPV: extract_curves_from_tets requires multi-component mode" << std::endl;
            return;
        }

        const auto& vec_array = data.at(vector_var_name);
        auto reg_mesh = dynamic_cast<const RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset = reg_mesh ? reg_mesh->get_offset() : std::vector<uint64_t>{0,0,0,0};

        std::mutex mutex;
        std::atomic<uint64_t> visited(0), found(0);

        // Iterate over all 3-simplices (tetrahedra)
        const_cast<Mesh*>(mesh)->iterate_simplices(3, [&](const Simplex& s) {
            visited++;

            // Extract values at 4 vertices
            T values[4][6];
            for (int i = 0; i < 4; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                for (int j = 0; j < 6; ++j) {
                    if (coords.size() == 3) {
                        // 3D spatial: [6, nx, ny, nz]
                        values[i][j] = vec_array.f(j, coords[0], coords[1], coords[2]);
                    } else if (coords.size() == 4) {
                        // 3D spatial + time: [6, nx, ny, nz, nt]
                        values[i][j] = vec_array.f(j, coords[0], coords[1], coords[2], coords[3]);
                    }
                }
            }

            // Extract curve
            PVCurveSegment segment;
            if (extract_tetrahedron(s, values, segment)) {
                // Store tetrahedron vertex coordinates for physical position computation
                for (int i = 0; i < 4; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    for (int j = 0; j < 3 && j < coords.size(); ++j) {
                        segment.tet_vertices[i][j] = coords[j];
                    }
                }

                found++;
                std::lock_guard<std::mutex> lock(mutex);
                curve_segments.push_back(segment);
            }
        });

        std::cout << "Extracted " << found << " curves from " << visited << " tetrahedra" << std::endl;
    }

#ifdef __CUDACC__
    struct Device {
        static constexpr int codimension = 2;
        double sos_q;

        template <typename DeviceMesh>
        __device__
        bool extract_device(const Simplex& s, const CudaDataView<T>* data, int n_vars,
                          const DeviceMesh& mesh, FeatureElement& el) const {
            // TODO: Implement CUDA extraction
            return false;
        }
    };

    Device get_device() const { return {this->sos_q}; }
#endif
};

} // namespace ftk2
