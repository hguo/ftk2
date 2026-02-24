#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/complex.hpp>
#include <ftk2/core/sos.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <chrono>

#if FTK_HAVE_CUDA && defined(__CUDACC__)
#include <ftk2/core/cuda_engine.hpp>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif
#endif

namespace ftk2 {

/**
 * @brief Simple Union-Find for manifold stitching.
 */
template <typename IDType = uint64_t>
class UnionFind {
public:
    IDType add() {
        IDType id = parent_.size();
        parent_.push_back(id);
        return id;
    }

    IDType find(IDType i) {
        if (parent_[i] == i) return i;
        return parent_[i] = find(parent_[i]);
    }

    void unite(IDType i, IDType j) {
        IDType root_i = find(i);
        IDType root_j = find(j);
        if (root_i != root_j) {
            if (root_i < root_j) parent_[root_j] = root_i;
            else parent_[root_i] = root_j;
        }
    }

private:
    std::vector<IDType> parent_;
};

/**
 * @brief The Unified Simplicial Engine for FTK2.
 */
template <typename T, typename PredicateType, typename IDType = uint64_t>
class SimplicialEngine {
public:
    static constexpr int m = PredicateType::codimension;

    SimplicialEngine(std::shared_ptr<Mesh> mesh, PredicateType pred = PredicateType()) 
        : mesh_(mesh), predicate_(pred) {}

    void execute(const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int d_total = mesh_->get_total_dimension();

        std::vector<const ftk::ndarray<T>*> arrays;
        std::vector<std::string> resolved_vars = var_names;
        if (resolved_vars.empty()) {
            if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { resolved_vars = {predicate_.var_name}; }
            else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
                for(int i=0; i<m; ++i) resolved_vars.push_back(predicate_.var_names[i]);
                if (!predicate_.scalar_var_name.empty()) resolved_vars.push_back(predicate_.scalar_var_name);
            }
        }
        for (const auto& name : resolved_vars) arrays.push_back(&data.at(name));

        std::vector<FeatureElement> elements;
        mesh_->iterate_simplices(m, [&](const Simplex& s) {
            T values[m + 1][m];
            for (int i = 0; i <= m; ++i) {
                auto coords = mesh_->get_vertex_coordinates(s.vertices[i]);
                for (int j = 0; j < m; ++j) {
                    const auto& arr = *arrays[j];
                    if (coords.size() == 1) values[i][j] = arr.f(coords[0]);
                    else if (coords.size() == 2) values[i][j] = arr.f(coords[0], coords[1]);
                    else if (coords.size() == 3) values[i][j] = arr.f(coords[0], coords[1], coords[2]);
                    else if (coords.size() == 4) values[i][j] = arr.f(coords[0], coords[1], coords[2], coords[3]);
                }
            }
            FeatureElement el;
            if (predicate_.extract_it(s, values, el, arrays, mesh_.get())) elements.push_back(el);
        });
        
        std::sort(elements.begin(), elements.end(), [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });
        for (const auto& el : elements) {
            uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
            node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
        }
        auto t_nodes = std::chrono::high_resolution_clock::now();

        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(mesh_);
        if (reg_mesh) {
            uint64_t n_v = reg_mesh->get_num_vertices();
            for (uint64_t v_idx = 0; v_idx < n_v; ++v_idx) {
                auto local_coords = reg_mesh->get_vertex_coords_local(v_idx);
                if (reg_mesh->is_hypercube_base(local_coords)) {
                    uint64_t hc_idx = reg_mesh->hypercube_coords_to_idx(local_coords);
                    int n_p = 1; for(int i=1; i<=d_total; ++i) n_p *= i;
                    for (int p_idx = 0; p_idx < n_p; ++p_idx) {
                        Simplex cell; reg_mesh->get_d_simplex(hc_idx, p_idx, cell);
                        if (m == 1 && d_total == 4) marching_pentatope(cell, data, resolved_vars[0]);
                        else if (m == 1 && d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0]);
                        else form_general_manifold_patches(cell, data);
                    }
                }
            }
        } else {
            mesh_->iterate_simplices(d_total, [&](const Simplex& cell) {
                if (m == 1 && d_total == 4) marching_pentatope(cell, data, resolved_vars[0]);
                else if (m == 1 && d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0]);
                else form_general_manifold_patches(cell, data);
            });
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_nodes = std::chrono::duration_cast<std::chrono::milliseconds>(t_nodes - t_start).count();
        auto d_manifold = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_nodes).count();
        std::cout << "CPU Execution Breakdown: Nodes=" << d_nodes << "ms, Manifold=" << d_manifold << "ms, Total=" << (d_nodes+d_manifold) << "ms" << std::endl;
    }

#if FTK_HAVE_CUDA && defined(__CUDACC__)
    void execute_cuda(const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(mesh_);
        if (!reg_mesh) return;

        RegularSimplicialMeshDevice d_mesh; d_mesh.ndims = reg_mesh->get_total_dimension();
        auto l_dims = reg_mesh->get_local_dims(); auto off = reg_mesh->get_offset(); auto g_dims = reg_mesh->get_global_dims();
        for(int i=0; i<4; ++i) { 
            d_mesh.local_dims[i] = (i < d_mesh.ndims) ? l_dims[i] : 1; 
            d_mesh.offset[i] = (i < d_mesh.ndims) ? off[i] : 0; 
            d_mesh.global_dims[i] = (i < d_mesh.ndims) ? g_dims[i] : 1; 
        }

        auto t_setup = std::chrono::high_resolution_clock::now();
        std::vector<std::string> vars = var_names;
        if (vars.empty()) {
            if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { vars = {predicate_.var_name}; }
            else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
                for(int i=0; i<m; ++i) vars.push_back(predicate_.var_names[i]);
                if (!predicate_.scalar_var_name.empty()) vars.push_back(predicate_.scalar_var_name);
            }
        }

        std::vector<CudaDataView<T>> h_views;
        for (const auto& name : vars) {
            const auto& arr = data.at(name);
            CudaDataView<T> view;
            CUDA_CHECK(cudaMalloc((void**)&view.data, arr.nelem() * sizeof(T)));
            CUDA_CHECK(cudaMemcpy((void*)view.data, arr.pdata(), arr.nelem() * sizeof(T), cudaMemcpyHostToDevice));
            auto lattice = arr.get_lattice(); view.ndims = arr.nd();
            for(int i=0; i<4; ++i) { view.dims[i] = (i < arr.nd()) ? arr.dimf(i) : 1; view.s[i] = (i < arr.nd()) ? lattice.prod_[arr.nd() - 1 - i] : 0; }
            h_views.push_back(view);
        }

        CudaDataView<T>* d_views;
        CUDA_CHECK(cudaMalloc(&d_views, h_views.size() * sizeof(CudaDataView<T>)));
        CUDA_CHECK(cudaMemcpy(d_views, h_views.data(), h_views.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice));

        // Adaptive Buffer Execution Loop
        int max_nodes = 1000000; int max_conn = 3000000;
        bool buffer_overflow = true;
        CudaExtractionResult<IDType> res;
        auto t_h2d = std::chrono::high_resolution_clock::now();
        auto t_kernel = std::chrono::high_resolution_clock::now();

        while (buffer_overflow) {
            res.max_nodes = max_nodes; res.max_conn = max_conn;
            
            // Check VRAM availability
            size_t free_byte, total_byte;
            CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
            size_t required = (size_t)max_nodes * sizeof(FeatureElement) + (size_t)max_conn * sizeof(DeviceManifoldSimplex<IDType, 1>);
            if (required > free_byte * 0.8) {
                throw std::runtime_error("Insufficient VRAM for required feature extraction buffers. Required: " + std::to_string(required / (1024*1024)) + " MB");
            }

            CUDA_CHECK(cudaMalloc(&res.nodes, res.max_nodes * sizeof(FeatureElement)));
            CUDA_CHECK(cudaMemset(res.nodes, 0, res.max_nodes * sizeof(FeatureElement)));
            CUDA_CHECK(cudaMalloc(&res.node_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&res.edges, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 1>)));
            CUDA_CHECK(cudaMalloc(&res.edge_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&res.faces, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 2>)));
            CUDA_CHECK(cudaMalloc(&res.face_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&res.volumes, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 3>)));
            CUDA_CHECK(cudaMalloc(&res.volume_count, sizeof(int)));
            CUDA_CHECK(cudaMemset(res.node_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.edge_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.face_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.volume_count, 0, sizeof(int)));

            t_h2d = std::chrono::high_resolution_clock::now();
            uint64_t n_v = d_mesh.get_num_vertices();
            extraction_kernel<<< (n_v+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, h_views.size(), res);
            CUDA_CHECK(cudaDeviceSynchronize());
            t_kernel = std::chrono::high_resolution_clock::now();

            int h_node_count, h_edge_count, h_face_count, h_vol_count;
            CUDA_CHECK(cudaMemcpy(&h_node_count, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_edge_count, res.edge_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_face_count, res.face_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_vol_count, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost));

            int actual_max_conn = std::max({h_edge_count, h_face_count, h_vol_count});

            if (h_node_count > max_nodes || actual_max_conn > max_conn) {
                std::cout << "CUDA Warning: Buffer overflow detected. Retrying with required capacity..." << std::endl;
                max_nodes = h_node_count + 1000; max_conn = actual_max_conn + 1000;
                
                CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
                CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count));
                CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count));
                CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));
            } else {
                buffer_overflow = false;
            }
        }

        int h_node_count; CUDA_CHECK(cudaMemcpy(&h_node_count, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<FeatureElement> h_elements(h_node_count);
        CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes, h_elements.size() * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

        auto t_d2h = std::chrono::high_resolution_clock::now();
        std::sort(h_elements.begin(), h_elements.end(), [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });
        for (const auto& el : h_elements) {
            uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
            node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
        }

        int h_counts[4];
        CUDA_CHECK(cudaMemcpy(&h_counts[1], res.edge_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_counts[2], res.face_count, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_counts[3], res.volume_count, sizeof(int), cudaMemcpyDeviceToHost));

        auto t_d2h_manifold = std::chrono::high_resolution_clock::now();
        for (int dim = 1; dim <= 3; ++dim) {
            int count = h_counts[dim]; if (count <= 0) continue;
            if (dim == 1) {
                std::vector<DeviceManifoldSimplex<IDType, 1>> h_edges(count);
                CUDA_CHECK(cudaMemcpy(h_edges.data(), res.edges, h_edges.size()*sizeof(h_edges[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_edges) {
                    std::vector<IDType> nodes;
                    for (int i=0; i<=dim; ++i) { IDType nid; if (resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                    if (nodes.size() == dim + 1) { manifold_simplices_[dim].push_back(nodes); for(int i=1; i<=dim; ++i) uf_.unite(nodes[0], nodes[i]); }
                }
            } else if (dim == 2) {
                std::vector<DeviceManifoldSimplex<IDType, 2>> h_faces(count);
                CUDA_CHECK(cudaMemcpy(h_faces.data(), res.faces, h_faces.size()*sizeof(h_faces[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_faces) {
                    std::vector<IDType> nodes;
                    for (int i=0; i<=dim; ++i) { IDType nid; if (resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                    if (nodes.size() == dim + 1) { manifold_simplices_[dim].push_back(nodes); for(int i=1; i<=dim; ++i) uf_.unite(nodes[0], nodes[i]); }
                }
            } else if (dim == 3) {
                std::vector<DeviceManifoldSimplex<IDType, 3>> h_vols(count);
                CUDA_CHECK(cudaMemcpy(h_vols.data(), res.volumes, h_vols.size()*sizeof(h_vols[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_vols) {
                    std::vector<IDType> nodes;
                    for (int i=0; i<=dim; ++i) { IDType nid; if (resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                    if (nodes.size() == dim + 1) { manifold_simplices_[dim].push_back(nodes); for(int i=1; i<=dim; ++i) uf_.unite(nodes[0], nodes[i]); }
                }
            }
        }

        auto t_uf = std::chrono::high_resolution_clock::now();
        for (auto& v : h_views) CUDA_CHECK(cudaFree((void*)v.data));
        CUDA_CHECK(cudaFree(d_views)); CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count)); CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count)); CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count)); CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));

        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_setup = std::chrono::duration_cast<std::chrono::milliseconds>(t_setup - t_start).count();
        auto d_h2d = std::chrono::duration_cast<std::chrono::milliseconds>(t_h2d - t_setup).count();
        auto d_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(t_kernel - t_h2d).count();
        auto d_d2h = std::chrono::duration_cast<std::chrono::milliseconds>(t_d2h - t_kernel).count();
        auto d_d2h_m = std::chrono::duration_cast<std::chrono::milliseconds>(t_d2h_manifold - t_d2h).count();
        auto d_uf = std::chrono::duration_cast<std::chrono::milliseconds>(t_uf - t_d2h_manifold).count();
        auto d_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "CUDA Execution Breakdown: Setup=" << d_setup << "ms, H2D=" << d_h2d << "ms, Kernel=" << d_kernel << "ms, D2H(N)=" << d_d2h << "ms, D2H(M)=" << d_d2h_m << "ms, UF+Conn=" << d_uf << "ms, Total=" << d_total_ms << "ms" << std::endl;
    }
#endif

    FeatureComplex get_complex() {
        FeatureComplex complex;
        std::map<IDType, uint32_t> node_to_stable_idx;
        for (auto const& [s, node_id] : active_nodes_) {
            uint64_t root = uf_.find(node_id);
            FeatureElement el = node_elements_.at(node_id); el.track_id = (uint64_t)root;
            node_to_stable_idx[node_id] = complex.vertices.size(); complex.vertices.push_back(el);
        }
        for (int dim = 1; dim <= 3; ++dim) {
            if (!manifold_simplices_[dim].empty()) {
                FeatureComplex::SimplexIndices conn; conn.dimension = dim;
                for (auto const& simplex : manifold_simplices_[dim]) { for (IDType nid : simplex) conn.indices.push_back(node_to_stable_idx.at(nid)); }
                complex.connectivity.push_back(conn);
            }
        }
        complex.sort();
        return complex;
    }

private:
    void marching_pentatope(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var) {
        T vals[5]; uint64_t idx[5]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { threshold = predicate_.threshold; } else return;
        for (int i=0; i<5; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh_->get_vertex_coordinates(idx[i]);
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2], coords[3]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { Simplex edge = make_edge(idx[single], idx[o]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]); }
            if (nodes.size() == 4) { manifold_simplices_[3].push_back(nodes); for (int i=1; i<4; ++i) uf_.unite(nodes[0], nodes[i]); }
        } else if (A.size() == 2 || B.size() == 2) {
            const auto& two = (A.size() == 2) ? A : B; const auto& three = (A.size() == 2) ? B : A;
            std::vector<uint64_t> T0, T1;
            for (int t : three) {
                Simplex e0 = make_edge(idx[two[0]], idx[t]); Simplex e1 = make_edge(idx[two[1]], idx[t]);
                if (active_nodes_.count(e0)) T0.push_back(active_nodes_[e0]); if (active_nodes_.count(e1)) T1.push_back(active_nodes_[e1]);
            }
            if (T0.size() == 3 && T1.size() == 3) {
                manifold_simplices_[3].push_back({T0[0], T0[1], T0[2], T1[2]}); manifold_simplices_[3].push_back({T0[0], T0[1], T1[1], T1[2]}); manifold_simplices_[3].push_back({T0[0], T1[0], T1[1], T1[2]});
                for (int i=0; i<3; ++i) { uf_.unite(T0[0], T0[i]); uf_.unite(T0[0], T1[i]); }
            }
        }
    }

    void marching_tetrahedron(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var) {
        T vals[4]; uint64_t idx[4]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { threshold = predicate_.threshold; } else return;
        for (int i=0; i<4; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh_->get_vertex_coordinates(idx[i]);
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { Simplex edge = make_edge(idx[single], idx[o]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]); }
            if (nodes.size() == 3) { manifold_simplices_[2].push_back(nodes); for (int i=1; i<3; ++i) uf_.unite(nodes[0], nodes[i]); }
        } else if (A.size() == 2 && B.size() == 2) {
            std::vector<IDType> nodes;
            for (int a : A) for (int b : B) { Simplex edge = make_edge(idx[a], idx[b]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]); }
            if (nodes.size() == 4) { 
                manifold_simplices_[2].push_back({nodes[0], nodes[2], nodes[3]}); manifold_simplices_[2].push_back({nodes[0], nodes[3], nodes[1]}); 
                for (int i=1; i<4; ++i) uf_.unite(nodes[0], nodes[i]);
            }
        }
    }

    void form_general_manifold_patches(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data) {
        std::vector<IDType> nodes; find_m_subsimplices(cell, m, [&](const Simplex& f) { if (active_nodes_.count(f)) nodes.push_back(active_nodes_[f]); });
        if (nodes.size() >= 2) {
            std::sort(nodes.begin(), nodes.end());
            int k = mesh_->get_total_dimension() - m; for (size_t i = 1; i < nodes.size(); ++i) uf_.unite(nodes[0], nodes[i]);
            if (k == 1) { for (size_t i = 1; i < nodes.size(); ++i) manifold_simplices_[1].push_back({nodes[0], nodes[i]}); }
            else if (k == 2) { for (size_t i = 1; i < nodes.size() - 1; ++i) manifold_simplices_[2].push_back({nodes[0], nodes[i], nodes[i+1]}); }
        }
    }

    Simplex make_edge(uint64_t v0, uint64_t v1) { Simplex s; s.dimension = 1; s.vertices[0] = std::min(v0, v1); s.vertices[1] = std::max(v0, v1); return s; }

    void find_m_subsimplices(const Simplex& s, int target_m, std::function<void(const Simplex&)> callback) {
        int n = s.dimension + 1; int r = target_m + 1; std::vector<int> p(r); std::iota(p.begin(), p.end(), 0);
        while (p[0] <= n - r) {
            Simplex f; f.dimension = target_m; for (int i = 0; i < r; ++i) f.vertices[i] = s.vertices[p[i]];
            callback(f); int i = r - 1; while (i >= 0 && p[i] == n - r + i) i--; if (i < 0) break;
            p[i]++; for (int j = i + 1; j < r; j++) p[j] = p[i] + j - i;
        }
    }

    bool resolve_simplex_to_node(uint64_t encoded_id, const std::shared_ptr<RegularSimplicialMesh>& reg_mesh, int dim, IDType& nid) {
        uint64_t v0 = encoded_id >> 24; uint64_t combined_mask = encoded_id & 0xFFFFFF;
        auto c0 = reg_mesh->id_to_grid_index(v0);
        Simplex s; s.dimension = dim; s.vertices[0] = v0;
        for (int i = 1; i <= dim; ++i) {
            uint64_t mask = (combined_mask >> ((i - 1) * 4)) & 0xF;
            std::vector<uint64_t> ci = c0; for (int k = 0; k < 4; ++k) if ((mask >> k) & 1) ci[k]++;
            s.vertices[i] = reg_mesh->grid_index_to_id(ci);
        }
        s.sort_vertices();
        auto it = active_nodes_.find(s);
        if (it != active_nodes_.end()) { nid = it->second; return true; }
        return false;
    }

    std::shared_ptr<Mesh> mesh_; PredicateType predicate_; UnionFind<IDType> uf_;
    std::map<Simplex, IDType> active_nodes_;
    std::map<IDType, FeatureElement> node_elements_;
    std::map<IDType, Simplex> node_id_to_simplex_;
    std::map<int, std::vector<std::vector<IDType>>> manifold_simplices_;
};

} // namespace ftk2
