#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/complex.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/parallel.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <map>
#include <vector>
#include <set>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <functional>
#include <mutex>

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

    void clear() { parent_.clear(); }
    size_t size() const { return parent_.size(); }

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

    void clear_results() {
        active_nodes_.clear();
        node_elements_.clear();
        node_id_to_simplex_.clear();
        manifold_simplices_.clear();
        uf_.clear();
    }

    void execute(const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        clear_results();
        feed(mesh_, data, var_names);
    }

    void feed(std::shared_ptr<Mesh> slab_mesh, const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int d_total = slab_mesh->get_total_dimension();
        
        std::vector<std::string> resolved_vars = var_names;
        if (resolved_vars.empty()) {
            if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { resolved_vars = {predicate_.var_name}; }
            else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
                for(int i=0; i<m; ++i) resolved_vars.push_back(predicate_.var_names[i]);
                if (!predicate_.scalar_var_name.empty()) resolved_vars.push_back(predicate_.scalar_var_name);
            } else if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
                resolved_vars = {predicate_.var_names[0], predicate_.var_names[1]};
            }
        }

        // 1. Discover all active nodes
        std::vector<FeatureElement> elements;
        std::mutex mutex;
        slab_mesh->iterate_simplices(m, [&](const Simplex& s) {
            FeatureElement el;
            if (extract_simplex(s, data, el, slab_mesh.get())) {
                std::lock_guard<std::mutex> lock(mutex);
                elements.push_back(el);
            }
        });
        
        std::sort(elements.begin(), elements.end(), [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });
        for (const auto& el : elements) {
            if (active_nodes_.find(el.simplex) == active_nodes_.end()) {
                uint64_t node_id = uf_.add(); 
                active_nodes_[el.simplex] = node_id;
                node_elements_[node_id] = el; 
                node_id_to_simplex_[node_id] = el.simplex;
            }
        }
        auto t_nodes = std::chrono::high_resolution_clock::now();

        // 2. Perform manifold patching
        int n_threads = ftk2::get_num_threads();
        std::vector<ThreadData> tls(n_threads);
        
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(slab_mesh);
        if (reg_mesh) {
            uint64_t n_v = reg_mesh->get_num_vertices();
            ftk2::parallel_for(uint64_t(0), n_v, [&](uint64_t v_idx, int tid) {
                auto& local = tls[tid];
                auto local_coords = reg_mesh->get_vertex_coords_local(v_idx);
                if (reg_mesh->is_hypercube_base(local_coords)) {
                    uint64_t hc_idx = reg_mesh->hypercube_coords_to_idx(local_coords);
                    int n_p = 1; for(int i=1; i<=d_total; ++i) n_p *= i;
                    for (int p_idx = 0; p_idx < n_p; ++p_idx) {
                        Simplex cell; reg_mesh->get_d_simplex(hc_idx, p_idx, cell);
                        if constexpr (m == 1) {
                            if (d_total == 4) marching_pentatope(cell, data, resolved_vars[0], local, reg_mesh.get());
                            else if (d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0], local, reg_mesh.get());
                            else form_general_manifold_patches(cell, data, local, mutex, reg_mesh.get());
                        } else form_general_manifold_patches(cell, data, local, mutex, reg_mesh.get());
                    }
                }
            });
        } else {
            slab_mesh->iterate_simplices(d_total, [&](const Simplex& cell) {
                ThreadData local; 
                if constexpr (m == 1) {
                    if (d_total == 4) marching_pentatope(cell, data, resolved_vars[0], local, slab_mesh.get());
                    else if (d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0], local, slab_mesh.get());
                    else form_general_manifold_patches(cell, data, local, mutex, slab_mesh.get());
                } else form_general_manifold_patches(cell, data, local, mutex, slab_mesh.get());
                
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    manifold_simplices_[1].insert(manifold_simplices_[1].end(), local.dim1.begin(), local.dim1.end());
                    manifold_simplices_[2].insert(manifold_simplices_[2].end(), local.dim2.begin(), local.dim2.end());
                    manifold_simplices_[3].insert(manifold_simplices_[3].end(), local.dim3.begin(), local.dim3.end());
                }
            });
        }
        
        for (const auto& local : tls) {
            manifold_simplices_[1].insert(manifold_simplices_[1].end(), local.dim1.begin(), local.dim1.end());
            manifold_simplices_[2].insert(manifold_simplices_[2].end(), local.dim2.begin(), local.dim2.end());
            manifold_simplices_[3].insert(manifold_simplices_[3].end(), local.dim3.begin(), local.dim3.end());
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_nodes = std::chrono::duration_cast<std::chrono::milliseconds>(t_nodes - t_start).count();
        auto d_manifold = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_nodes).count();
        std::cout << "CPU Feed Breakdown: Nodes=" << d_nodes << "ms, Manifold=" << d_manifold << "ms, Total=" << (d_nodes+d_manifold) << "ms" << std::endl;
    }

    void execute_stream(ftk::stream<>& s, const std::vector<std::string>& var_names = {}) {
        int n_timesteps = s.total_timesteps();
        if (n_timesteps < 2) return;

        std::vector<std::string> resolved_vars = var_names;
        if (resolved_vars.empty()) {
            if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) resolved_vars = {predicate_.var_name};
            else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
                for(int i=0; i<m; ++i) resolved_vars.push_back(predicate_.var_names[i]);
                if (!predicate_.scalar_var_name.empty()) resolved_vars.push_back(predicate_.scalar_var_name);
            } else if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
                resolved_vars = {predicate_.var_names[0], predicate_.var_names[1]};
            }
        }

        clear_results();
        auto group_prev = s.read(0);
        for (int t = 0; t < n_timesteps - 1; ++t) {
            auto group_curr = s.read(t + 1);
            
            std::map<std::string, ftk::ndarray<T>> slab_data;
            std::vector<uint64_t> spatial_dims;
            bool dims_set = false;

            for (const auto& name : resolved_vars) {
                const auto& arr_prev = group_prev->template get_ref<T>(name);
                const auto& arr_curr = group_curr->template get_ref<T>(name);
                
                if (!dims_set) {
                    auto shape = arr_prev.shapef();
                    for (auto d : shape) spatial_dims.push_back(d);
                    dims_set = true;
                }

                std::vector<size_t> slab_shape = arr_prev.shapef();
                slab_shape.push_back(2);
                ftk::ndarray<T> arr_slab(slab_shape);
                size_t n_spatial = arr_prev.nelem();
                for (size_t i = 0; i < n_spatial; ++i) {
                    arr_slab[i] = arr_prev[i];
                    arr_slab[i + n_spatial] = arr_curr[i];
                }
                slab_data[name] = std::move(arr_slab);
            }

            std::vector<uint64_t> local_dims = spatial_dims; local_dims.push_back(2);
            std::vector<uint64_t> offset(spatial_dims.size(), 0); offset.push_back(t);
            std::vector<uint64_t> global_dims = spatial_dims; global_dims.push_back(n_timesteps);
            
            auto slab_mesh = std::make_shared<RegularSimplicialMesh>(local_dims, offset, global_dims);
            feed(slab_mesh, slab_data, resolved_vars);

            group_prev = group_curr;
        }
    }

#if FTK_HAVE_CUDA && defined(__CUDACC__)
    void execute_cuda(const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(mesh_);
        if (!reg_mesh) return;

        active_nodes_.clear();
        node_elements_.clear();
        node_id_to_simplex_.clear();
        manifold_simplices_.clear();
        uf_.clear();

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
            } else if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
                vars = {predicate_.var_names[0], predicate_.var_names[1]};
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

        int max_nodes = 1000000; int max_conn = 3000000;
        bool buffer_overflow = true;
        CudaExtractionResult<IDType> res;
        auto t_h2d = std::chrono::high_resolution_clock::now();
        auto t_kernel = std::chrono::high_resolution_clock::now();

        while (buffer_overflow) {
            res.max_nodes = max_nodes; res.max_conn = max_conn;
            CUDA_CHECK(cudaMalloc(&res.nodes, res.max_nodes * sizeof(FeatureElement)));
            CUDA_CHECK(cudaMemset(res.nodes, 0, res.max_nodes * sizeof(FeatureElement)));
            CUDA_CHECK(cudaMalloc(&res.node_count, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&res.edges, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 1>)));
            CUDA_CHECK(cudaMalloc(&res.faces, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 2>)));
            CUDA_CHECK(cudaMalloc(&res.volumes, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 3>)));
            CUDA_CHECK(cudaMalloc(&res.edge_count, sizeof(int))); CUDA_CHECK(cudaMalloc(&res.face_count, sizeof(int))); CUDA_CHECK(cudaMalloc(&res.volume_count, sizeof(int)));
            CUDA_CHECK(cudaMemset(res.node_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.edge_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.face_count, 0, sizeof(int))); CUDA_CHECK(cudaMemset(res.volume_count, 0, sizeof(int)));

            t_h2d = std::chrono::high_resolution_clock::now();
            uint64_t n_v = d_mesh.get_num_vertices();
            extraction_kernel<<< (n_v+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, h_views.size(), res);
            CUDA_CHECK(cudaDeviceSynchronize());
            t_kernel = std::chrono::high_resolution_clock::now();

            int h_n, h_e, h_f, h_v;
            CUDA_CHECK(cudaMemcpy(&h_n, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_e, res.edge_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_f, res.face_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_v, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_n > max_nodes || std::max({h_e, h_f, h_v}) > max_conn) {
                max_nodes = h_n + 1000; max_conn = std::max({h_e, h_f, h_v}) + 1000;
                CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
                CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count));
                CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count));
                CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));
            } else { buffer_overflow = false; }
        }

        int h_node_count; CUDA_CHECK(cudaMemcpy(&h_node_count, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<FeatureElement> h_elements(h_node_count);
        CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes, h_node_count * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

        auto t_d2h = std::chrono::high_resolution_clock::now();
        std::sort(h_elements.begin(), h_elements.end(), [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });
        for (const auto& el : h_elements) {
            uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
            node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
        }

        auto t_d2h_manifold = std::chrono::high_resolution_clock::now();
        for (int dim = 1; dim <= 3; ++dim) {
            int count = 0;
            if (dim == 1) { CUDA_CHECK(cudaMemcpy(&count, res.edge_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 1>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.edges, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }} else if (dim == 2) { CUDA_CHECK(cudaMemcpy(&count, res.face_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 2>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.faces, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }} else if (dim == 3) { CUDA_CHECK(cudaMemcpy(&count, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 3>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.volumes, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }}
        }

        auto t_uf = std::chrono::high_resolution_clock::now();
        for (auto& v : h_views) CUDA_CHECK(cudaFree((void*)v.data));
        CUDA_CHECK(cudaFree(d_views)); CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count)); CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count)); CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count)); CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));
        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "CUDA Execution Total=" << d_total_ms << "ms" << std::endl;
    }
#endif

    FeatureComplex get_complex() {
        FeatureComplex complex;
        for (int dim = 1; dim <= 3; ++dim) {
            for (auto& nodes : manifold_simplices_[dim]) std::sort(nodes.begin(), nodes.end());
            std::sort(manifold_simplices_[dim].begin(), manifold_simplices_[dim].end());
            manifold_simplices_[dim].erase(std::unique(manifold_simplices_[dim].begin(), manifold_simplices_[dim].end()), manifold_simplices_[dim].end());
        }

        uf_.clear();
        std::vector<Simplex> sorted_active_simplices;
        for (auto const& [s, _] : active_nodes_) sorted_active_simplices.push_back(s);
        std::sort(sorted_active_simplices.begin(), sorted_active_simplices.end());
        
        std::map<Simplex, IDType> canonical_active_nodes;
        for (const auto& s : sorted_active_simplices) canonical_active_nodes[s] = uf_.add();

        for (int dim = 1; dim <= 3; ++dim) {
            for (const auto& nodes : manifold_simplices_[dim]) {
                for (size_t i = 1; i < nodes.size(); ++i) uf_.unite(nodes[0], nodes[i]);
            }
        }

        std::map<uint64_t, Simplex> root_to_min_simplex;
        for (auto const& [s, original_id] : active_nodes_) {
            IDType canonical_id = canonical_active_nodes.at(s);
            uint64_t root = uf_.find(canonical_id);
            if (root_to_min_simplex.find(root) == root_to_min_simplex.end() || s < root_to_min_simplex[root]) root_to_min_simplex[root] = s;
        }

        std::map<IDType, uint32_t> node_to_stable_idx;
        for (const auto& s : sorted_active_simplices) {
            IDType original_id = active_nodes_.at(s);
            IDType canonical_id = canonical_active_nodes.at(s);
            uint64_t root = uf_.find(canonical_id);
            FeatureElement el = node_elements_.at(original_id); 
            el.track_id = root_to_min_simplex[root].vertices[0];
            node_to_stable_idx[original_id] = complex.vertices.size(); 
            complex.vertices.push_back(el);
        }
        
        for (int dim = 1; dim <= 3; ++dim) {
            if (!manifold_simplices_[dim].empty()) {
                FeatureComplex::SimplexIndices conn; conn.dimension = dim;
                for (auto const& nodes : manifold_simplices_[dim]) {
                    for (IDType nid : nodes) conn.indices.push_back(node_to_stable_idx.at(nid));
                }
                complex.connectivity.push_back(conn);
            }
        }
        complex.sort();
        return complex;
    }

private:
    bool extract_simplex(const Simplex& s, const std::map<std::string, ftk::ndarray<T>>& data, FeatureElement& el, Mesh* mesh) {
        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset;
        if (reg_mesh) offset = reg_mesh->get_offset();

        if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
            T values[3][2];
            for (int i = 0; i < 3; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                if (reg_mesh) for(size_t k=0; k<coords.size(); ++k) coords[k] -= offset[k];
                for (int j = 0; j < 2; ++j) {
                    const auto& arr = data.at(predicate_.var_names[j]);
                    if (coords.size() == 2) values[i][j] = arr.f(coords[0], coords[1]);
                    else if (coords.size() == 3) values[i][j] = arr.f(coords[0], coords[1], coords[2]);
                    else if (coords.size() == 4) values[i][j] = arr.f(coords[0], coords[1], coords[2], coords[3]);
                }
            }
            return predicate_.extract_it(s, values, el);
        } else if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) {
            T values[2][1];
            for (int i = 0; i < 2; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                if (reg_mesh) for(size_t k=0; k<coords.size(); ++k) coords[k] -= offset[k];
                const auto& arr = data.at(predicate_.var_name);
                if (coords.size() == 2) values[i][0] = arr.f(coords[0], coords[1]);
                else if (coords.size() == 3) values[i][0] = arr.f(coords[0], coords[1], coords[2]);
                else if (coords.size() == 4) values[i][0] = arr.f(coords[0], coords[1], coords[2], coords[3]);
            }
            return predicate_.extract_it(s, values, el);
        } else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
            T values[m+1][m];
            for (int i = 0; i <= m; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                if (reg_mesh) for(size_t k=0; k<coords.size(); ++k) coords[k] -= offset[k];
                for (int j = 0; j < m; ++j) {
                    const auto& arr = data.at(predicate_.var_names[j]);
                    if (coords.size() == 2) values[i][j] = arr.f(coords[0], coords[1]);
                    else if (coords.size() == 3) values[i][j] = arr.f(coords[0], coords[1], coords[2]);
                    else if (coords.size() == 4) values[i][j] = arr.f(coords[0], coords[1], coords[2], coords[3]);
                }
            }
            std::vector<const ftk::ndarray<T>*> arrays_ptrs;
            for(int k=0; k<m; ++k) arrays_ptrs.push_back(&data.at(predicate_.var_names[k]));
            if (!predicate_.scalar_var_name.empty()) arrays_ptrs.push_back(&data.at(predicate_.scalar_var_name));
            
            return predicate_.extract_it(s, values, el, arrays_ptrs, mesh);
        }
        return false;
    }

    struct ThreadData {
        std::vector<std::vector<IDType>> dim1, dim2, dim3;
    };

    void marching_pentatope(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var, ThreadData& local, Mesh* mesh) {
        T vals[5]; uint64_t idx[5]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) threshold = predicate_.thresholds[0]; 
        else if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) threshold = predicate_.threshold;
        else return;

        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset;
        if (reg_mesh) offset = reg_mesh->get_offset();

        for (int i=0; i<5; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh->get_vertex_coordinates(idx[i]);
            if (reg_mesh) for(size_t k=0; k<coords.size(); ++k) coords[k] -= offset[k];
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2], coords[3]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { Simplex edge = make_edge(idx[single], idx[o]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); }
            if (nodes.size() == 4) { std::sort(nodes.begin(), nodes.end()); local.dim3.push_back(nodes); }
        } else if (A.size() == 2 || B.size() == 2) {
            const auto& two = (A.size() == 2) ? A : B; const auto& three = (A.size() == 2) ? B : A;
            std::vector<uint64_t> T0, T1;
            for (int t : three) {
                Simplex e0 = make_edge(idx[two[0]], idx[t]); Simplex e1 = make_edge(idx[two[1]], idx[t]);
                if (active_nodes_.count(e0)) T0.push_back(active_nodes_.at(e0)); if (active_nodes_.count(e1)) T1.push_back(active_nodes_.at(e1));
            }
            if (T0.size() == 3 && T1.size() == 3) {
                std::vector<IDType> c0 = {T0[0], T0[1], T0[2], T1[2]}; std::sort(c0.begin(), c0.end());
                std::vector<IDType> c1 = {T0[0], T0[1], T1[1], T1[2]}; std::sort(c1.begin(), c1.end());
                std::vector<IDType> c2 = {T0[0], T1[0], T1[1], T1[2]}; std::sort(c2.begin(), c2.end());
                local.dim3.push_back(c0); local.dim3.push_back(c1); local.dim3.push_back(c2);
            }
        }
    }

    void marching_tetrahedron(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var, ThreadData& local, Mesh* mesh) {
        T vals[4]; uint64_t idx[4]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { threshold = predicate_.threshold; } else return;
        
        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset;
        if (reg_mesh) offset = reg_mesh->get_offset();

        for (int i=0; i<4; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh->get_vertex_coordinates(idx[i]);
            if (reg_mesh) for(size_t k=0; k<coords.size(); ++k) coords[k] -= offset[k];
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { Simplex edge = make_edge(idx[single], idx[o]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); }
            if (nodes.size() == 3) { std::sort(nodes.begin(), nodes.end()); local.dim2.push_back(nodes); }
        } else if (A.size() == 2 && B.size() == 2) {
            std::vector<IDType> nodes;
            for (int a : A) for (int b : B) { Simplex edge = make_edge(idx[a], idx[b]); if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); }
            if (nodes.size() == 4) { 
                std::sort(nodes.begin(), nodes.end());
                local.dim2.push_back({nodes[0], nodes[1], nodes[2]}); local.dim2.push_back({nodes[0], nodes[2], nodes[3]}); 
            }
        }
    }

    void form_general_manifold_patches(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, ThreadData& local, std::mutex& mutex, Mesh* mesh) {
        int d = cell.dimension; int k = d - m; if (k <= 0) return;
        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        struct NodeInfo { IDType id; uint64_t code; Simplex s; };
        std::vector<NodeInfo> nodes;
        find_m_subsimplices(cell, m, [&](const Simplex& f_orig) { 
            Simplex f = f_orig; f.sort_vertices();
            IDType node_id;
            bool found = false;
            
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (active_nodes_.count(f)) {
                    node_id = active_nodes_.at(f);
                    found = true;
                } else {
                    FeatureElement el;
                    if (extract_simplex(f, data, el, mesh)) {
                        node_id = uf_.add();
                        active_nodes_[f] = node_id;
                        node_elements_[node_id] = el;
                        node_id_to_simplex_[node_id] = f;
                        found = true;
                    }
                }
            }

            if (found) {
                uint64_t code = reg_mesh ? encode_simplex_id(f, *reg_mesh) : 0;
                nodes.push_back({node_id, code, f});
            }
        });
        if (nodes.empty()) return;
        
        if (reg_mesh) {
            std::sort(nodes.begin(), nodes.end(), [](const NodeInfo& a, const NodeInfo& b) { return a.code < b.code; });
        } else {
            std::sort(nodes.begin(), nodes.end(), [](const NodeInfo& a, const NodeInfo& b) { return a.id < b.id; });
        }
        
        IDType h_S = nodes[0].id;

        if (k == 1) {
            for (size_t i = 1; i < nodes.size(); ++i) {
                std::vector<IDType> edge = {h_S, nodes[i].id}; std::sort(edge.begin(), edge.end()); local.dim1.push_back(edge);
            }
        } else if (k == 2) {
            for (int i = 0; i <= d; ++i) {
                int tet_mask = ((1 << (d + 1)) - 1) ^ (1 << i);
                std::vector<NodeInfo> nodes_T;
                for (const auto& n : nodes) {
                    bool on_face = true;
                    for (int j = 0; j <= m; ++j) {
                        bool found = false;
                        for (int l = 0; l <= d; ++l) {
                            if ((tet_mask & (1 << l)) && n.s.vertices[j] == cell.vertices[l]) { found = true; break; }
                        }
                        if (!found) { on_face = false; break; }
                    }
                    if (on_face) nodes_T.push_back(n);
                }
                
                if (nodes_T.size() < 2) continue;
                IDType h_T = nodes_T[0].id;
                
                for (size_t j = 1; j < nodes_T.size(); ++j) {
                    IDType n_id = nodes_T[j].id;
                    if (h_S != h_T && h_S != n_id) {
                        std::vector<IDType> tri = {h_S, h_T, n_id};
                        std::sort(tri.begin(), tri.end());
                        local.dim2.push_back(tri);
                    }
                }
            }
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
        uint64_t v_min = encoded_id >> 24; uint64_t combined_mask = encoded_id & 0xFFFFFF;
        auto c_min = reg_mesh->id_to_grid_index(v_min);
        Simplex s; s.dimension = dim; s.vertices[0] = v_min;
        for (int i = 1; i <= dim; ++i) {
            uint64_t mask = (combined_mask >> ((i - 1) * 4)) & 0xF;
            std::vector<uint64_t> ci = c_min; for (int k = 0; k < 4; ++k) if ((mask >> k) & 1) ci[k]++;
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
