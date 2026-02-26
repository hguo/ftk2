#pragma once

#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>
#include <ftk2/core/feature.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/core/complex.hpp>
#include <ftk2/core/sos.hpp>
#include <ftk2/core/parallel.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/ndarray_stream.hh>
#include <ndarray/device.hh>
#include <map>
#include <vector>
#include <set>
#include <queue>
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
        std::atomic<uint64_t> visited(0), found(0);
        slab_mesh->iterate_simplices(m, [&](const Simplex& s) {
            visited++;
            FeatureElement el;
            if (extract_simplex(s, data, el, slab_mesh.get())) {
                found++;
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
                            if (d_total == 4) marching_pentatope(cell, data, resolved_vars[0], local, mutex, reg_mesh.get());
                            else if (d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0], local, mutex, reg_mesh.get());
                            else form_general_manifold_patches(cell, data, local, mutex, reg_mesh.get());
                        } else form_general_manifold_patches(cell, data, local, mutex, reg_mesh.get());
                    }
                }
            });
        } else {
            slab_mesh->iterate_simplices(d_total, [&](const Simplex& cell) {
                ThreadData local; 
                if constexpr (m == 1) {
                    if (d_total == 4) marching_pentatope(cell, data, resolved_vars[0], local, mutex, slab_mesh.get());
                    else if (d_total == 3) marching_tetrahedron(cell, data, resolved_vars[0], local, mutex, slab_mesh.get());
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
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(mesh_);
        if (reg_mesh) { execute_cuda_regular(reg_mesh, data, var_names); return; }

        auto unst_mesh = std::dynamic_pointer_cast<UnstructuredSimplicialMesh>(mesh_);
        if (unst_mesh) { execute_cuda_unstructured(unst_mesh, data, var_names); return; }

        auto ext_mesh = std::dynamic_pointer_cast<ExtrudedSimplicialMesh>(mesh_);
        if (ext_mesh) { execute_cuda_extruded(ext_mesh, data, var_names); return; }
    }

    /**
     * @brief Streaming CUDA execution - holds only 2 consecutive timesteps in GPU memory
     * @param stream The ndarray stream to read from
     * @param spatial_mesh The spatial mesh (without time dimension)
     * @param var_names Variables to track (empty = auto-detect from predicate)
     */
    void execute_cuda_streaming(ftk::stream<ftk::native_storage>& stream,
                                std::shared_ptr<Mesh> spatial_mesh,
                                const std::vector<std::string>& var_names = {}) {
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(spatial_mesh);
        if (reg_mesh) {
            execute_cuda_streaming_regular(stream, reg_mesh, var_names);
            return;
        }

        throw std::runtime_error("Streaming CUDA execution only supports regular meshes currently");
    }

    void execute_cuda_regular(std::shared_ptr<RegularSimplicialMesh> reg_mesh, const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        clear_results();

        RegularSimplicialMeshDevice d_mesh; d_mesh.ndims = reg_mesh->get_total_dimension();
        auto l_dims = reg_mesh->get_local_dims(); auto off = reg_mesh->get_offset(); auto g_dims = reg_mesh->get_global_dims();
        for(int i=0; i<4; ++i) { 
            d_mesh.local_dims[i] = (i < d_mesh.ndims) ? l_dims[i] : 1; 
            d_mesh.offset[i] = (i < d_mesh.ndims) ? off[i] : 0; 
            d_mesh.global_dims[i] = (i < d_mesh.ndims) ? g_dims[i] : 1; 
        }

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

        // Upload arrays to device using ndarray's built-in CUDA support
        std::vector<ftk::ndarray<T>> device_arrays;
        std::vector<CudaDataView<T>> h_views;

        for (const auto& name : vars) {
            // Make a copy and upload to device
            ftk::ndarray<T> arr = data.at(name);
            arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

            CudaDataView<T> view;
            view.data = static_cast<T*>(arr.get_devptr());
            auto lattice = arr.get_lattice(); view.ndims = arr.nd();
            for(int i=0; i<4; ++i) {
                view.dims[i] = (i < arr.nd()) ? arr.dimf(i) : 1;
                view.s[i] = (i < arr.nd()) ? lattice.prod_[arr.nd() - 1 - i] : 0;
            }
            h_views.push_back(view);
            device_arrays.push_back(std::move(arr));  // Keep array alive
        }

        CudaDataView<T>* d_views;
        CUDA_CHECK(cudaMalloc(&d_views, h_views.size() * sizeof(CudaDataView<T>)));
        CUDA_CHECK(cudaMemcpy(d_views, h_views.data(), h_views.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice));

        int max_nodes = 1000000; int max_conn = 3000000;
        bool buffer_overflow = true;
        CudaExtractionResult<IDType> res;

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

            uint64_t n_v = d_mesh.get_num_vertices();
            extraction_kernel<<< (n_v+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, h_views.size(), res);
            CUDA_CHECK(cudaDeviceSynchronize());

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
        if (h_node_count > 0) CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes, h_node_count * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

        std::sort(h_elements.begin(), h_elements.end(), [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });
        for (const auto& el : h_elements) {
            if (active_nodes_.find(el.simplex) == active_nodes_.end()) {
                uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
                node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
            }
        }

        for (int dim = 1; dim <= 3; ++dim) {
            int count = 0;
            if (dim == 1) { CUDA_CHECK(cudaMemcpy(&count, res.edge_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 1>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.edges, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node_regular(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }} else if (dim == 2) { CUDA_CHECK(cudaMemcpy(&count, res.face_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 2>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.faces, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node_regular(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }} else if (dim == 3) { CUDA_CHECK(cudaMemcpy(&count, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost)); if (count > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 3>> h_s(count); CUDA_CHECK(cudaMemcpy(h_s.data(), res.volumes, count*sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) { std::vector<IDType> nodes; for(int i=0; i<=dim; ++i) { IDType nid; if(resolve_simplex_to_node_regular(s.nodes[i], reg_mesh, m, nid)) nodes.push_back(nid); }
                if(nodes.size()==dim+1) { std::sort(nodes.begin(), nodes.end()); manifold_simplices_[dim].push_back(nodes); } }
            }}
        }

        // device_arrays will automatically clean up device memory via ndarray's RAII
        CUDA_CHECK(cudaFree(d_views));
        CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
        CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count));
        CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count));
        CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));
        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "CUDA Execution Total=" << d_total_ms << "ms" << std::endl;
    }

    void execute_cuda_unstructured(std::shared_ptr<UnstructuredSimplicialMesh> unst_mesh, const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        clear_results();

        UnstructuredSimplicialMeshDevice d_mesh;
        d_mesh.spatial_dim = unst_mesh->get_spatial_dimension();
        d_mesh.cell_dim = unst_mesh->get_total_dimension();
        
        for (int k = 0; k <= d_mesh.cell_dim; ++k) {
            std::atomic<uint64_t> n_s(0); unst_mesh->iterate_simplices(k, [&](const Simplex& s) { n_s++; });
            d_mesh.n_simplices[k] = n_s.load();
            std::vector<Simplex> h_simplices(d_mesh.n_simplices[k]);
            std::atomic<uint64_t> counter(0);
            unst_mesh->iterate_simplices(k, [&](const Simplex& s) { h_simplices[counter.fetch_add(1)] = s; });
            CUDA_CHECK(cudaMalloc((void**)&d_mesh.simplices[k], d_mesh.n_simplices[k] * sizeof(Simplex)));
            CUDA_CHECK(cudaMemcpy((void*)d_mesh.simplices[k], h_simplices.data(), d_mesh.n_simplices[k] * sizeof(Simplex), cudaMemcpyHostToDevice));
        }

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

        // Upload arrays to device using ndarray's built-in CUDA support
        std::vector<ftk::ndarray<T>> device_arrays;
        std::vector<CudaDataView<T>> h_views;

        for (const auto& name : vars) {
            ftk::ndarray<T> arr = data.at(name);
            arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

            CudaDataView<T> view;
            view.data = static_cast<T*>(arr.get_devptr());
            view.ndims = 1; view.dims[0] = arr.nelem(); view.s[0] = 1;
            h_views.push_back(view);
            device_arrays.push_back(std::move(arr));
        }

        CudaDataView<T>* d_views;
        CUDA_CHECK(cudaMalloc(&d_views, h_views.size() * sizeof(CudaDataView<T>)));
        CUDA_CHECK(cudaMemcpy(d_views, h_views.data(), h_views.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice));

        int max_nodes = 1000000;
        CudaExtractionResult<IDType> res;
        CUDA_CHECK(cudaMalloc(&res.nodes, max_nodes * sizeof(FeatureElement)));
        CUDA_CHECK(cudaMalloc(&res.node_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(res.node_count, 0, sizeof(int)));
        res.max_nodes = max_nodes;

        uint64_t n_total = std::max(d_mesh.n_simplices[m], d_mesh.n_simplices[d_mesh.cell_dim]);
        extraction_kernel_unstructured<<< (n_total+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, h_views.size(), res);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_n; CUDA_CHECK(cudaMemcpy(&h_n, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<FeatureElement> h_elements(h_n);
        if (h_n > 0) CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes, h_n * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

        for (const auto& el : h_elements) {
            if (active_nodes_.find(el.simplex) == active_nodes_.end()) {
                uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
                node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
            }
        }

        // device_arrays will automatically clean up device memory via ndarray's RAII
        CUDA_CHECK(cudaFree(d_views)); CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
        for (int k = 0; k <= d_mesh.cell_dim; ++k) CUDA_CHECK(cudaFree((void*)d_mesh.simplices[k]));
        
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA Unstructured Total=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms" << std::endl;
    }

    void execute_cuda_extruded(std::shared_ptr<ExtrudedSimplicialMesh> ext_mesh, const std::map<std::string, ftk::ndarray<T>>& data, const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        clear_results();

        auto unst_base = std::dynamic_pointer_cast<UnstructuredSimplicialMesh>(ext_mesh->get_base_mesh());
        if (!unst_base) { std::cerr << "CUDA Extruded requires Unstructured base mesh." << std::endl; return; }

        ExtrudedSimplicialMeshDevice d_mesh;
        d_mesh.n_layers = ext_mesh->get_n_layers();
        d_mesh.n_spatial_verts = ext_mesh->get_n_spatial_verts();
        d_mesh.base_mesh.spatial_dim = unst_base->get_spatial_dimension();
        d_mesh.base_mesh.cell_dim = unst_base->get_total_dimension();
        
        std::set<int> needed_dims = {0, m, m-1};
        for (int k : needed_dims) {
            if (k < 0 || k > d_mesh.base_mesh.cell_dim) continue;
            std::atomic<uint64_t> n_s(0); unst_base->iterate_simplices(k, [&](const Simplex& s) { n_s++; });
            d_mesh.base_mesh.n_simplices[k] = n_s.load();
            std::vector<Simplex> h_simplices(d_mesh.base_mesh.n_simplices[k]);
            std::atomic<uint64_t> counter(0);
            unst_base->iterate_simplices(k, [&](const Simplex& s) { h_simplices[counter.fetch_add(1)] = s; });
            CUDA_CHECK(cudaMalloc((void**)&d_mesh.base_mesh.simplices[k], d_mesh.base_mesh.n_simplices[k] * sizeof(Simplex)));
            CUDA_CHECK(cudaMemcpy((void*)d_mesh.base_mesh.simplices[k], h_simplices.data(), d_mesh.base_mesh.n_simplices[k] * sizeof(Simplex), cudaMemcpyHostToDevice));
        }

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

        // Upload arrays to device using ndarray's built-in CUDA support
        std::vector<ftk::ndarray<T>> device_arrays;
        std::vector<CudaDataView<T>> h_views;

        for (const auto& name : vars) {
            ftk::ndarray<T> arr = data.at(name);
            arr.copy_to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

            CudaDataView<T> view;
            view.data = static_cast<T*>(arr.get_devptr());
            view.ndims = 1; view.dims[0] = arr.nelem(); view.s[0] = 1;
            h_views.push_back(view);
            device_arrays.push_back(std::move(arr));
        }

        CudaDataView<T>* d_views;
        CUDA_CHECK(cudaMalloc(&d_views, h_views.size() * sizeof(CudaDataView<T>)));
        CUDA_CHECK(cudaMemcpy(d_views, h_views.data(), h_views.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice));

        int max_nodes = 2000000;
        CudaExtractionResult<IDType> res;
        CUDA_CHECK(cudaMalloc(&res.nodes, max_nodes * sizeof(FeatureElement)));
        CUDA_CHECK(cudaMalloc(&res.node_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(res.node_count, 0, sizeof(int)));
        res.max_nodes = max_nodes;

        uint64_t n_spatial_m = d_mesh.base_mesh.n_simplices[m];
        uint64_t n_spatial_m_minus_1 = (m > 0) ? d_mesh.base_mesh.n_simplices[m-1] : 0;
        uint64_t n_total = n_spatial_m * (d_mesh.n_layers + 1) + n_spatial_m_minus_1 * d_mesh.n_layers * m;

        extraction_kernel_extruded<<< (n_total+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, h_views.size(), res);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_n; CUDA_CHECK(cudaMemcpy(&h_n, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<FeatureElement> h_elements(h_n);
        if (h_n > 0) CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes, h_n * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

        for (const auto& el : h_elements) {
            if (active_nodes_.find(el.simplex) == active_nodes_.end()) {
                uint64_t node_id = uf_.add(); active_nodes_[el.simplex] = node_id;
                node_elements_[node_id] = el; node_id_to_simplex_[node_id] = el.simplex;
            }
        }

        // device_arrays will automatically clean up device memory via ndarray's RAII
        CUDA_CHECK(cudaFree(d_views)); CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
        for (int k : needed_dims) if (k >= 0 && k <= d_mesh.base_mesh.cell_dim) CUDA_CHECK(cudaFree((void*)d_mesh.base_mesh.simplices[k]));
        
        auto t_end = std::chrono::high_resolution_clock::now();
        std::cout << "CUDA Extruded Total=" << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms" << std::endl;
    }

    /**
     * @brief Streaming CUDA execution for regular meshes
     * Only holds 2 consecutive timesteps in GPU memory, swaps buffers between iterations
     */
    void execute_cuda_streaming_regular(ftk::stream<ftk::native_storage>& stream,
                                       std::shared_ptr<RegularSimplicialMesh> spatial_mesh,
                                       const std::vector<std::string>& var_names = {}) {
        auto t_start = std::chrono::high_resolution_clock::now();
        clear_results();

        int n_timesteps = stream.total_timesteps();
        std::cout << "CUDA Streaming: Processing " << n_timesteps << " timesteps (2 in memory)" << std::endl;

        // Determine variables to track
        std::vector<std::string> vars = var_names;
        if (vars.empty()) {
            if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) {
                vars = {predicate_.var_name};
            } else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
                if (predicate_.use_multicomponent && !predicate_.vector_var_name.empty()) {
                    vars = {predicate_.vector_var_name};
                } else {
                    for(int i=0; i<m; ++i) vars.push_back(predicate_.var_names[i]);
                    if (!predicate_.scalar_var_name.empty()) vars.push_back(predicate_.scalar_var_name);
                }
            } else if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
                vars = {predicate_.var_names[0], predicate_.var_names[1]};
            }
        }

        // Get spatial dimensions from first timestep
        auto group_0 = stream.read(0);
        const auto& first_arr = group_0->template get_ref<T>(vars[0]);
        std::vector<uint64_t> spatial_dims;

        // Handle multi-component arrays: skip first dimension if it's component count
        int start_dim = 0;
        if (first_arr.nd() >= 2 && first_arr.dimf(0) >= 1 && first_arr.dimf(0) <= 16) {
            start_dim = 1;  // Skip component dimension
        }
        for (int d = start_dim; d < first_arr.nd(); ++d) {
            spatial_dims.push_back(first_arr.dimf(d));
        }

        // Allocate persistent device buffers for 2 timesteps (slab)
        // These buffers will be reused across all timestep pairs
        std::vector<ftk::ndarray<T>> device_buffer_t0, device_buffer_t1;
        std::vector<CudaDataView<T>> h_views_t0, h_views_t1;

        for (const auto& var : vars) {
            const auto& ref_arr = group_0->template get_ref<T>(var);

            // Allocate device buffers with spatial shape (no time dimension)
            ftk::ndarray<T> d_buf0, d_buf1;
            std::vector<size_t> shape;
            for (size_t d = 0; d < ref_arr.nd(); ++d) shape.push_back(ref_arr.dimf(d));
            d_buf0.reshapef(shape);
            d_buf1.reshapef(shape);

            // Allocate on device
            d_buf0.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);
            d_buf1.to_device(ftk::NDARRAY_DEVICE_CUDA, 0);

            // Create views
            CudaDataView<T> view0, view1;
            view0.data = static_cast<T*>(d_buf0.get_devptr());
            view1.data = static_cast<T*>(d_buf1.get_devptr());

            auto lattice = ref_arr.get_lattice();
            for (int i = 0; i < 2; ++i) {
                auto& view = (i == 0) ? view0 : view1;
                view.ndims = ref_arr.nd();
                for(int j=0; j<4; ++j) {
                    view.dims[j] = (j < ref_arr.nd()) ? ref_arr.dimf(j) : 1;
                    view.s[j] = (j < ref_arr.nd()) ? lattice.prod_[ref_arr.nd() - 1 - j] : 0;
                }
            }

            device_buffer_t0.push_back(std::move(d_buf0));
            device_buffer_t1.push_back(std::move(d_buf1));
            h_views_t0.push_back(view0);
            h_views_t1.push_back(view1);
        }

        // Allocate slab views array (combines t0 and t1)
        std::vector<CudaDataView<T>> h_views_slab;
        for (size_t i = 0; i < vars.size(); ++i) {
            h_views_slab.push_back(h_views_t0[i]);
            h_views_slab.push_back(h_views_t1[i]);
        }

        CudaDataView<T>* d_views;
        CUDA_CHECK(cudaMalloc(&d_views, h_views_slab.size() * sizeof(CudaDataView<T>)));

        // Setup device mesh for slab (2 timesteps)
        RegularSimplicialMeshDevice d_mesh;
        d_mesh.ndims = spatial_dims.size() + 1;  // spatial + time
        std::vector<uint64_t> slab_dims = spatial_dims;
        slab_dims.push_back(2);  // 2 timesteps
        for(int i=0; i<4; ++i) {
            d_mesh.local_dims[i] = (i < d_mesh.ndims) ? slab_dims[i] : 1;
            d_mesh.offset[i] = 0;  // Will update per slab
            d_mesh.global_dims[i] = (i < d_mesh.ndims) ? (i == d_mesh.ndims - 1 ? n_timesteps : slab_dims[i]) : 1;
        }

        // Allocate extraction result buffers
        int max_nodes = 1000000, max_conn = 3000000;
        CudaExtractionResult<IDType> res;
        res.max_nodes = max_nodes; res.max_conn = max_conn;
        CUDA_CHECK(cudaMalloc(&res.nodes, res.max_nodes * sizeof(FeatureElement)));
        CUDA_CHECK(cudaMemset(res.nodes, 0, res.max_nodes * sizeof(FeatureElement)));
        CUDA_CHECK(cudaMalloc(&res.node_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&res.edges, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 1>)));
        CUDA_CHECK(cudaMalloc(&res.faces, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 2>)));
        CUDA_CHECK(cudaMalloc(&res.volumes, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 3>)));
        CUDA_CHECK(cudaMalloc(&res.edge_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&res.face_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&res.volume_count, sizeof(int)));

        // Stream through timestep pairs
        for (int t = 0; t < n_timesteps - 1; ++t) {
            auto t_slab_start = std::chrono::high_resolution_clock::now();

            // Read timestep pair from stream
            auto group_t0 = stream.read(t);
            auto group_t1 = stream.read(t + 1);

            // Upload data to device buffers (reusing allocated memory)
            for (size_t i = 0; i < vars.size(); ++i) {
                const auto& arr_t0 = group_t0->template get_ref<T>(vars[i]);
                const auto& arr_t1 = group_t1->template get_ref<T>(vars[i]);

                CUDA_CHECK(cudaMemcpy(device_buffer_t0[i].get_devptr(), arr_t0.pdata(),
                                     arr_t0.nelem() * sizeof(T), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(device_buffer_t1[i].get_devptr(), arr_t1.pdata(),
                                     arr_t1.nelem() * sizeof(T), cudaMemcpyHostToDevice));
            }

            // Update device views and mesh offset for current slab
            CUDA_CHECK(cudaMemcpy(d_views, h_views_slab.data(),
                                 h_views_slab.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice));

            d_mesh.offset[d_mesh.ndims - 1] = t;  // Time offset

            // Reset result counters
            CUDA_CHECK(cudaMemset(res.node_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(res.edge_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(res.face_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(res.volume_count, 0, sizeof(int)));

            // Launch extraction kernel
            uint64_t n_v = d_mesh.get_num_vertices();
            extraction_kernel<<< (n_v+255)/256, 256 >>>(d_mesh, predicate_.get_device(), d_views, vars.size(), res);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy results back to host
            int h_node_count;
            CUDA_CHECK(cudaMemcpy(&h_node_count, res.node_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_node_count > 0) {
                std::vector<FeatureElement> h_elements(h_node_count);
                CUDA_CHECK(cudaMemcpy(h_elements.data(), res.nodes,
                                     h_node_count * sizeof(FeatureElement), cudaMemcpyDeviceToHost));

                // Insert into tracking structure
                std::sort(h_elements.begin(), h_elements.end(),
                         [](const FeatureElement& a, const FeatureElement& b) { return a.simplex < b.simplex; });

                for (const auto& el : h_elements) {
                    if (active_nodes_.find(el.simplex) == active_nodes_.end()) {
                        uint64_t node_id = uf_.add();
                        active_nodes_[el.simplex] = node_id;
                        node_elements_[node_id] = el;
                        node_id_to_simplex_[node_id] = el.simplex;
                    }
                }
            }

            // Process manifold connectivity
            int h_e, h_f, h_v;
            CUDA_CHECK(cudaMemcpy(&h_e, res.edge_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_f, res.face_count, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_v, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost));

            if (h_e > 0) {
                std::vector<DeviceManifoldSimplex<IDType, 1>> h_s(h_e);
                CUDA_CHECK(cudaMemcpy(h_s.data(), res.edges, h_e * sizeof(h_s[0]), cudaMemcpyDeviceToHost));
                for (const auto& s : h_s) {
                    std::vector<IDType> nodes;
                    for(int i=0; i<=1; ++i) {
                        IDType nid;
                        if(resolve_simplex_to_node_regular(s.nodes[i], spatial_mesh, m, nid))
                            nodes.push_back(nid);
                    }
                    if(nodes.size()==2) {
                        std::sort(nodes.begin(), nodes.end());
                        manifold_simplices_[1].push_back(nodes);
                    }
                }
            }

            auto t_slab_end = std::chrono::high_resolution_clock::now();
            auto slab_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_slab_end - t_slab_start).count();
            std::cout << "  Slab [" << t << ", " << (t+1) << "]: " << h_node_count << " features, "
                      << slab_ms << "ms" << std::endl;
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_views));
        CUDA_CHECK(cudaFree(res.nodes)); CUDA_CHECK(cudaFree(res.node_count));
        CUDA_CHECK(cudaFree(res.edges)); CUDA_CHECK(cudaFree(res.edge_count));
        CUDA_CHECK(cudaFree(res.faces)); CUDA_CHECK(cudaFree(res.face_count));
        CUDA_CHECK(cudaFree(res.volumes)); CUDA_CHECK(cudaFree(res.volume_count));
        // device_buffer_t0 and device_buffer_t1 cleaned up automatically via RAII

        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "CUDA Streaming Total=" << d_total_ms << "ms (memory: 2 timesteps)" << std::endl;
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
        std::map<IDType, IDType> old_to_canonical;  // Map original IDs to canonical IDs
        for (const auto& s : sorted_active_simplices) {
            IDType original_id = active_nodes_.at(s);
            IDType canonical_id = uf_.add();
            canonical_active_nodes[s] = canonical_id;
            old_to_canonical[original_id] = canonical_id;
        }

        // Original manifold stitching within each pentatope (using mapped IDs)
        int n_manifold_unions = 0;
        for (int dim = 1; dim <= 3; ++dim) {
            for (const auto& nodes : manifold_simplices_[dim]) {
                for (size_t i = 1; i < nodes.size(); ++i) {
                    if (old_to_canonical.count(nodes[0]) && old_to_canonical.count(nodes[i])) {
                        uf_.unite(old_to_canonical.at(nodes[0]), old_to_canonical.at(nodes[i]));
                        n_manifold_unions++;
                    }
                }
            }
        }
        std::cout << "Manifold stitching: " << n_manifold_unions << " unions from manifold simplices" << std::endl;

        // Debug: count roots after manifold stitching only and check timestep connectivity
        {
            std::set<uint64_t> roots_after_manifold;
            std::map<uint64_t, std::set<int>> root_to_timesteps;

            for (const auto& [s, _] : active_nodes_) {
                if (canonical_active_nodes.count(s)) {
                    IDType id = canonical_active_nodes.at(s);
                    uint64_t root = uf_.find(id);
                    roots_after_manifold.insert(root);

                    // Find timestep of this CP
                    auto c0 = mesh_->get_vertex_coordinates(s.vertices[0]);
                    int t = (int)(c0[3] + 0.5);
                    root_to_timesteps[root].insert(t);
                }
            }

            std::cout << "  Unique roots after manifold only: " << roots_after_manifold.size() << std::endl;

            // Check how many components span multiple timesteps
            int n_cross_timestep_components = 0;
            for (const auto& [root, timesteps] : root_to_timesteps) {
                if (timesteps.size() > 1) n_cross_timestep_components++;
            }
            std::cout << "  Components spanning multiple timesteps: " << n_cross_timestep_components << " / " << roots_after_manifold.size() << std::endl;
        }

        // ADDITIONAL CONNECTIVITY: For 4D spacetime, use cofaces/faces like old FTK
        // This matches the approach in ftk/filters/critical_point_tracker_3d_unstructured.hh
        if (mesh_ && mesh_->get_total_dimension() == 4) {
            std::cout << "Connecting CPs via cofaces/faces (FTK-style)..." << std::endl;

            int n_unions_cofaces = 0;
            int total_cofaces_found = 0;
            int cps_with_no_cofaces = 0;

            // For each CP, find its neighbors through cofaces/faces and unite with CP neighbors
            // This matches FTK's approach: neighbors(f) returns all tets in shared pentatopes,
            // then we filter to only unite with CPs
            for (const auto& [cp_tet, _] : active_nodes_) {
                if (!canonical_active_nodes.count(cp_tet)) continue;

                IDType cp_id = canonical_active_nodes.at(cp_tet);
                auto cp_c0 = mesh_->get_vertex_coordinates(cp_tet.vertices[0]);
                int cp_t = (int)(cp_c0[3] + 0.5);

                // Find all neighbor tets through cofaces (pentatopes) -> faces (tets)
                std::set<Simplex> neighbor_tets;
                int n_cofaces = 0;

                mesh_->cofaces(cp_tet, [&](const Simplex& pent) {
                    if (pent.dimension == 4) {
                        n_cofaces++;
                        mesh_->faces(pent, [&](const Simplex& neighbor_tet) {
                            if (neighbor_tet.dimension == 3) {
                                Simplex sorted = neighbor_tet;
                                sorted.sort_vertices();
                                neighbor_tets.insert(sorted);
                            }
                        });
                    }
                });

                total_cofaces_found += n_cofaces;
                if (n_cofaces == 0) cps_with_no_cofaces++;

                // Unite this CP with all neighbor tets that are also CPs
                for (const auto& neigh : neighbor_tets) {
                    if (canonical_active_nodes.count(neigh)) {
                        IDType neigh_id = canonical_active_nodes.at(neigh);

                        // Check if this union actually merges different components
                        IDType root1 = uf_.find(cp_id);
                        IDType root2 = uf_.find(neigh_id);
                        if (root1 != root2) {
                            uf_.unite(cp_id, neigh_id);
                            n_unions_cofaces++;
                        }
                    }
                }
            }

            std::cout << "  Cofaces found: total=" << total_cofaces_found
                      << " avg=" << (total_cofaces_found / (double)active_nodes_.size())
                      << " CPs with no cofaces=" << cps_with_no_cofaces << std::endl;
            std::cout << "  Cofaces connectivity: " << n_unions_cofaces << " unions" << std::endl;

            // Count unique roots after cofaces
            std::set<uint64_t> roots_after_cofaces;
            for (const auto& [s, _] : active_nodes_) {
                if (canonical_active_nodes.count(s)) {
                    IDType id = canonical_active_nodes.at(s);
                    roots_after_cofaces.insert(uf_.find(id));
                }
            }
            std::cout << "  Unique roots after cofaces: " << roots_after_cofaces.size() << std::endl;
        }

        // Assign unique track_id per root (use root ID directly)
        std::set<uint64_t> unique_roots;
        for (auto const& [s, original_id] : active_nodes_) {
            IDType canonical_id = canonical_active_nodes.at(s);
            uint64_t root = uf_.find(canonical_id);
            unique_roots.insert(root);
        }
        std::cout << "Final connected components: " << unique_roots.size() << " (from " << active_nodes_.size() << " nodes)" << std::endl;

        std::map<IDType, uint32_t> node_to_stable_idx;
        for (const auto& s : sorted_active_simplices) {
            IDType original_id = active_nodes_.at(s);
            IDType canonical_id = canonical_active_nodes.at(s);
            uint64_t root = uf_.find(canonical_id);
            FeatureElement el = node_elements_.at(original_id);
            el.track_id = root;  // Use root ID as track_id
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
    T get_value(const ftk::ndarray<T>& arr, uint64_t v_id, const std::vector<double>& coords, const std::vector<uint64_t>& offset) {
        if (arr.nd() == 1) return arr[v_id];
        
        std::vector<double> local_coords = coords;
        for (size_t i = 0; i < local_coords.size() && i < offset.size(); ++i) local_coords[i] -= offset[i];

        if (local_coords.size() == 2) return arr.f(local_coords[0], local_coords[1]);
        else if (local_coords.size() == 3) return arr.f(local_coords[0], local_coords[1], local_coords[2]);
        else if (local_coords.size() == 4) return arr.f(local_coords[0], local_coords[1], local_coords[2], local_coords[3]);
        return (T)0;
    }

    bool extract_simplex(const Simplex& s, const std::map<std::string, ftk::ndarray<T>>& data, FeatureElement& el, Mesh* mesh) {
        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset = reg_mesh ? reg_mesh->get_offset() : std::vector<uint64_t>{0,0,0,0};

        bool success = false;
        int num_vertices = 0;

        if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) {
            T values[3][2];
            num_vertices = 3;
            for (int i = 0; i < num_vertices; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                for (int j = 0; j < 2; ++j) values[i][j] = get_value(data.at(predicate_.var_names[j]), s.vertices[i], coords, offset);
            }
            success = predicate_.extract_it(s, values, el);
        } else if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) {
            T values[2][1];
            num_vertices = 2;
            for (int i = 0; i < num_vertices; ++i) {
                auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                values[i][0] = get_value(data.at(predicate_.var_name), s.vertices[i], coords, offset);
            }
            success = predicate_.extract_it(s, values, el);
        } else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
            T values[m+1][m];
            num_vertices = m + 1;
            std::vector<const ftk::ndarray<T>*> arrays_ptrs;

            if (predicate_.use_multicomponent && !predicate_.vector_var_name.empty()) {
                // Multi-component mode: single array with shape [M, spatial..., time]
                const auto& vec_array = data.at(predicate_.vector_var_name);
                arrays_ptrs.push_back(&vec_array);

                // Extract component values from multi-component array
                for (int i = 0; i <= m; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    for (int j = 0; j < m; ++j) {
                        // Access component j at vertex position
                        // Array shape: [M, spatial..., time]
                        // Need to prepend component index to coords
                        if (coords.size() == 2) {
                            // 1D spatial + time: [M, nx, nt]
                            values[i][j] = vec_array.f(j, coords[0], coords[1]);
                        } else if (coords.size() == 3) {
                            // 2D spatial + time: [M, nx, ny, nt]
                            values[i][j] = vec_array.f(j, coords[0], coords[1], coords[2]);
                        } else if (coords.size() == 4) {
                            // 3D spatial + time: [M, nx, ny, nz, nt]
                            values[i][j] = vec_array.f(j, coords[0], coords[1], coords[2], coords[3]);
                        }
                    }
                }
            } else {
                // Legacy mode: M separate scalar arrays
                for (int i = 0; i <= m; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    for (int j = 0; j < m; ++j) {
                        values[i][j] = get_value(data.at(predicate_.var_names[j]), s.vertices[i], coords, offset);
                    }
                }
                for(int k=0; k<m; ++k) arrays_ptrs.push_back(&data.at(predicate_.var_names[k]));
            }

            // Optional scalar field
            if (!predicate_.scalar_var_name.empty()) {
                arrays_ptrs.push_back(&data.at(predicate_.scalar_var_name));
            }

            success = predicate_.extract_it(s, values, el, arrays_ptrs, mesh);
        }

        // If feature extraction succeeded, interpolate user-specified attributes
        if (success && !predicate_.attributes.empty()) {
            for (const auto& attr_spec : predicate_.attributes) {
                if (attr_spec.slot < 0 || attr_spec.slot >= 16) continue;

                // Check if attribute source exists in data
                auto it = data.find(attr_spec.source);
                if (it == data.end()) continue;

                const auto& attr_array = it->second;

                // Interpolate attribute value at feature location using barycentric coordinates
                T attr_value = 0;
                for (int i = 0; i < num_vertices; ++i) {
                    auto coords = mesh->get_vertex_coordinates(s.vertices[i]);
                    T vertex_value = 0;

                    // Handle multi-component arrays (check if first dimension is component count)
                    bool is_multicomp = (attr_array.nd() >= 2 && attr_array.dimf(0) >= 1 && attr_array.dimf(0) <= 16);

                    if (is_multicomp && attr_spec.component >= 0) {
                        // Extract specific component
                        int comp = attr_spec.component;
                        if (coords.size() == 2) {
                            vertex_value = attr_array.f(comp, coords[0], coords[1]);
                        } else if (coords.size() == 3) {
                            vertex_value = attr_array.f(comp, coords[0], coords[1], coords[2]);
                        } else if (coords.size() == 4) {
                            vertex_value = attr_array.f(comp, coords[0], coords[1], coords[2], coords[3]);
                        }
                    } else if (is_multicomp && attr_spec.type == "magnitude") {
                        // Compute magnitude of vector
                        T mag_sq = 0;
                        int ncomp = attr_array.dimf(0);
                        for (int c = 0; c < ncomp; ++c) {
                            T comp_val = 0;
                            if (coords.size() == 2) {
                                comp_val = attr_array.f(c, coords[0], coords[1]);
                            } else if (coords.size() == 3) {
                                comp_val = attr_array.f(c, coords[0], coords[1], coords[2]);
                            } else if (coords.size() == 4) {
                                comp_val = attr_array.f(c, coords[0], coords[1], coords[2], coords[3]);
                            }
                            mag_sq += comp_val * comp_val;
                        }
                        vertex_value = std::sqrt(mag_sq);
                    } else {
                        // Scalar field or single value
                        vertex_value = get_value(attr_array, s.vertices[i], coords, offset);
                    }

                    // Accumulate weighted by barycentric coordinate
                    attr_value += el.barycentric_coords[0][i] * vertex_value;
                }

                el.attributes[attr_spec.slot] = (float)attr_value;
            }
        }

        return success;
    }

    struct ThreadData {
        std::vector<std::vector<IDType>> dim1, dim2, dim3;
    };

    void marching_pentatope(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var, ThreadData& local, std::mutex& mutex, Mesh* mesh) {
        T vals[5]; uint64_t idx[5]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, FiberPredicate<T>>) threshold = predicate_.thresholds[0]; 
        else if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) threshold = predicate_.threshold;
        else return;

        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset = reg_mesh ? reg_mesh->get_offset() : std::vector<uint64_t>{0,0,0,0};

        for (int i=0; i<5; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh->get_vertex_coordinates(idx[i]);
            vals[i] = get_value(data.at(var), idx[i], coords, offset) - threshold;
            if (sos::sign(vals[i], idx[i], predicate_.sos_q) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { 
                Simplex edge = make_edge(idx[single], idx[o]); 
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); 
                else {
                    FeatureElement el;
                    if (extract_simplex(edge, data, el, mesh)) {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge));
                        else {
                            IDType node_id = uf_.add();
                            active_nodes_[edge] = node_id;
                            node_elements_[node_id] = el;
                            node_id_to_simplex_[node_id] = edge;
                            nodes.push_back(node_id);
                        }
                    }
                }
            }
            if (nodes.size() == 4) { std::sort(nodes.begin(), nodes.end()); local.dim3.push_back(nodes); }
        } else if (A.size() == 2 || B.size() == 2) {
            const auto& two = (A.size() == 2) ? A : B; const auto& three = (A.size() == 2) ? B : A;
            std::vector<IDType> T0, T1;
            for (int t : three) {
                Simplex e0 = make_edge(idx[two[0]], idx[t]); 
                Simplex e1 = make_edge(idx[two[1]], idx[t]);
                
                auto ensure_node = [&](const Simplex& s, std::vector<IDType>& list) {
                    if (active_nodes_.count(s)) list.push_back(active_nodes_.at(s));
                    else {
                        FeatureElement el;
                        if (extract_simplex(s, data, el, mesh)) {
                            std::lock_guard<std::mutex> lock(mutex);
                            if (active_nodes_.count(s)) list.push_back(active_nodes_.at(s));
                            else {
                                IDType node_id = uf_.add();
                                active_nodes_[s] = node_id;
                                node_elements_[node_id] = el;
                                node_id_to_simplex_[node_id] = s;
                                list.push_back(node_id);
                            }
                        }
                    }
                };
                ensure_node(e0, T0); ensure_node(e1, T1);
            }
            if (T0.size() == 3 && T1.size() == 3) {
                std::vector<IDType> c0 = {T0[0], T0[1], T0[2], T1[2]}; std::sort(c0.begin(), c0.end());
                std::vector<IDType> c1 = {T0[0], T0[1], T1[1], T1[2]}; std::sort(c1.begin(), c1.end());
                std::vector<IDType> c2 = {T0[0], T1[0], T1[1], T1[2]}; std::sort(c2.begin(), c2.end());
                local.dim3.push_back(c0); local.dim3.push_back(c1); local.dim3.push_back(c2);
            }
        }
    }

    void marching_tetrahedron(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data, const std::string& var, ThreadData& local, std::mutex& mutex, Mesh* mesh) {
        T vals[4]; uint64_t idx[4]; std::vector<int> A, B; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) { threshold = predicate_.threshold; } else return;
        
        auto reg_mesh = dynamic_cast<RegularSimplicialMesh*>(mesh);
        std::vector<uint64_t> offset = reg_mesh ? reg_mesh->get_offset() : std::vector<uint64_t>{0,0,0,0};

        for (int i=0; i<4; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh->get_vertex_coordinates(idx[i]);
            vals[i] = get_value(data.at(var), idx[i], coords, offset) - threshold;
            if (sos::sign(vals[i], idx[i], predicate_.sos_q) > 0) A.push_back(i); else B.push_back(i);
        }
        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0]; const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) { 
                Simplex edge = make_edge(idx[single], idx[o]); 
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); 
                else {
                    FeatureElement el;
                    if (extract_simplex(edge, data, el, mesh)) {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge));
                        else {
                            IDType node_id = uf_.add();
                            active_nodes_[edge] = node_id;
                            node_elements_[node_id] = el;
                            node_id_to_simplex_[node_id] = edge;
                            nodes.push_back(node_id);
                        }
                    }
                }
            }
            if (nodes.size() == 3) { std::sort(nodes.begin(), nodes.end()); local.dim2.push_back(nodes); }
        } else if (A.size() == 2 && B.size() == 2) {
            std::vector<IDType> nodes;
            for (int a : A) for (int b : B) { 
                Simplex edge = make_edge(idx[a], idx[b]); 
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge)); 
                else {
                    FeatureElement el;
                    if (extract_simplex(edge, data, el, mesh)) {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (active_nodes_.count(edge)) nodes.push_back(active_nodes_.at(edge));
                        else {
                            IDType node_id = uf_.add();
                            active_nodes_[edge] = node_id;
                            node_elements_[node_id] = el;
                            node_id_to_simplex_[node_id] = edge;
                            nodes.push_back(node_id);
                        }
                    }
                }
            }
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

    // OLD CODE - REMOVED
    void connect_cps_via_bfs_OLD(const std::map<Simplex, IDType>& canonical_active_nodes) {
        // Direct approach: Unite CPs that are reachable within a small neighborhood
        // For each tet with CP, look at surrounding pentatopes and unite with other CPs found there

        int n_unions_cofaces = 0;
        int n_cofaces_calls = 0;
        for (const auto& [cp_tet, _] : active_nodes_) {
            if (!canonical_active_nodes.count(cp_tet)) continue;
            IDType cp_id = canonical_active_nodes.at(cp_tet);

            int n_pents_for_this_tet = 0;
            // Find pentatopes containing this tet
            mesh_->cofaces(cp_tet, [&](const Simplex& pent) {
                n_pents_for_this_tet++;
                // Unite with all CPs in this pentatope
                mesh_->faces(pent, [&](const Simplex& other_tet) {
                    if (other_tet.dimension == 3) {
                        Simplex sorted_other = other_tet;
                        sorted_other.sort_vertices();
                        if (canonical_active_nodes.count(sorted_other)) {
                            IDType other_id = canonical_active_nodes.at(sorted_other);
                            uf_.unite(cp_id, other_id);
                            n_unions_cofaces++;
                        }
                    }
                });
            });
            n_cofaces_calls++;
            if (n_pents_for_this_tet == 0 && n_cofaces_calls <= 5) {
                // Debug: tet with no cofaces
                auto coords = mesh_->get_vertex_coordinates(cp_tet.vertices[0]);
                std::cout << "  WARNING: Tet with CP has 0 cofaces! First vertex at t=" << coords[coords.size()-1] << std::endl;
            }
        }

        // Count components after cofaces unions
        std::set<uint64_t> roots_after_cofaces;
        for (const auto& [s, _] : active_nodes_) {
            if (canonical_active_nodes.count(s)) {
                IDType id = canonical_active_nodes.at(s);
                roots_after_cofaces.insert(uf_.find(id));
            }
        }

        std::cout << "Direct cofaces: " << n_unions_cofaces << " unions via cofaces" << std::endl;
        std::cout << "Components after cofaces: " << roots_after_cofaces.size() << std::endl;
        return;  // Skip the BFS part

        // (old BFS code follows but is not reached)
        std::map<Simplex, std::set<Simplex>> adjacency;

        // Count edges in adjacency graph
        int n_edges = 0;
        for (const auto& [tet, neighbors] : adjacency) {
            n_edges += neighbors.size();
        }
        n_edges /= 2;  // Each edge counted twice
        std::cout << "  Adjacency graph: " << adjacency.size() << " nodes, " << n_edges << " edges" << std::endl;

        // BFS to find connected components and unite them
        std::set<Simplex> visited;
        int n_components_before = 0;
        for (const auto& [start_tet, _] : active_nodes_) {
            if (visited.count(start_tet)) continue;

            // BFS from start_tet
            std::vector<Simplex> component;
            std::queue<Simplex> q;
            q.push(start_tet);
            visited.insert(start_tet);

            while (!q.empty()) {
                Simplex current = q.front();
                q.pop();
                component.push_back(current);

                for (const auto& neighbor : adjacency[current]) {
                    if (!visited.count(neighbor)) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }

            // Unite all tets in this component
            if (component.size() > 0 && canonical_active_nodes.count(component[0])) {
                IDType root_id = canonical_active_nodes.at(component[0]);
                for (size_t i = 1; i < component.size(); ++i) {
                    if (canonical_active_nodes.count(component[i])) {
                        IDType id = canonical_active_nodes.at(component[i]);
                        uf_.unite(root_id, id);
                    }
                }
            }
            n_components_before++;
        }

        std::cout << "BFS connectivity: Found " << n_components_before << " components via BFS" << std::endl;
    }

    void connect_cps_across_pentatopes_OLD(const std::map<Simplex, IDType>& canonical_active_nodes) {
        // First, analyze which kinds of tets have CPs
        int n_spatial_tets = 0, n_spacetime_tets = 0;
        std::map<int, int> cps_by_time;  // time -> count
        for (const auto& [tet_simplex, _] : active_nodes_) {
            auto c0 = mesh_->get_vertex_coordinates(tet_simplex.vertices[0]);
            auto c1 = mesh_->get_vertex_coordinates(tet_simplex.vertices[1]);
            auto c2 = mesh_->get_vertex_coordinates(tet_simplex.vertices[2]);
            auto c3 = mesh_->get_vertex_coordinates(tet_simplex.vertices[3]);
            int d = c0.size() - 1;  // time is last coordinate
            double t0 = c0[d], t1 = c1[d], t2 = c2[d], t3 = c3[d];
            double tmin = std::min({t0, t1, t2, t3});
            double tmax = std::max({t0, t1, t2, t3});
            if (tmin == tmax) {
                n_spatial_tets++;
                cps_by_time[(int)tmin]++;
            } else {
                n_spacetime_tets++;
                // Count as both timesteps for spacetime tets
                cps_by_time[(int)tmin]++;
                cps_by_time[(int)tmax]++;
            }
        }
        std::cout << "CP tet types: " << n_spatial_tets << " pure spatial, " << n_spacetime_tets << " spacetime" << std::endl;
        std::cout << "CPs by timestep: ";
        for (const auto& [t, count] : cps_by_time) {
            std::cout << "t=" << t << ":" << count << " ";
        }
        std::cout << std::endl;

        // Build map: tet -> list of pentatopes containing it
        std::map<Simplex, std::vector<Simplex>> tet_to_pentatopes;
        int n_pents = 0;
        int n_pents_with_multiple_cps = 0;
        int max_cps_in_pent = 0;

        mesh_->iterate_simplices(4, [&](const Simplex& pent) {
            n_pents++;
            // Find all 3-simplices (tets) in this pentatope that contain CPs
            std::vector<Simplex> tets_with_cps;
            find_m_subsimplices(pent, 3, [&](const Simplex& tet) {
                Simplex sorted_tet = tet;
                sorted_tet.sort_vertices();
                if (active_nodes_.count(sorted_tet)) {
                    tets_with_cps.push_back(sorted_tet);
                }
            });

            if (tets_with_cps.size() > 1) n_pents_with_multiple_cps++;
            if ((int)tets_with_cps.size() > max_cps_in_pent) max_cps_in_pent = tets_with_cps.size();

            // Record that these tets are in this pentatope
            for (const auto& tet : tets_with_cps) {
                tet_to_pentatopes[tet].push_back(pent);
            }
        });

        std::cout << "  Pentatopes with multiple CPs: " << n_pents_with_multiple_cps << ", max CPs in one pent: " << max_cps_in_pent << std::endl;

        // For each tet, unite it with all other tets in the same pentatopes
        int n_unions = 0;
        int n_spatial_with_spacetime_neighbors = 0;
        for (const auto& tet_pents_pair : tet_to_pentatopes) {
            const Simplex& tet = tet_pents_pair.first;
            const std::vector<Simplex>& pents = tet_pents_pair.second;

            if (!canonical_active_nodes.count(tet)) continue;
            IDType tet_id = canonical_active_nodes.at(tet);

            // Check if this is spatial tet
            auto tc0 = mesh_->get_vertex_coordinates(tet.vertices[0]);
            bool is_spatial = true;
            double t0 = tc0[tc0.size()-1];
            for (int j = 1; j < 4; ++j) {
                auto tcj = mesh_->get_vertex_coordinates(tet.vertices[j]);
                if (tcj[tcj.size()-1] != t0) { is_spatial = false; break; }
            }

            std::set<Simplex> neighbor_tets;
            bool has_spacetime_neighbor = false;
            for (const auto& pent : pents) {
                // Find all tets with CPs in this pentatope
                find_m_subsimplices(pent, 3, [&](const Simplex& neighbor_tet) {
                    Simplex sorted_neighbor = neighbor_tet;
                    sorted_neighbor.sort_vertices();
                    bool is_same = (sorted_neighbor.dimension == tet.dimension);
                    for (int k = 0; k <= sorted_neighbor.dimension && is_same; ++k) {
                        if (sorted_neighbor.vertices[k] != tet.vertices[k]) is_same = false;
                    }
                    if (active_nodes_.count(sorted_neighbor) && !is_same) {
                        neighbor_tets.insert(sorted_neighbor);

                        // Check if neighbor is spacetime
                        auto nc0 = mesh_->get_vertex_coordinates(sorted_neighbor.vertices[0]);
                        double nt0 = nc0[nc0.size()-1];
                        for (int j = 1; j < 4; ++j) {
                            auto ncj = mesh_->get_vertex_coordinates(sorted_neighbor.vertices[j]);
                            if (ncj[ncj.size()-1] != nt0) {
                                has_spacetime_neighbor = true;
                                break;
                            }
                        }
                    }
                });
            }

            if (is_spatial && has_spacetime_neighbor) n_spatial_with_spacetime_neighbors++;

            // Unite this tet with all its neighbors
            for (const auto& neighbor : neighbor_tets) {
                if (canonical_active_nodes.count(neighbor)) {
                    IDType neighbor_id = canonical_active_nodes.at(neighbor);
                    uf_.unite(tet_id, neighbor_id);
                    n_unions++;
                }
            }
        }

        std::cout << "CP Connectivity: Checked " << n_pents << " pentatopes, found "
                  << tet_to_pentatopes.size() << " tets with CPs, performed " << n_unions << " unions" << std::endl;
        std::cout << "  Spatial tets with spacetime neighbors: " << n_spatial_with_spacetime_neighbors << "/" << n_spatial_tets << std::endl;
    }

    void find_m_subsimplices(const Simplex& s, int target_m, std::function<void(const Simplex&)> callback) {
        int n = s.dimension + 1; int r = target_m + 1; std::vector<int> p(r); std::iota(p.begin(), p.end(), 0);
        while (p[0] <= n - r) {
            Simplex f; f.dimension = target_m; for (int i = 0; i < r; ++i) f.vertices[i] = s.vertices[p[i]];
            callback(f); int i = r - 1; while (i >= 0 && p[i] == n - r + i) i--; if (i < 0) break;
            p[i]++; for (int j = i + 1; j < r; j++) p[j] = p[i] + j - i;
        }
    }

    bool resolve_simplex_to_node_regular(uint64_t encoded_id, const std::shared_ptr<RegularSimplicialMesh>& reg_mesh, int dim, IDType& nid) {
        uint64_t v_min = encoded_id >> 24; uint64_t combined_mask = encoded_id & 0xFFFFFF;
        auto c_min = reg_mesh->id_to_grid_index(v_min);
        Simplex s; s.dimension = dim; s.vertices[0] = v_min;
        for (int i = 1; i <= dim; ++i) {
            uint64_t mask = (combined_mask >> ((dim - i) * 4)) & 0xF;
            uint64_t ci[4] = {0};
            for (int k = 0; k < 4 && k < c_min.size(); ++k) { ci[k] = c_min[k]; if ((mask >> k) & 1) ci[k]++; }
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
