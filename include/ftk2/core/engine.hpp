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
            // Deterministic linking: always attach larger ID to smaller ID
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

    /**
     * @brief Run extraction and tracking on a time-varying stream (sliding window).
     */
    void execute_stream(ftk::stream<T>& stream) {
        if (!stream.advance()) return;
        auto data_t0 = stream.get_current_data();

        while (stream.advance()) {
            auto data_t1 = stream.get_current_data();
            
            // For regular grids, we combine t0 and t1 into a temporary local window data
            std::map<std::string, ftk::ndarray<T>> window_data;
            for (auto const& [name, arr0] : data_t0) {
                const auto& arr1 = data_t1.at(name);
                // In a real implementation, we'd use a zero-copy view if ndarray supports it
                // For now, assume predicates can handle two separate timesteps
            }
            
            // process_window(data_t0, data_t1);
            
            data_t0 = data_t1;
        }
    }

    void execute(const std::map<std::string, ftk::ndarray<T>>& data) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int d_total = mesh_->get_total_dimension();
        mesh_->iterate_simplices(m, [&](const Simplex& s) {
            auto elements = predicate_.extract(s, data, *mesh_);
            if (!elements.empty()) {
                uint64_t node_id = uf_.add();
                active_nodes_[s] = node_id;
                
                FeatureElement element = elements[0];
                node_elements_[node_id] = element; 
                node_id_to_simplex_[node_id] = s;
            }
        });
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
                        Simplex cell;
                        reg_mesh->get_d_simplex(hc_idx, p_idx, cell);
                        if (m == 1 && d_total == 4) marching_pentatope(cell, data);
                        else if (m == 1 && d_total == 3) marching_tetrahedron(cell, data);
                        else form_general_manifold_patches(cell, data);
                    }
                }
            }
        } else {
            mesh_->iterate_simplices(d_total, [&](const Simplex& cell) {
                if (m == 1 && d_total == 4) marching_pentatope(cell, data);
                else if (m == 1 && d_total == 3) marching_tetrahedron(cell, data);
                else form_general_manifold_patches(cell, data);
            });
        }
        auto t_end = std::chrono::high_resolution_clock::now();
        auto d_nodes = std::chrono::duration_cast<std::chrono::milliseconds>(t_nodes - t_start).count();
        auto d_manifold = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_nodes).count();
        std::cout << "CPU Execution Breakdown: Nodes=" << d_nodes << "ms, Manifold=" << d_manifold << "ms, Total=" << (d_nodes+d_manifold) << "ms" << std::endl;
    }

#if FTK_HAVE_CUDA && defined(__CUDACC__)
    void execute_cuda(const std::map<std::string, ftk::ndarray<T>>& data) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        auto reg_mesh = std::dynamic_pointer_cast<RegularSimplicialMesh>(mesh_);
        if (!reg_mesh) return;

        RegularSimplicialMeshDevice d_mesh;
        d_mesh.ndims = reg_mesh->get_total_dimension();
        auto l_dims = reg_mesh->get_local_dims();
        auto off = reg_mesh->get_offset();
        auto g_dims = reg_mesh->get_global_dims();
        for(int i=0; i<d_mesh.ndims && i<4; ++i) {
            d_mesh.local_dims[i] = l_dims[i];
            d_mesh.offset[i] = off[i];
            d_mesh.global_dims[i] = g_dims[i];
        }

        std::vector<std::string> vars;
        if constexpr (m == 1 && std::is_same_v<PredicateType, ContourPredicate<T>>) vars = {predicate_.var_name};
        else if constexpr (std::is_same_v<PredicateType, CriticalPointPredicate<m, T>>) {
            for(int i=0; i<m; ++i) vars.push_back(predicate_.var_names[i]);
            if (predicate_.scalar_var_name[0] != '\0') vars.push_back(std::string(predicate_.scalar_var_name));
        }

        auto t_setup = std::chrono::high_resolution_clock::now();

        std::vector<CudaDataView<T>> h_views;
        for (const auto& var : vars) {
            const auto& arr = data.at(var);
            CudaDataView<T> view;
            cudaMalloc((void**)&view.data, arr.nelem() * sizeof(T));
            cudaMemcpy((void*)view.data, arr.pdata(), arr.nelem() * sizeof(T), cudaMemcpyHostToDevice);
            auto lattice = arr.get_lattice();
            for(int i=0; i<4; ++i) { 
                view.dims[i] = (i < arr.nd()) ? arr.dimf(i) : 1; 
                view.s[i] = (i < arr.nd()) ? lattice.prod_[arr.nd() - 1 - i] : 0; 
            }
            view.ndims = arr.nd();
            h_views.push_back(view);
        }

        CudaDataView<T>* d_views;
        cudaMalloc(&d_views, h_views.size() * sizeof(CudaDataView<T>));
        cudaMemcpy(d_views, h_views.data(), h_views.size() * sizeof(CudaDataView<T>), cudaMemcpyHostToDevice);

        CudaExtractionResult<IDType> res;
        res.max_nodes = 1000000; res.max_conn = 3000000;
        cudaMalloc(&res.nodes, res.max_nodes * sizeof(FeatureElement));
        cudaMalloc(&res.node_count, sizeof(int));
        cudaMalloc(&res.edges, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 1>));
        cudaMalloc(&res.edge_count, sizeof(int));
        cudaMalloc(&res.faces, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 2>));
        cudaMalloc(&res.face_count, sizeof(int));
        cudaMalloc(&res.volumes, res.max_conn * sizeof(DeviceManifoldSimplex<IDType, 3>));
        cudaMalloc(&res.volume_count, sizeof(int));
        cudaMemset(res.node_count, 0, sizeof(int));
        cudaMemset(res.edge_count, 0, sizeof(int));
        cudaMemset(res.face_count, 0, sizeof(int));
        cudaMemset(res.volume_count, 0, sizeof(int));

        auto t_h2d = std::chrono::high_resolution_clock::now();

        uint64_t n_v = d_mesh.get_num_vertices();
        extraction_kernel<<< (n_v+255)/256, 256 >>>(d_mesh, predicate_, d_views, h_views.size(), res);
        cudaDeviceSynchronize();

        auto t_kernel = std::chrono::high_resolution_clock::now();

        int h_node_count;
        cudaMemcpy(&h_node_count, res.node_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        std::vector<FeatureElement> h_elements(std::min(h_node_count, res.max_nodes));
        cudaMemcpy(h_elements.data(), res.nodes, h_elements.size() * sizeof(FeatureElement), cudaMemcpyDeviceToHost);

        auto t_d2h = std::chrono::high_resolution_clock::now();

        // SORT elements to ensure deterministic Union-Find IDs
        std::sort(h_elements.begin(), h_elements.end(), [](const FeatureElement& a, const FeatureElement& b) {
            return a.simplex < b.simplex;
        });

        for (const auto& el : h_elements) {
            uint64_t node_id = uf_.add();
            active_nodes_[el.simplex] = node_id;
            
            FeatureElement element = el;
            // Ensure any non-trivial mapping or initialization happens here
            node_elements_[node_id] = element;
            node_id_to_simplex_[node_id] = el.simplex;
        }

        int h_vol_count;
        cudaMemcpy(&h_vol_count, res.volume_count, sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<DeviceManifoldSimplex<IDType, 3>> h_volumes(std::min(h_vol_count, res.max_conn));
        cudaMemcpy(h_volumes.data(), res.volumes, h_volumes.size() * sizeof(DeviceManifoldSimplex<IDType, 3>), cudaMemcpyDeviceToHost);

        auto t_d2h_manifold = std::chrono::high_resolution_clock::now();

        for (const auto& vol : h_volumes) {
            std::vector<IDType> nodes;
            for (int i = 0; i < 4; ++i) {
                uint64_t id = (uint64_t)vol.nodes[i];
                uint64_t v0 = id >> 4;
                uint64_t mask = id & 0xF;
                
                auto c0 = reg_mesh->id_to_grid_index(v0);
                std::vector<uint64_t> c1 = c0;
                for (int k = 0; k < 4; ++k) if ((mask >> k) & 1) c1[k]++;
                uint64_t v1 = reg_mesh->grid_index_to_id(c1);

                Simplex edge; edge.dimension = 1;
                edge.vertices[0] = v0; edge.vertices[1] = v1;
                // edge.sort_vertices(); // already v0 < v1

                if (active_nodes_.count(edge)) {
                    nodes.push_back(active_nodes_[edge]);
                }
            }
            if (nodes.size() == 4) {
                manifold_simplices_[3].push_back(nodes);
                for (int i = 1; i < 4; ++i) uf_.unite(nodes[0], nodes[i]);
            }
        }

        auto t_uf = std::chrono::high_resolution_clock::now();

        for (auto& v : h_views) cudaFree((void*)v.data);
        cudaFree(d_views);
        cudaFree(res.nodes); cudaFree(res.node_count);
        cudaFree(res.edges); cudaFree(res.edge_count);
        cudaFree(res.faces); cudaFree(res.face_count);
        cudaFree(res.volumes); cudaFree(res.volume_count);

        auto t_end = std::chrono::high_resolution_clock::now();

        auto d_setup = std::chrono::duration_cast<std::chrono::milliseconds>(t_setup - t_start).count();
        auto d_h2d = std::chrono::duration_cast<std::chrono::milliseconds>(t_h2d - t_setup).count();
        auto d_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(t_kernel - t_h2d).count();
        auto d_d2h = std::chrono::duration_cast<std::chrono::milliseconds>(t_d2h - t_kernel).count();
        auto d_d2h_m = std::chrono::duration_cast<std::chrono::milliseconds>(t_d2h_manifold - t_d2h).count();
        auto d_uf = std::chrono::duration_cast<std::chrono::milliseconds>(t_uf - t_d2h_manifold).count();
        auto d_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

        std::cout << "CUDA Execution Breakdown:" << std::endl;
        std::cout << "  - Setup:    " << d_setup << " ms" << std::endl;
        std::cout << "  - H2D:      " << d_h2d << " ms" << std::endl;
        std::cout << "  - Kernel:   " << d_kernel << " ms" << std::endl;
        std::cout << "  - D2H (N):  " << d_d2h << " ms" << std::endl;
        std::cout << "  - D2H (M):  " << d_d2h_m << " ms" << std::endl;
        std::cout << "  - UF+Conn:  " << d_uf << " ms" << std::endl;
        std::cout << "  - TOTAL:    " << d_total_ms << " ms" << std::endl;
    }
#endif

    FeatureComplex get_complex() {
        FeatureComplex complex;
        std::map<IDType, uint32_t> node_to_stable_idx;
        
        // 1. Define stable indices based on Simplex order (using active_nodes_)
        for (auto const& [s, node_id] : active_nodes_) {
            uint64_t root = uf_.find(node_id);
            FeatureElement el = node_elements_.at(node_id);
            el.track_id = (uint64_t)root;
            
            node_to_stable_idx[node_id] = complex.vertices.size();
            complex.vertices.push_back(el);
        }

        // 2. Build connectivity using stable indices
        for (int dim = 1; dim <= 3; ++dim) {
            if (!manifold_simplices_[dim].empty()) {
                FeatureComplex::SimplexIndices conn; conn.dimension = dim;
                for (auto const& simplex : manifold_simplices_[dim]) {
                    for (IDType nid : simplex) conn.indices.push_back(node_to_stable_idx.at(nid));
                }
                complex.connectivity.push_back(conn);
            }
        }
        
        // 3. Sort connectivity itself
        complex.sort();
        
        return complex;
    }

private:
    void marching_pentatope(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data) {
        T vals[5]; uint64_t idx[5]; std::vector<int> A, B;
        std::string var; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) {
            var = predicate_.var_name; threshold = predicate_.threshold;
        } else return;

        for (int i=0; i<5; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh_->get_vertex_coordinates(idx[i]);
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2], coords[3]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }

        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0];
            const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) {
                Simplex edge = make_edge(idx[single], idx[o]);
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]);
            }
            if (nodes.size() == 4) {
                manifold_simplices_[3].push_back({nodes[0], nodes[1], nodes[2], nodes[3]});
                for (int i=1; i<4; ++i) uf_.unite(nodes[0], nodes[i]);
            }
        } else if (A.size() == 2 || B.size() == 2) {
            const auto& two = (A.size() == 2) ? A : B;
            const auto& three = (A.size() == 2) ? B : A;
            std::vector<uint64_t> T0, T1;
            for (int t : three) {
                Simplex e0 = make_edge(idx[two[0]], idx[t]);
                Simplex e1 = make_edge(idx[two[1]], idx[t]);
                if (active_nodes_.count(e0)) T0.push_back(active_nodes_[e0]);
                if (active_nodes_.count(e1)) T1.push_back(active_nodes_[e1]);
            }
            if (T0.size() == 3 && T1.size() == 3) {
                manifold_simplices_[3].push_back({T0[0], T0[1], T0[2], T1[2]});
                manifold_simplices_[3].push_back({T0[0], T0[1], T1[1], T1[2]});
                manifold_simplices_[3].push_back({T0[0], T1[0], T1[1], T1[2]});
                for (int i=0; i<3; ++i) { uf_.unite(T0[0], T0[i]); uf_.unite(T0[0], T1[i]); }
            }
        }
    }

    void marching_tetrahedron(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data) {
        T vals[4]; uint64_t idx[4]; std::vector<int> A, B;
        std::string var; T threshold = 0;
        if constexpr (std::is_same_v<PredicateType, ContourPredicate<T>>) {
            var = predicate_.var_name; threshold = predicate_.threshold;
        } else return;

        for (int i=0; i<4; ++i) {
            idx[i] = cell.vertices[i]; auto coords = mesh_->get_vertex_coordinates(idx[i]);
            vals[i] = data.at(var).f(coords[0], coords[1], coords[2]) - threshold;
            if (sos::sign(vals[i], idx[i]) > 0) A.push_back(i); else B.push_back(i);
        }

        if (A.size() == 1 || B.size() == 1) {
            const auto& single = (A.size() == 1) ? A[0] : B[0];
            const auto& others = (A.size() == 1) ? B : A;
            std::vector<IDType> nodes;
            for (int o : others) {
                Simplex edge = make_edge(idx[single], idx[o]);
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]);
            }
            if (nodes.size() == 3) manifold_simplices_[2].push_back({nodes[0], nodes[1], nodes[2]});
        } else if (A.size() == 2 && B.size() == 2) {
            std::vector<IDType> nodes;
            for (int a : A) for (int b : B) {
                Simplex edge = make_edge(idx[a], idx[b]);
                if (active_nodes_.count(edge)) nodes.push_back(active_nodes_[edge]);
            }
            if (nodes.size() == 4) {
                manifold_simplices_[2].push_back({nodes[0], nodes[1], nodes[2]});
                manifold_simplices_[2].push_back({nodes[0], nodes[2], nodes[3]});
            }
        }
    }

    void form_general_manifold_patches(const Simplex& cell, const std::map<std::string, ftk::ndarray<T>>& data) {
        std::vector<IDType> nodes;
        find_m_subsimplices(cell, m, [&](const Simplex& f) {
            if (active_nodes_.count(f)) nodes.push_back(active_nodes_[f]);
        });
        if (nodes.size() >= 2) {
            int k = mesh_->get_total_dimension() - m;
            for (size_t i = 1; i < nodes.size(); ++i) uf_.unite(nodes[0], nodes[i]);
            if (k == 1 && nodes.size() == 2) manifold_simplices_[1].push_back({nodes[0], nodes[1]});
            else if (k == 2 && nodes.size() >= 3) {
                for (size_t i = 1; i < nodes.size() - 1; ++i) manifold_simplices_[2].push_back({nodes[0], nodes[i], nodes[i+1]});
            }
        }
    }

    Simplex make_edge(uint64_t v0, uint64_t v1) {
        Simplex s; s.dimension = 1;
        s.vertices[0] = std::min(v0, v1); s.vertices[1] = std::max(v0, v1);
        return s;
    }

    void find_m_subsimplices(const Simplex& s, int target_m, std::function<void(const Simplex&)> callback) {
        int n = s.dimension + 1; int r = target_m + 1;
        std::vector<int> p(r); std::iota(p.begin(), p.end(), 0);
        while (p[0] <= n - r) {
            Simplex f; f.dimension = target_m;
            for (int i = 0; i < r; ++i) f.vertices[i] = s.vertices[p[i]];
            callback(f);
            int i = r - 1;
            while (i >= 0 && p[i] == n - r + i) i--;
            if (i < 0) break;
            p[i]++;
            for (int j = i + 1; j < r; j++) p[j] = p[i] + j - i;
        }
    }

    std::shared_ptr<Mesh> mesh_; PredicateType predicate_; UnionFind<IDType> uf_;
    std::map<Simplex, IDType> active_nodes_;
    std::map<IDType, FeatureElement> node_elements_;
    std::map<IDType, Simplex> node_id_to_simplex_;
    std::map<int, std::vector<std::vector<IDType>>> manifold_simplices_;
};

} // namespace ftk2
