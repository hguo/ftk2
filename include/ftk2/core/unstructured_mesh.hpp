#pragma once

#include <ftk2/core/mesh.hpp>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

namespace ftk2 {

class UnstructuredSimplicialMesh : public Mesh {
public:
    // cells are flattened indices. cell_dim is the dimension of the cell (e.g. 3 for tet).
    // coords are always 3D from VTK reader
    UnstructuredSimplicialMesh(int spatial_dim, int cell_dim, 
                               const std::vector<double>& coords, 
                               const std::vector<uint64_t>& cells) 
        : spatial_dim_(spatial_dim), cell_dim_(cell_dim), coords_(coords) 
    {
        // Build dimension 0 (vertices)
        // VTK reader always gives 3 coordinates per point
        int n_verts = coords_.size() / 3;
        std::cout << "UnstructuredMesh: building " << n_verts << " vertices." << std::endl;
        simplices_[0].resize(n_verts);
        for (int i = 0; i < n_verts; ++i) {
            simplices_[0][i].dimension = 0;
            simplices_[0][i].vertices[0] = i;
        }

        // Build top dimension cells
        int n_cells = cells.size() / (cell_dim + 1);
        std::cout << "UnstructuredMesh: building " << n_cells << " cells of dim " << cell_dim << std::endl;
        simplices_[cell_dim].resize(n_cells);
        for (int i = 0; i < n_cells; ++i) {
            simplices_[cell_dim][i].dimension = cell_dim;
            for (int j = 0; j <= cell_dim; ++j) {
                simplices_[cell_dim][i].vertices[j] = cells[i * (cell_dim + 1) + j];
            }
            simplices_[cell_dim][i].sort_vertices();
        }

        // Build intermediate dimensions
        for (int d = cell_dim - 1; d >= 1; --d) {
            std::set<Simplex> face_set;
            for (const auto& s : simplices_[d + 1]) {
                faces(s, [&](const Simplex& f) {
                    face_set.insert(f);
                });
            }
            simplices_[d].assign(face_set.begin(), face_set.end());
            std::cout << "UnstructuredMesh: found " << simplices_[d].size() << " unique simplices of dim " << d << std::endl;
        }

        // Build simplex-to-index maps for all dimensions
        for (int k = 0; k <= cell_dim; ++k) {
            for (uint64_t i = 0; i < simplices_[k].size(); ++i) {
                simplex_id_[k][simplices_[k][i]] = i;
            }
        }

        // Build coface adjacency tables for dimensions 0..cell_dim-1
        for (int k = 0; k < cell_dim; ++k) {
            cofaces_table_[k].resize(simplices_[k].size());
        }
        for (int k = cell_dim - 1; k >= 0; --k) {
            for (uint64_t j = 0; j < simplices_[k + 1].size(); ++j) {
                faces(simplices_[k + 1][j], [&](const Simplex& f) {
                    auto it = simplex_id_[k].find(f);
                    if (it != simplex_id_[k].end()) {
                        cofaces_table_[k][it->second].push_back(j);
                    }
                });
            }
        }
    }

    int get_spatial_dimension() const override { return spatial_dim_; }
    int get_total_dimension() const override { return cell_dim_; }
    uint64_t get_num_vertices() const override { return simplices_[0].size(); }

    void iterate_simplices(int k, std::function<void(const Simplex&)> callback) const override {
        if (k < 0 || k > cell_dim_) return;
        
        ftk2::parallel_for(size_t(0), simplices_[k].size(), [&](size_t i, int tid) {
            callback(simplices_[k][i]);
        });
    }

    void cofaces(const Simplex& s, std::function<void(const Simplex&)> callback) const override {
        int k = s.dimension;
        if (k < 0 || k >= cell_dim_) return;
        auto it = simplex_id_[k].find(s);
        if (it == simplex_id_[k].end()) return;
        for (uint64_t coface_idx : cofaces_table_[k][it->second]) {
            callback(simplices_[k + 1][coface_idx]);
        }
    }

    std::vector<double> get_vertex_coordinates(uint64_t vertex_id) const override {
        if (vertex_id >= simplices_[0].size()) return {};
        std::vector<double> c(spatial_dim_);
        for (int i = 0; i < spatial_dim_; ++i) {
            c[i] = coords_[vertex_id * 3 + i]; // Always 3 coords per vertex in buffer
        }
        return c;
    }

private:
    int spatial_dim_;
    int cell_dim_; 
    std::vector<double> coords_;
    std::vector<Simplex> simplices_[4];
    std::map<Simplex, uint64_t> simplex_id_[4];           // simplex → index in simplices_[k]
    std::vector<std::vector<uint64_t>> cofaces_table_[4]; // cofaces_table_[k][i] = (k+1)-simplex indices containing simplices_[k][i]
};

} // namespace ftk2
