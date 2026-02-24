#include <ftk2/core/mesh.hpp>
#include <ftk2/core/unstructured_mesh.hpp>

namespace ftk2 {

std::unique_ptr<Mesh> MeshFactory::create_regular_mesh(const std::vector<uint64_t>& local_dims, 
                                                     const std::vector<uint64_t>& offset,
                                                     const std::vector<uint64_t>& global_dims) {
    return std::make_unique<RegularSimplicialMesh>(local_dims, offset, global_dims);
}

std::unique_ptr<Mesh> MeshFactory::create_unstructured_mesh(int spatial_dim, int cell_dim,
                                                          const std::vector<double>& coords,
                                                          const std::vector<uint64_t>& cells) {
    return std::make_unique<UnstructuredSimplicialMesh>(spatial_dim, cell_dim, coords, cells);
}

std::unique_ptr<Mesh> MeshFactory::create_extruded_mesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers) {
    return std::make_unique<ExtrudedSimplicialMesh>(base_mesh, n_layers);
}

} // namespace ftk2
