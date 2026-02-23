#include <ftk2/core/mesh.hpp>

namespace ftk2 {

std::unique_ptr<Mesh> MeshFactory::create_regular_mesh(const std::vector<uint64_t>& local_dims, 
                                                     const std::vector<uint64_t>& offset,
                                                     const std::vector<uint64_t>& global_dims) {
    return std::make_unique<RegularSimplicialMesh>(local_dims, offset, global_dims);
}

std::unique_ptr<Mesh> MeshFactory::create_extruded_mesh(std::shared_ptr<Mesh> base_mesh, uint64_t n_layers) {
    return std::make_unique<ExtrudedSimplicialMesh>(base_mesh, n_layers);
}

} // namespace ftk2
