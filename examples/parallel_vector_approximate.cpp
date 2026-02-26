#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/numeric/cross_product.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>

using namespace ftk2;

/**
 * @brief Generate synthetic vector fields with parallel vectors
 *
 * Creates two vector fields U and V that are parallel along certain curves.
 * These curves represent vortex cores (Sujudi-Haimes criterion).
 */
void generate_parallel_vector_field(int nx, int ny, int nz, int nt,
                                   ftk::ndarray<double>& u,
                                   ftk::ndarray<double>& v)
{
    u.reshapef({3, (size_t)nx, (size_t)ny, (size_t)nz, (size_t)nt});
    v.reshapef({3, (size_t)nx, (size_t)ny, (size_t)nz, (size_t)nt});

    // Create a simple vortex with parallel velocity and vorticity along core
    double cx = nx / 2.0;
    double cy = ny / 2.0;
    double cz = nz / 2.0;

    for (int t = 0; t < nt; ++t) {
        double offset = 0.1 * t;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double x = i - cx - offset;
                    double y = j - cy;
                    double z = k - cz;

                    double r = std::sqrt(x*x + y*y);
                    double r_total = std::sqrt(x*x + y*y + z*z);

                    // Velocity field (U): swirling motion around z-axis
                    double theta = std::atan2(y, x);
                    double vtheta = std::tanh(r / 3.0);

                    u.f(0, i, j, k, t) = -vtheta * std::sin(theta);  // u_x
                    u.f(1, i, j, k, t) =  vtheta * std::cos(theta);  // u_y
                    u.f(2, i, j, k, t) =  0.1 * z / (nz + 1.0);      // u_z (axial flow)

                    // Vorticity-like field (V): should be parallel to U along vortex core
                    // Along core (r ≈ 0), make V parallel to U
                    // Away from core, add some deviation
                    double parallel_factor = std::exp(-r_total / 2.0);
                    double deviation = (1.0 - parallel_factor);

                    v.f(0, i, j, k, t) = u.f(0, i, j, k, t) * parallel_factor + deviation * y / (ny + 1.0);
                    v.f(1, i, j, k, t) = u.f(1, i, j, k, t) * parallel_factor + deviation * x / (nx + 1.0);
                    v.f(2, i, j, k, t) = u.f(2, i, j, k, t) * parallel_factor + deviation * z / (nz + 1.0);
                }
            }
        }
    }
}

int main() {
    std::cout << "=== FTK2 Approximate Parallel Vector Tracking ===" << std::endl;
    std::cout << "Method: Cross product W = U × V, track fiber W_0=W_1=0, filter by |W_2|" << std::endl;

    // Parameters
    const int nx = 16, ny = 16, nz = 16, nt = 5;
    const double w2_threshold = 0.1;  // Filter PV curves where |W_2| < threshold

    std::cout << "\n[1/6] Generating synthetic vector fields..." << std::endl;
    std::cout << "Domain: " << nx << "x" << ny << "x" << nz << " spatial, " << nt << " timesteps" << std::endl;

    ftk::ndarray<double> u, v;
    generate_parallel_vector_field(nx, ny, nz, nt, u, v);

    std::cout << "  U (velocity): [3, " << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;
    std::cout << "  V (vorticity): [3, " << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;

    // Compute cross product W = U × V
    std::cout << "\n[2/6] Computing cross product W = U × V..." << std::endl;
    ftk::ndarray<double> w;
    cross_product_3d(u, v, w);

    std::cout << "  W = U × V: [3, " << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;
    std::cout << "  Where U ∥ V (parallel), W ≈ 0" << std::endl;

    // Decompose W into separate components
    std::cout << "\n[3/6] Decomposing W into components..." << std::endl;
    auto w_components = decompose_components(w);

    std::cout << "  W_0: [" << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;
    std::cout << "  W_1: [" << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;
    std::cout << "  W_2: [" << nx << ", " << ny << ", " << nz << ", " << nt << "]" << std::endl;

    // Prepare data for fiber tracking
    std::map<std::string, ftk::ndarray<double>> data;
    data["w0"] = w_components[0];
    data["w1"] = w_components[1];
    data["w2"] = w_components[2];  // For attribute recording

    // Create spacetime mesh
    std::cout << "\n[4/6] Creating spacetime mesh..." << std::endl;
    std::vector<uint64_t> dims = {(uint64_t)nx, (uint64_t)ny, (uint64_t)nz, (uint64_t)nt};
    auto mesh = std::make_shared<RegularSimplicialMesh>(dims);

    std::cout << "  Mesh: " << nx << "x" << ny << "x" << nz << "x" << nt
              << " (" << mesh->get_num_vertices() << " vertices)" << std::endl;

    // Create fiber predicate for W_0 = 0 and W_1 = 0
    std::cout << "\n[5/6] Tracking fiber surface W_0 = W_1 = 0..." << std::endl;
    FiberPredicate<double> predicate;
    predicate.var_names[0] = "w0";
    predicate.var_names[1] = "w1";
    predicate.thresholds[0] = 0.0;
    predicate.thresholds[1] = 0.0;

    // Configure W_2 as attribute for filtering
    AttributeSpec w2_attr;
    w2_attr.name = "w2";
    w2_attr.source = "w2";
    w2_attr.type = "scalar";
    w2_attr.slot = 0;
    predicate.attributes.push_back(w2_attr);

    std::cout << "  Predicate: Fiber (W_0=0, W_1=0)" << std::endl;
    std::cout << "  Attribute: W_2 (for filtering)" << std::endl;

    // Create engine and execute
    SimplicialEngine<double, FiberPredicate<double>> engine(mesh, predicate);
    engine.execute(data);

    // Get results
    auto complex = engine.get_complex();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Total detections: " << complex.vertices.size() << std::endl;

    // Filter by |W_2| threshold
    int count_filtered = 0;
    for (const auto& v : complex.vertices) {
        double w2_val = v.attributes[0];  // W_2 is in slot 0
        if (std::abs(w2_val) < w2_threshold) {
            count_filtered++;
        }
    }

    std::cout << "\nFiltering by |W_2| < " << w2_threshold << ":" << std::endl;
    std::cout << "  Passed filter: " << count_filtered << " / " << complex.vertices.size()
              << " (" << (100.0 * count_filtered / complex.vertices.size()) << "%)" << std::endl;

    // Analyze W_2 distribution
    std::cout << "\nW_2 statistics:" << std::endl;
    double w2_min = 1e10, w2_max = -1e10, w2_sum = 0;
    for (const auto& v : complex.vertices) {
        double w2_val = v.attributes[0];
        w2_min = std::min(w2_min, std::abs(w2_val));
        w2_max = std::max(w2_max, std::abs(w2_val));
        w2_sum += std::abs(w2_val);
    }
    double w2_mean = w2_sum / complex.vertices.size();

    std::cout << "  |W_2| range: [" << w2_min << ", " << w2_max << "]" << std::endl;
    std::cout << "  |W_2| mean: " << w2_mean << std::endl;

    // Print sample PV curves
    std::map<uint64_t, int> track_counts;
    for (const auto& v : complex.vertices) {
        double w2_val = v.attributes[0];
        if (std::abs(w2_val) < w2_threshold) {
            track_counts[v.track_id]++;
        }
    }

    std::cout << "\nParallel vector curves (|W_2| < " << w2_threshold << "):" << std::endl;
    int count = 0;
    for (const auto& [track_id, n_points] : track_counts) {
        std::cout << "  Track " << track_id << ": " << n_points << " points" << std::endl;
        if (++count >= 5) break;
    }

    // Write output
    std::cout << "\n[6/6] Writing results..." << std::endl;
    write_complex_to_vtu(complex, *mesh, "parallel_vector_approximate.vtu");

    std::cout << "\n=== Complete ===" << std::endl;
    std::cout << "Method summary:" << std::endl;
    std::cout << "  1. Computed W = U × V (cross product)" << std::endl;
    std::cout << "  2. Tracked fiber W_0 = W_1 = 0" << std::endl;
    std::cout << "  3. Recorded W_2 as attribute" << std::endl;
    std::cout << "  4. Filtered by |W_2| < " << w2_threshold << std::endl;
    std::cout << "\nThis approximates parallel vectors (U ∥ V) without cubic solver!" << std::endl;
    std::cout << "Applications: Sujudi-Haimes vortex cores, parallel vector criteria" << std::endl;

    return 0;
}
