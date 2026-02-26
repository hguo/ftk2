#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ndarray/ndarray.hh>
#include <ndarray/synthetic.hh>
#include <vector>
#include <string>
#include <map>
#include <iostream>

using namespace ftk2;

static int total_tests = 0;
static int passed_tests = 0;

#define ASSERT_TRUE(...) \
    total_tests++; \
    if ((__VA_ARGS__)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: " << #__VA_ARGS__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    }

#define ASSERT_EQ(a, b) \
    total_tests++; \
    if ((a) == (b)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_EQ(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    }

#define ASSERT_NEAR(a, b, eps) \
    total_tests++; \
    if (std::abs((a) - (b)) < (eps)) { \
        passed_tests++; \
    } else { \
        std::cerr << "FAILED: ASSERT_NEAR(" << #a << ", " << #b << ") got " << (a) << ", expected " << (b) << " within " << (eps) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    }

template <typename T, typename PredicateType>
void verify_parity(std::shared_ptr<RegularSimplicialMesh> mesh, 
                  PredicateType pred, 
                  const std::map<std::string, ftk::ndarray<T>>& data,
                  const std::vector<std::string>& vars) 
{
    // 1. Run CPU
    SimplicialEngine<T, PredicateType> cpu_engine(mesh, pred);
    cpu_engine.execute(data, vars);
    auto cpu_complex = cpu_engine.get_complex();

    // 2. Run GPU
#if FTK_HAVE_CUDA
    SimplicialEngine<T, PredicateType> gpu_engine(mesh, pred);
    gpu_engine.execute_cuda(data, vars);
    auto gpu_complex = gpu_engine.get_complex();

    if (cpu_complex.vertices.size() != gpu_complex.vertices.size()) {
        std::cerr << "  CPU: Nodes=" << cpu_complex.vertices.size() << ", Cells=" << (cpu_complex.connectivity.empty() ? 0 : cpu_complex.connectivity[0].indices.size()/(cpu_complex.connectivity[0].dimension+1)) << std::endl;
        std::cerr << "  GPU: Nodes=" << gpu_complex.vertices.size() << ", Cells=" << (gpu_complex.connectivity.empty() ? 0 : gpu_complex.connectivity[0].indices.size()/(gpu_complex.connectivity[0].dimension+1)) << std::endl;
    }

    // 3. Compare Results
    ASSERT_EQ(cpu_complex.vertices.size(), gpu_complex.vertices.size());
    
    for (size_t i = 0; i < cpu_complex.vertices.size(); ++i) {
        if (cpu_complex.vertices[i].track_id != gpu_complex.vertices[i].track_id || std::abs(cpu_complex.vertices[i].scalar - gpu_complex.vertices[i].scalar) > 1e-6) {
             std::cerr << "Mismatch at node " << i << std::endl;
             std::cerr << "  CPU: TrackID=" << cpu_complex.vertices[i].track_id << ", Scalar=" << cpu_complex.vertices[i].scalar << std::endl;
             std::cerr << "  GPU: TrackID=" << gpu_complex.vertices[i].track_id << ", Scalar=" << gpu_complex.vertices[i].scalar << std::endl;
             ASSERT_TRUE(false);
        }
        passed_tests += 2; total_tests += 2; // Hack to count internal checks
    }

    ASSERT_EQ(cpu_complex.connectivity.size(), gpu_complex.connectivity.size());
    for (size_t i = 0; i < cpu_complex.connectivity.size(); ++i) {
        ASSERT_EQ(cpu_complex.connectivity[i].dimension, gpu_complex.connectivity[i].dimension);
        size_t n_indices = cpu_complex.connectivity[i].indices.size();
        ASSERT_EQ(n_indices, gpu_complex.connectivity[i].indices.size());
        
        int n = cpu_complex.connectivity[i].dimension + 1;
        for (size_t j = 0; j < n_indices; j += n) {
            for (int k = 0; k < n; ++k) {
                if (cpu_complex.connectivity[i].indices[j + k] != gpu_complex.connectivity[i].indices[j + k]) {
                    std::cerr << "Mismatch in cell " << (j/n) << " at index " << k << " (dim " << (n-1) << ")" << std::endl;
                    std::cerr << "  CPU cell: "; for(int l=0; l<n; ++l) std::cerr << cpu_complex.connectivity[i].indices[j+l] << " "; std::cerr << std::endl;
                    std::cerr << "  GPU cell: "; for(int l=0; l<n; ++l) std::cerr << gpu_complex.connectivity[i].indices[j+l] << " "; std::cerr << std::endl;
                    ASSERT_TRUE(false);
                }
                passed_tests++; total_tests++;
            }
        }
    }
#endif
}

void test_parity() {
    std::cout << "Testing CPU/GPU Parity..." << std::endl;

    // 1. Levelset 2D Parity (m=1, d=3)
    {
        std::cout << "Case 1: Levelset 2D" << std::endl;
        const int DW = 16, DH = 16, DT = 5;
        ftk::ndarray<double> scalar({(size_t)DW, (size_t)DH, (size_t)DT});
        for (int t = 0; t < DT; ++t) {
            double time = (double)t / (DT - 1) * M_PI;
            auto s = ftk::synthetic_merger_2D<double>(DW, DH, time);
            for (int y = 0; y < DH; ++y) for (int x = 0; x < DW; ++x) scalar.f(x, y, t) = s.f(x, y);
        }
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT});
        ContourPredicate<double> pred; pred.var_name = "S"; pred.threshold = 0.5;
        std::map<std::string, ftk::ndarray<double>> data = {{"S", scalar}};
        std::vector<std::string> vars = {"S"};
        verify_parity<double, ContourPredicate<double>>(mesh, pred, data, vars);
    }

    // 2. Critical Point 2D Parity (m=2, d=3)
    {
        std::cout << "Case 2: Critical Point 2D" << std::endl;
        const int DW = 16, DH = 16, DT = 5;
        ftk::ndarray<double> scalar = ftk::synthetic_woven_2Dt<double>(DW, DH, DT);
        ftk::ndarray<double> u({(size_t)DW, (size_t)DH, (size_t)DT}), v({(size_t)DW, (size_t)DH, (size_t)DT});
        for (int t = 0; t < DT; ++t) {
            for (int y = 1; y < DH - 1; ++y) {
                for (int x = 1; x < DW - 1; ++x) {
                    u.f(x, y, t) = (scalar.f(x + 1, y, t) - scalar.f(x - 1, y, t)) / 2.0;
                    v.f(x, y, t) = (scalar.f(x, y + 1, t) - scalar.f(x, y - 1, t)) / 2.0;
                }
            }
        }
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW-2, (uint64_t)DH-2, (uint64_t)DT}, std::vector<uint64_t>{1, 1, 0}, std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DT});
        CriticalPointPredicate<2, double> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.scalar_var_name = "W";
        std::map<std::string, ftk::ndarray<double>> data = {{"U", u}, {"V", v}, {"W", scalar}};
        std::vector<std::string> vars = {"U", "V", "W"};
        verify_parity<double, CriticalPointPredicate<2, double>>(mesh, pred, data, vars);
    }

    // 3. Levelset 3D Parity (m=1, d=4)
    {
        std::cout << "Case 3: Levelset 3D" << std::endl;
        const int DW = 16, DH = 16, DD = 16, DT = 5;
        ftk::ndarray<float> scalar({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
        for (int t = 0; t < DT; ++t) {
            float time = (float)t / (DT - 1);
            for (int z = 0; z < DD; ++z) for (int y = 0; y < DH; ++y) for (int x = 0; x < DW; ++x) {
                float dx = (float)x - DW/2, dy = (float)y - DH/2, dz = (float)z - DD/2;
                scalar.f(x, y, z, t) = std::sqrt(dx*dx + dy*dy + dz*dz) - (5.0f + time);
            }
        }
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});
        ContourPredicate<float> pred; pred.var_name = "S"; pred.threshold = 0.0f;
        std::map<std::string, ftk::ndarray<float>> data = {{"S", scalar}};
        std::vector<std::string> vars = {"S"};
        verify_parity<float, ContourPredicate<float>>(mesh, pred, data, vars);
    }

    // 4. Critical Point 3D Parity (m=3, d=4)
    {
        std::cout << "Case 4: Critical Point 3D" << std::endl;
        const int DW = 16, DH = 16, DD = 16, DT = 5;
        ftk::ndarray<float> u({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT}), v({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT}), w({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT}), s({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
        for (int t = 0; t < DT; ++t) {
            float cx = 8.0f + t * 0.1f, cy = 8.0f, cz = 8.0f;
            for (int z = 0; z < DD; ++z) for (int y = 0; y < DH; ++y) for (int x = 0; x < DW; ++x) {
                u.f(x, y, z, t) = (float)x - cx; v.f(x, y, z, t) = (float)y - cy; w.f(x, y, z, t) = (float)z - cz; s.f(x, y, z, t) = 0.0f;
            }
        }
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});
        CriticalPointPredicate<3, float> pred; pred.var_names[0] = "U"; pred.var_names[1] = "V"; pred.var_names[2] = "W";
        std::map<std::string, ftk::ndarray<float>> data = {{"U", u}, {"V", v}, {"W", w}, {"S", s}};
        std::vector<std::string> vars = {"U", "V", "W", "S"};
        verify_parity<float, CriticalPointPredicate<3, float>>(mesh, pred, data, vars);
    }

    // 5. Fiber 3D Parity (m=2, d=4)
    {
        std::cout << "Case 5: Fiber 3D" << std::endl;
        const int DW = 16, DH = 16, DD = 16, DT = 5;
        ftk::ndarray<double> s1({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
        ftk::ndarray<double> s2({(size_t)DW, (size_t)DH, (size_t)DD, (size_t)DT});
        for (int t = 0; t < DT; ++t) {
            double c1x = 8.0, c1y = 8.0, c1z = 8.0, r1 = 5.0;
            double c2x = 11.0, c2y = 8.0 + t * 0.2, c2z = 8.0, r2 = 4.0;
            for (int z = 0; z < DD; ++z) for (int y = 0; y < DH; ++y) for (int x = 0; x < DW; ++x) {
                double d1 = std::sqrt(std::pow(x-c1x, 2) + std::pow(y-c1y, 2) + std::pow(z-c1z, 2));
                double d2 = std::sqrt(std::pow(x-c2x, 2) + std::pow(y-c2y, 2) + std::pow(z-c2z, 2));
                s1.f(x, y, z, t) = d1 - r1;
                s2.f(x, y, z, t) = d2 - r2;
            }
        }
        auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{(uint64_t)DW, (uint64_t)DH, (uint64_t)DD, (uint64_t)DT});
        FiberPredicate<double> pred; pred.var_names[0] = "S1"; pred.var_names[1] = "S2";
        std::map<std::string, ftk::ndarray<double>> data = {{"S1", s1}, {"S2", s2}};
        std::vector<std::string> vars = {"S1", "S2"};
        verify_parity<double, FiberPredicate<double>>(mesh, pred, data, vars);
    }
}

int main() {
    std::cout << "Running CPU/GPU Parity tests..." << std::endl;
    test_parity();
    std::cout << "Summary: " << passed_tests << "/" << total_tests << " tests passed." << std::endl;
    return (passed_tests == total_tests) ? 0 : 1;
}
