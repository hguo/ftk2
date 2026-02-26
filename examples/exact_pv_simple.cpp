#include <ftk2/core/mesh.hpp>
#include <ftk2/core/engine.hpp>
#include <ftk2/core/predicate.hpp>
#include <ftk2/utils/vtk.hpp>
#include <ndarray/ndarray.hh>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyLine.h>
#include <vtkFloatArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSmartPointer.h>

using namespace ftk2;

int main() {
    std::cout << "ExactPV Simple Example - Parallel Vector Detection" << std::endl;

    // Create a 3D mesh
    const int N = 6;  // Medium size for better interior coverage
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{N, N, N});

    // Create combined vector field: [ux, uy, uz, vx, vy, vz]
    ftk::ndarray<double> uv({6, N, N, N});  // [6 components][x][y][z]

    // Generate field with simple known PV curve
    // Approach: Create u and v that are nearly parallel everywhere,
    // but exactly parallel along a specific curve
    double cx = N / 2.0, cy = N / 2.0;

    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double dz = z - N/2.0;

                // Base field U - varies smoothly
                double r_xy = std::sqrt(dx*dx + dy*dy) + 0.1;
                uv.f(0, x, y, z) = dx / r_xy;
                uv.f(1, x, y, z) = dy / r_xy;
                uv.f(2, x, y, z) = (double)z / N;

                // V = U + small perpendicular perturbation
                // The perturbation vanishes at a specific curve
                double perturb_x = dy * (dx*dx + dy*dy - 1.0) * 0.1;
                double perturb_y = -dx * (dx*dx + dy*dy - 1.0) * 0.1;
                uv.f(3, x, y, z) = uv.f(0, x, y, z) + perturb_x;
                uv.f(4, x, y, z) = uv.f(1, x, y, z) + perturb_y;
                uv.f(5, x, y, z) = uv.f(2, x, y, z);
            }
        }
    }

    std::cout << "Fields generated: U radial, V = U + perturbation" << std::endl;
    std::cout << "PV locus is a circle at radius ~1 in x-y plane" << std::endl;

    // Prepare data map
    std::map<std::string, ftk::ndarray<double>> data;
    data["uv"] = uv;

    // Set up ExactPV predicate
    ExactPVPredicate<double> pred;
    pred.vector_var_name = "uv";

    // Create engine
    SimplicialEngine<double, ExactPVPredicate<double>> engine(mesh, pred);

    // Execute extraction for puncture points (from triangles)
    std::cout << "Extracting parallel vector puncture points from triangles..." << std::endl;
    engine.execute(data, {"uv"});

    // Get puncture point results
    auto complex = engine.get_complex();

    // Extract parametric curves from tetrahedra
    // Note: Each curve segment represents the PV curve within a single tetrahedron.
    // Multiple segments may be part of one connected curve but aren't stitched yet.
    std::cout << "\nExtracting parallel vector curves from tetrahedra..." << std::endl;
    pred.extract_curves_from_tets(mesh.get(), data);

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Found " << complex.vertices.size() << " parallel vector puncture points (from triangles)" << std::endl;
    std::cout << "  Found " << pred.curve_segments.size() << " parametric curve segments (from tetrahedra)" << std::endl;

    // Display first few puncture points
    int display_count = std::min(5, (int)complex.vertices.size());
    for (int i = 0; i < display_count; ++i) {
        const auto& v = complex.vertices[i];
        std::cout << "  Point " << i << ": simplex_dim=" << v.simplex.dimension
                  << ", vertices=[" << v.simplex.vertices[0] << ","
                  << v.simplex.vertices[1] << "," << v.simplex.vertices[2] << "]"
                  << ", barycentric=(" << v.barycentric_coords[0][0] << ", "
                  << v.barycentric_coords[0][1] << ", "
                  << v.barycentric_coords[0][2] << ")"
                  << ", lambda=" << v.scalar << std::endl;
    }

    // Display curve segment info
    if (!pred.curve_segments.empty()) {
        std::cout << "\nCurve segments:" << std::endl;
        display_count = std::min(3, (int)pred.curve_segments.size());
        for (int i = 0; i < display_count; ++i) {
            const auto& seg = pred.curve_segments[i];
            std::cout << "  Curve " << i << ": simplex=" << seg.simplex_id
                      << ", lambda_range=[" << seg.lambda_min << ", " << seg.lambda_max << "]"
                      << ", critical_points=" << seg.critical_points.size() << std::endl;
        }
    }

    // Write puncture points to VTP (use ASCII for debugging)
    std::cout << "\nWriting output files..." << std::endl;
    ftk2::write_complex_to_vtp(complex, *mesh, "exactpv_punctures.vtp", -1, true);
    std::cout << "  Wrote puncture points to: exactpv_punctures.vtp" << std::endl;

    // First, check how many curves are truly non-degenerate (have multiple non-zero P)
    int non_degen_count = 0;
    if (!pred.curve_segments.empty()) {
        for (const auto& seg : pred.curve_segments) {
            int non_zero_P = 0;
            for (int p = 0; p < 4; ++p) {
                bool p_nonzero = false;
                for (int c = 0; c <= 3; ++c) {
                    if (std::abs(seg.P[p].coeffs[c]) > 1e-10) {
                        p_nonzero = true;
                        break;
                    }
                }
                if (p_nonzero) non_zero_P++;
            }
            if (non_zero_P >= 2) non_degen_count++;  // At least 2 vertices contribute
        }
        std::cout << "  " << non_degen_count << " / " << pred.curve_segments.size() << " curves have ≥2 non-zero barycentric coords" << std::endl;
    }

    // Write curve segments to VTP (only if non-degenerate curves exist)
    if (non_degen_count > 0) {
        auto polydata = vtkSmartPointer<vtkPolyData>::New();
        auto points = vtkSmartPointer<vtkPoints>::New();
        auto lines = vtkSmartPointer<vtkCellArray>::New();
        auto lambda_data = vtkSmartPointer<vtkFloatArray>::New();
        lambda_data->SetName("Lambda");
        auto simplex_data = vtkSmartPointer<vtkIntArray>::New();
        simplex_data->SetName("SimplexID");

        vtkIdType point_id = 0;
        int seg_count = 0;
        int non_degen_written = 0;
        for (const auto& seg : pred.curve_segments) {
            // Check if curve has at least 2 non-zero barycentric coordinates
            int non_zero_P = 0;
            for (int p = 0; p < 4; ++p) {
                bool p_nonzero = false;
                for (int c = 0; c <= 3; ++c) {
                    if (std::abs(seg.P[p].coeffs[c]) > 1e-10) {
                        p_nonzero = true;
                        break;
                    }
                }
                if (p_nonzero) non_zero_P++;
            }

            // Skip degenerate curves (stuck at single vertex)
            if (non_zero_P < 2) {
                seg_count++;
                continue;
            }

            // Sample the curve at regular intervals
            int n_samples = 20;
            double lambda_range = seg.lambda_max - seg.lambda_min;

            vtkSmartPointer<vtkPolyLine> polyLine = vtkSmartPointer<vtkPolyLine>::New();
            polyLine->GetPointIds()->SetNumberOfIds(n_samples);

            // Debug first non-degenerate curve
            if (non_degen_written == 0) {
                std::cout << "  Debug curve 0:" << std::endl;
                std::cout << "    Tet vertices: ";
                for (int v = 0; v < 4; ++v) {
                    std::cout << "(" << seg.tet_vertices[v][0] << "," << seg.tet_vertices[v][1] << "," << seg.tet_vertices[v][2] << ") ";
                }
                std::cout << std::endl;
                std::cout << "    Lambda range: [" << seg.lambda_min << ", " << seg.lambda_max << "]" << std::endl;
                std::cout << "    Q(λ) = " << seg.Q.coeffs[0] << " + " << seg.Q.coeffs[1] << "λ + "
                          << seg.Q.coeffs[2] << "λ² + " << seg.Q.coeffs[3] << "λ³" << std::endl;
                for (int p = 0; p < 4; ++p) {
                    std::cout << "    P" << p << "(λ) = " << seg.P[p].coeffs[0] << " + " << seg.P[p].coeffs[1] << "λ + "
                              << seg.P[p].coeffs[2] << "λ² + " << seg.P[p].coeffs[3] << "λ³" << std::endl;
                }
            }

            for (int i = 0; i < n_samples; ++i) {
                double lambda = seg.lambda_min + (double)i / (n_samples - 1) * lambda_range;

                // Get physical (x,y,z) coordinates at this parameter value
                auto pos = seg.get_physical_coords(lambda);

                if (non_degen_written == 0 && i < 3) {
                    auto bary = seg.get_barycentric(lambda);
                    std::cout << "    Sample " << i << ": lambda=" << lambda << ", bary=("
                              << bary[0] << "," << bary[1] << "," << bary[2] << "," << bary[3]
                              << "), pos=(" << pos[0] << "," << pos[1] << "," << pos[2] << ")" << std::endl;
                }

                points->InsertNextPoint(pos[0], pos[1], pos[2]);
                lambda_data->InsertNextValue((float)lambda);
                simplex_data->InsertNextValue(seg.simplex_id);
                polyLine->GetPointIds()->SetId(i, point_id++);
            }

            seg_count++;

            lines->InsertNextCell(polyLine);
            non_degen_written++;
            seg_count++;
        }

        polydata->SetPoints(points);
        polydata->SetLines(lines);
        polydata->GetPointData()->AddArray(lambda_data);
        polydata->GetPointData()->AddArray(simplex_data);

        auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
        writer->SetFileName("exactpv_curves.vtp");
        writer->SetInputData(polydata);
        writer->SetDataModeToAscii();
        writer->Write();

        std::cout << "  Wrote curve segments to: exactpv_curves.vtp" << std::endl;
    } else {
        std::cout << "  Note: All curve segments are degenerate (collapse to points)" << std::endl;
        std::cout << "  This field configuration doesn't create curves through tet interiors." << std::endl;
    }

    std::cout << "\nDone!" << std::endl;
    std::cout << "Output files generated in: " << std::filesystem::current_path() << std::endl;
    if (non_degen_count > 0) {
        std::cout << "  - exactpv_punctures.vtp: " << complex.vertices.size() << " puncture points, "
                  << complex.connectivity.size() << " trajectory component(s)" << std::endl;
        std::cout << "  - exactpv_curves.vtp: " << non_degen_count << " non-degenerate PV curves through tet interiors" << std::endl;
    } else {
        std::cout << "  - exactpv_punctures.vtp: " << complex.vertices.size() << " puncture points" << std::endl;
        std::cout << "  - Note: No non-degenerate curves (PV locus only touches tet boundaries)" << std::endl;
    }
    return 0;
}
