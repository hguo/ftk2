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

    // Create a 3D mesh - high resolution for single continuous curve
    const int N = 16;  // High resolution to capture full circle
    auto mesh = std::make_shared<RegularSimplicialMesh>(std::vector<uint64_t>{N, N, N});

    // Create combined vector field: [ux, uy, uz, vx, vy, vz]
    ftk::ndarray<double> uv({6, N, N, N});  // [6 components][x][y][z]

    // Generate synthetic field with explicit PV curve design
    // Strategy: Make u and v nearly parallel everywhere, with cross product
    // vanishing exactly on a known circle
    double cx = N / 2.0 + 0.1;  // Slight offset to avoid symmetry
    double cy = N / 2.0 + 0.1;
    double cz = N / 2.0;
    double radius = N / 3.0;  // Scale radius with mesh size

    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                double dx = x - cx;
                double dy = y - cy;
                double dz = z - cz;
                double r_xy = std::sqrt(dx*dx + dy*dy) + 0.01;

                // Angle in xy plane
                double theta = std::atan2(dy, dx);

                // Base vectors that vary smoothly
                double ux = std::cos(theta + 0.1*dz);
                double uy = std::sin(theta + 0.1*dz);
                double uz = 0.3 + 0.1*std::cos(2*theta);

                // V = U + correction that vanishes on the circle
                double dist_to_circle = r_xy - radius;
                double dist_to_plane = dz;

                // Perpendicular perturbation
                double vx = ux + dist_to_circle * dx/r_xy + 0.1*dist_to_plane*dx/r_xy;
                double vy = uy + dist_to_circle * dy/r_xy + 0.1*dist_to_plane*dy/r_xy;
                double vz = uz + 0.2*dist_to_plane + 0.1*dist_to_circle;

                uv.f(0, x, y, z) = ux;
                uv.f(1, x, y, z) = uy;
                uv.f(2, x, y, z) = uz;
                uv.f(3, x, y, z) = vx;
                uv.f(4, x, y, z) = vy;
                uv.f(5, x, y, z) = vz;
            }
        }
    }

    std::cout << "Fields generated: Synthetic PV field" << std::endl;
    std::cout << "PV locus: circle at z=" << cz << ", radius=" << radius << std::endl;

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

    // Check how many curves truly pass through tet INTERIOR (all 4 barycentric coords > 0)
    int non_degen_count = 0;
    if (!pred.curve_segments.empty()) {
        for (const auto& seg : pred.curve_segments) {
            // Sample curve at multiple points to check if any point is in tet interior
            bool has_interior_point = false;
            int n_samples = 10;
            double lambda_range = seg.lambda_max - seg.lambda_min;

            for (int i = 0; i < n_samples; ++i) {
                double lambda = seg.lambda_min + (double)i / (n_samples - 1) * lambda_range;
                auto bary = seg.get_barycentric(lambda);

                // Check if all 4 barycentric coords are positive (in interior)
                bool all_positive = true;
                for (int j = 0; j < 4; ++j) {
                    if (bary[j] <= 1e-10) {  // Essentially zero or negative
                        all_positive = false;
                        break;
                    }
                }

                if (all_positive) {
                    has_interior_point = true;
                    break;
                }
            }

            if (has_interior_point) non_degen_count++;
        }
        std::cout << "  " << non_degen_count << " / " << pred.curve_segments.size()
                  << " curves pass through tet INTERIOR (all 4 bary coords > 0)" << std::endl;
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
            // Check if curve passes through tet interior (all 4 bary coords > 0)
            bool has_interior_point = false;
            int n_check = 10;
            double lambda_range = seg.lambda_max - seg.lambda_min;

            for (int i = 0; i < n_check; ++i) {
                double lambda = seg.lambda_min + (double)i / (n_check - 1) * lambda_range;
                auto bary = seg.get_barycentric(lambda);

                bool all_positive = true;
                for (int j = 0; j < 4; ++j) {
                    if (bary[j] <= 1e-10) {
                        all_positive = false;
                        break;
                    }
                }

                if (all_positive) {
                    has_interior_point = true;
                    break;
                }
            }

            // Skip curves on tet boundary (not in interior)
            if (!has_interior_point) {
                seg_count++;
                continue;
            }

            // Sample the curve at regular intervals
            int n_samples = 20;
            // lambda_range already computed above

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

        std::cout << "  Wrote " << non_degen_written << " interior curves to: exactpv_curves.vtp" << std::endl;
    } else {
        std::cout << "  Note: No curves pass through tet interiors (all on boundaries)" << std::endl;
        std::cout << "  This field configuration only creates PV curves on tet faces/edges." << std::endl;
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
