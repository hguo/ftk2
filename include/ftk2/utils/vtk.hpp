#pragma once

#include <ftk2/core/complex.hpp>
#include <ftk2/core/mesh.hpp>
#include <string>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkCellType.h>

namespace ftk2 {

/**
 * @brief Internal helper to convert FeatureComplex to VTK points and attributes.
 */
inline vtkSmartPointer<vtkPoints> complex_to_vtk_points(const FeatureComplex& complex, const Mesh& mesh, vtkPointData* pointData) {
    auto points = vtkSmartPointer<vtkPoints>::New();
    
    auto scalar_data = vtkSmartPointer<vtkFloatArray>::New();
    scalar_data->SetName("Scalar");
    auto track_id_data = vtkSmartPointer<vtkIntArray>::New();
    track_id_data->SetName("TrackID");
    auto type_data = vtkSmartPointer<vtkIntArray>::New();
    type_data->SetName("Type");
    auto time_data = vtkSmartPointer<vtkFloatArray>::New();
    time_data->SetName("Time");

    for (auto const& el : complex.vertices) {
        std::vector<double> phys_pos(mesh.get_total_dimension(), 0.0);
        for (int i = 0; i <= el.simplex.dimension; ++i) {
            auto v_coords = mesh.get_vertex_coordinates(el.simplex.vertices[i]);
            for (int j = 0; j < phys_pos.size(); ++j) {
                phys_pos[j] += (double)el.barycentric_coords[0][i] * v_coords[j];
            }
        }
        
        double p[3] = {0, 0, 0};
        for (int j = 0; j < 3 && j < phys_pos.size(); ++j) p[j] = phys_pos[j];
        
        points->InsertNextPoint(p);
        scalar_data->InsertNextValue(el.scalar);
        track_id_data->InsertNextValue((int)el.track_id);
        type_data->InsertNextValue((int)el.type);
        time_data->InsertNextValue((float)phys_pos.back());
    }

    if (pointData) {
        pointData->AddArray(scalar_data);
        pointData->AddArray(track_id_data);
        pointData->AddArray(type_data);
        pointData->AddArray(time_data);
    }
    return points;
}

/**
 * @brief Write a feature complex to a VTK PolyData file (.vtp).
 */
inline void write_complex_to_vtp(const FeatureComplex& complex, const Mesh& mesh, const std::string& filename, int target_dim = -1) {
    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(complex_to_vtk_points(complex, mesh, polydata->GetPointData()));

    auto verts = vtkSmartPointer<vtkCellArray>::New();
    auto lines = vtkSmartPointer<vtkCellArray>::New();
    auto polys = vtkSmartPointer<vtkCellArray>::New();

    for (auto const& conn : complex.connectivity) {
        if (target_dim != -1 && conn.dimension != target_dim) continue;

        int n_pts = conn.dimension + 1;
        for (size_t i = 0; i < conn.indices.size(); i += n_pts) {
            std::vector<vtkIdType> cell_pts(n_pts);
            for (int j = 0; j < n_pts; ++j) cell_pts[j] = conn.indices[i + j];
            
            if (conn.dimension == 0) verts->InsertNextCell(1, cell_pts.data());
            else if (conn.dimension == 1) lines->InsertNextCell(2, cell_pts.data());
            else if (conn.dimension == 2) polys->InsertNextCell(3, cell_pts.data());
        }
    }

    if (target_dim == -1 || target_dim == 0) polydata->SetVerts(verts);
    if (target_dim == -1 || target_dim == 1) polydata->SetLines(lines);
    if (target_dim == -1 || target_dim == 2) polydata->SetPolys(polys);

    auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(polydata);
    writer->SetDataModeToAppended();
    writer->SetEncodeAppendedData(0);
    writer->SetCompressorTypeToNone();
    writer->Write();
}

/**
 * @brief Write a feature complex to a VTK UnstructuredGrid file (.vtu).
 */
inline void write_complex_to_vtu(const FeatureComplex& complex, const Mesh& mesh, const std::string& filename, int target_dim = -1) {
    auto grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(complex_to_vtk_points(complex, mesh, grid->GetPointData()));

    for (auto const& conn : complex.connectivity) {
        if (target_dim != -1 && conn.dimension != target_dim) continue;

        int vtk_type = VTK_EMPTY_CELL;
        int n_pts = conn.dimension + 1;

        switch (conn.dimension) {
            case 0: vtk_type = VTK_VERTEX; break;
            case 1: vtk_type = VTK_LINE; break;
            case 2: vtk_type = VTK_TRIANGLE; break;
            case 3: vtk_type = VTK_TETRA; break;
            default: continue; 
        }

        for (size_t i = 0; i < conn.indices.size(); i += n_pts) {
            std::vector<vtkIdType> cell_pts(n_pts);
            for (int j = 0; j < n_pts; ++j) cell_pts[j] = conn.indices[i + j];
            grid->InsertNextCell(vtk_type, n_pts, cell_pts.data());
        }
    }

    auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToAscii();
    writer->Write();
}

} // namespace ftk2
