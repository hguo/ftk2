# ParaView Integration Strategy

## Goal
Enable interactive feature tracking directly in ParaView with FTK2 as the backend.

## Architecture

### 1. VTK Filter Architecture
```
vtkFTK2FeatureTracker (vtkPolyDataAlgorithm)
  ├─ vtkFTK2CriticalPointTracker
  ├─ vtkFTK2LevelsetTracker
  ├─ vtkFTK2FiberTracker
  └─ vtkFTK2ParticleTracer
```

### 2. Plugin Components

**Core Filter (`vtkFTK2FeatureTracker`):**
- Input: `vtkUnstructuredGrid` (spatial) or `vtkTemporalDataSet` (spacetime)
- Output: `vtkPolyData` (trajectories as lines/tubes)
- Parameters:
  - Feature type (critical points, levelsets, fibers)
  - Field selection (vector/scalar)
  - SoS quantization factor
  - GPU acceleration toggle

**Property Panel:**
- Feature type dropdown
- Field array selection
- Threshold/parameter controls
- Advanced options (SoS, threading)

### 3. Implementation Steps

#### Step 3.1: Create VTK Filter Base
```cpp
// paraview/vtkFTK2FeatureTracker.h
class VTK_EXPORT vtkFTK2FeatureTracker : public vtkPolyDataAlgorithm {
public:
  static vtkFTK2FeatureTracker* New();
  vtkTypeMacro(vtkFTK2FeatureTracker, vtkPolyDataAlgorithm);

  // Feature type selection
  vtkSetMacro(FeatureType, int);
  vtkGetMacro(FeatureType, int);

  // Field array name
  vtkSetStringMacro(VectorFieldName);
  vtkGetStringMacro(VectorFieldName);

  // GPU acceleration
  vtkSetMacro(UseGPU, bool);
  vtkGetMacro(UseGPU, bool);

protected:
  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int FillInputPortInformation(int port, vtkInformation* info) override;

private:
  int FeatureType;
  char* VectorFieldName;
  bool UseGPU;
};
```

#### Step 3.2: Bridge FTK2 Engine
```cpp
// Convert vtkUnstructuredGrid to FTK2 mesh
auto mesh = ConvertVTKToFTK2(input);

// Extract field data
ndarray<double> field = ExtractVTKField(input, VectorFieldName);

// Run FTK2 engine
CriticalPointPredicate<3, double> pred;
SimplicialEngine<double, CriticalPointPredicate<3, double>> engine(mesh, pred);
engine.execute(data_map);

// Convert results to vtkPolyData
auto complex = engine.get_complex();
output = ConvertFTK2ToVTK(complex, mesh);
```

#### Step 3.3: Create ParaView XML
```xml
<ServerManagerConfiguration>
  <ProxyGroup name="filters">
    <SourceProxy name="FTK2FeatureTracker" class="vtkFTK2FeatureTracker">
      <Documentation>
        Track features in vector/scalar fields using FTK2.
      </Documentation>

      <IntVectorProperty name="FeatureType"
                        command="SetFeatureType"
                        default_values="0">
        <EnumerationDomain name="enum">
          <Entry value="0" text="Critical Points"/>
          <Entry value="1" text="Levelsets"/>
          <Entry value="2" text="Fibers"/>
        </EnumerationDomain>
      </IntVectorProperty>

      <StringVectorProperty name="VectorFieldName"
                           command="SetVectorFieldName"
                           number_of_elements="1">
        <ArrayListDomain name="array_list" attribute_type="Vectors">
          <RequiredProperties>
            <Property name="Input" function="Input"/>
          </RequiredProperties>
        </ArrayListDomain>
      </StringVectorProperty>

      <IntVectorProperty name="UseGPU"
                        command="SetUseGPU"
                        default_values="0">
        <BooleanDomain name="bool"/>
      </IntVectorProperty>
    </SourceProxy>
  </ProxyGroup>
</ServerManagerConfiguration>
```

#### Step 3.4: Build System
```cmake
# paraview/CMakeLists.txt
find_package(ParaView REQUIRED)

paraview_add_plugin(FTK2Plugin
  VERSION "1.0"
  SERVER_MANAGER_XML vtkFTK2FeatureTracker.xml
  SERVER_MANAGER_SOURCES vtkFTK2FeatureTracker.cxx
  MODULES FTK2::ftk2
)
```

### 4. Testing Strategy
- Test with ParaView's built-in datasets
- Validate against FTK ParaView plugin results
- Performance profiling with large datasets
- User testing with domain scientists

### 5. Documentation
- Tutorial videos
- Example workflows
- API documentation
- Troubleshooting guide

## Timeline
- Week 1-2: VTK filter base + data conversion
- Week 3-4: ParaView plugin integration + XML
- Week 5-6: Testing and optimization
- Week 7-8: Documentation and examples

## Dependencies
- VTK >= 9.0
- ParaView >= 5.10
- CUDA (optional, for GPU acceleration)
