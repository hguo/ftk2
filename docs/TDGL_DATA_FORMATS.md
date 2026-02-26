# TDGL Data Format Handling

## Overview

TDGL (Time-Dependent Ginzburg-Landau) simulations output complex order parameter data in various formats. FTK2 currently expects data in **Cartesian form (re/im)** but will support **polar form (rho/phi)** in the future.

## Current Status

### Supported Format

**Cartesian (re + im):**
```yaml
tracking:
  feature: tdgl_vortex

  input:
    type: complex
    variables: [re, im]  # Real and imaginary parts

  data:
    source: stream
    stream_yaml: tdgl_data.yaml
```

FTK2 expects:
- `re`: Real part of ψ (order parameter)
- `im`: Imaginary part of ψ

Arrays should have shape: `[nx, ny, nt]` for 2D spatial, or `[nx, ny, nz, nt]` for 3D spatial.

### Legacy Format (BDAT)

Legacy FTK and some TDGL simulations use **BDAT format**, which may contain:
- Both `re/im` and `rho/phi` (redundant storage)
- Only `rho/phi` (needs conversion)
- Custom binary layouts

**Note:** BDAT I/O handling is **outside FTK2's scope** and should be handled by the **ndarray library** or preprocessing scripts.

## Future Support (Planned)

### Polar Form (rho + phi)

```yaml
tracking:
  feature: tdgl_vortex

  input:
    type: complex
    representation: polar  # NEW: specify polar vs cartesian
    variables: [rho, phi]  # Amplitude and phase

  data:
    source: stream
    stream_yaml: tdgl_data.yaml
```

FTK2 will automatically convert:
```
re = rho * cos(phi)
im = rho * sin(phi)
```

This conversion will happen in the **preprocessing stage** (`feature_tracker.cpp:preprocess_data`).

### Both Formats

```yaml
input:
  type: complex
  representation: both  # Both forms available
  variables:
    cartesian: [re, im]
    polar: [rho, phi]
```

FTK2 will prefer the form that's already available (avoid redundant computation).

## Data Loading Strategy

### Recommended Workflow

1. **BDAT files → ndarray conversion** (external tool or ndarray library):
   ```bash
   # Convert BDAT to ndarray stream YAML
   bdat_to_stream tdgl_output.bdat > tdgl_stream.yaml
   ```

2. **FTK2 loads via ndarray stream**:
   ```yaml
   data:
     source: stream
     stream_yaml: tdgl_stream.yaml
   ```

3. **FTK2 performs tracking**:
   - No BDAT-specific code in FTK2
   - Clean separation of concerns

### Why This Design?

**Separation of concerns:**
- **ndarray library**: Handles all I/O formats (BDAT, HDF5, NetCDF, etc.)
- **FTK2**: Focuses on feature tracking algorithms

**Benefits:**
- ✅ FTK2 stays format-agnostic
- ✅ Easy to add new formats (in ndarray, not FTK2)
- ✅ Cleaner codebase
- ✅ Better modularity

## Current Workaround

If you have BDAT files with only `rho/phi`:

### Option 1: Preprocessing Script

```python
import numpy as np
import yaml

# Read BDAT (custom parser)
rho, phi = read_bdat("tdgl_output.bdat")

# Convert to Cartesian
re = rho * np.cos(phi)
im = rho * np.sin(phi)

# Write ndarray stream YAML
write_ndarray_stream("tdgl_stream.yaml", {"re": re, "im": im})

# Then use FTK2
```

### Option 2: ndarray BDAT Reader (Future)

```yaml
# ndarray stream with BDAT source
stream:
  - name: re
    source: tdgl_output.bdat
    variable: re  # or conversion: rho*cos(phi)

  - name: im
    source: tdgl_output.bdat
    variable: im  # or conversion: rho*sin(phi)
```

The ndarray library would handle BDAT reading and optional conversions.

## Implementation Plan

### Phase 1: Polar Format Support (Later)

**File:** `src/high_level/feature_tracker.cpp`

```cpp
std::map<std::string, ftk::ndarray<T>> FeatureTrackerImpl<T>::preprocess_data(
    const std::map<std::string, ftk::ndarray<T>>& raw_data)
{
    // ... existing code ...

    } else if (config_.input.type == InputType::Complex) {
        // Check if polar form
        if (data.find("rho") != data.end() && data.find("phi") != data.end()) {
            std::cout << "  Converting polar (rho, phi) to Cartesian (re, im)..." << std::endl;

            const auto& rho = data.at("rho");
            const auto& phi = data.at("phi");

            ftk::ndarray<T> re, im;
            re.reshapef(rho.shape());
            im.reshapef(rho.shape());

            for (size_t i = 0; i < rho.nelem(); ++i) {
                re[i] = rho[i] * std::cos(phi[i]);
                im[i] = rho[i] * std::sin(phi[i]);
            }

            std::map<std::string, ftk::ndarray<T>> converted;
            converted["re"] = re;
            converted["im"] = im;

            // Also keep rho/phi for attributes
            converted["rho"] = rho;
            converted["phi"] = phi;

            return converted;
        }

        // Already in Cartesian form
        return raw_data;
    }
```

### Phase 2: BDAT Reader in ndarray (Later)

**Scope:** ndarray library, not FTK2

Add BDAT format support to ndarray's stream loader:
- Read BDAT binary format
- Extract variables (re, im, rho, phi)
- Optional on-the-fly conversions

## Summary

### Current (FTK2 v1.0)

- ✅ Supports `re/im` (Cartesian form)
- ⏳ Polar form `rho/phi` - **future work**
- ⏳ BDAT I/O - **handled by ndarray library** (external to FTK2)

### Recommended Now

1. Convert BDAT → Cartesian (re/im) using external tool
2. Load via ndarray stream
3. Track with FTK2

### Later (Future Phases)

1. **FTK2**: Add polar format preprocessing
2. **ndarray library**: Add BDAT reader
3. **Integration**: Seamless BDAT → FTK2 pipeline

## Questions?

- **BDAT format spec?** → See legacy FTK documentation
- **ndarray BDAT support?** → Feature request to ndarray maintainers
- **Conversion scripts?** → Check `ftk2/utils/` or legacy FTK tools

## See Also

- **TDGL Vortex Tracking**: `docs/TDGL_VORTEX_TRACKING.md`
- **ndarray library**: https://github.com/hguo/ndarray
- **Legacy FTK BDAT**: `/ftk/io/bdat.hh` (for reference)
