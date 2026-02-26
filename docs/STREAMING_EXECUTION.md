# Streaming Execution Design

## Problem
Current implementation loads ALL timesteps into memory, which is infeasible for large datasets.

## Requirements
1. Hold only 2 consecutive timesteps in memory at once
2. Process spacetime slices incrementally
3. Streaming-aware manifold stitching for real-time labeling
4. Works with ndarray stream API

## Architecture

### Current Flow (Memory-Intensive)
```
1. Load ALL timesteps into memory
2. Create spacetime mesh for entire time range
3. Execute on full 4D mesh
4. Stitch all manifolds at once
```

### Streaming Flow (Memory-Efficient)
```
For each pair (t, t+1):
  1. Load timestep t and t+1
  2. Create spacetime slice mesh [t, t+1]
  3. Extract features in slice
  4. Incrementally stitch to existing trajectories
  5. Release timestep t, keep t+1 for next iteration
```

## Implementation Strategy

### Phase 1: Streaming Data Iterator
- Create `StreamingDataLoader` that yields consecutive timestep pairs
- Wraps ndarray stream with windowed reading

### Phase 2: Sliced Engine Execution
- Modify `SimplicialEngine` to work on temporal slices
- Execute on (spatial_mesh × [t, t+1]) instead of full spacetime

### Phase 3: Incremental Manifold Stitching
- Track connectivity across slice boundaries
- Build trajectory graph incrementally
- Real-time labeling as data arrives

## Data Structures

```cpp
// Streaming data pair
struct TimestepPair {
    int t0, t1;
    std::map<std::string, ftk::ndarray<T>> data_t0;
    std::map<std::string, ftk::ndarray<T>> data_t1;
};

// Streaming iterator
class StreamingDataLoader {
    ftk::stream stream;
    int current_timestep;
    std::map<std::string, ftk::ndarray<T>> cached_next;

    std::optional<TimestepPair> next_pair();
};
```

## Memory Complexity
- Before: O(N_spatial × N_timesteps)
- After: O(N_spatial × 2) = O(N_spatial)

## API Impact
- High-level API: No change to YAML config
- Engine: Add `execute_streaming()` method
- Results: Trajectories built incrementally
