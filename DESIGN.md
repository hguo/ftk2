# FTK2: Technical Design Specification

This document provides a deep dive into the engineering principles and algorithmic strategies of FTK2.

## 1. The Unified Mesh Interface (`Mesh`)

The `Mesh` class is the foundational topological abstraction of FTK2. It defines the "Space" in which features live, regardless of the underlying grid type.

### 1.1 Regular Simplicial Mesh (Implicit)
For regular hypercubic grids, we implement an implicit simplicial subdivision based on Kuhn's Triangulation.
*   **Implicit Connectivity**: Vertices, edges, and faces are computed using bitwise operations and permutations of grid indices.
*   **Memory Efficiency**: Zero bytes are stored for the mesh topology; it is entirely algorithmic.
*   **Dimensionality**: Supports $n$-dimensional grids ($d$) plus a temporal dimension ($+1$).

### 1.2 Extruded & Deforming Simplicial Mesh
For unstructured meshes, we "extrude" the spatial mesh into spacetime.
*   **Recursive Extrusion**: Supports chaining (e.g., 2D Mesh -> 3D Volume -> 4D Spacetime).
*   **Deforming Spacetime**: Vertex coordinates can change at each layer (e.g., following simulation grid movement), as detailed in arXiv:2309.02677.
*   **Topological Consistency**: Ensures that features crossing from one spatial simplex to another are correctly linked across time.

## 2. The Unified Simplicial Engine

The engine unifies extraction and tracking into a single **Graph-based Connected Component Labeling (CCL)** pass.

### 2.1 Tracing Intersections in Any Dimension
Features are defined by a set of $m$ equations. The engine searches for these features in all $m$-dimensional simplices. This covers zero-crossing features as well as manifold types like **Parallel Vectors**.

| Feature Type | Equations ($m$) | Spacetime Dim ($d+1$) | Manifold Dim ($k = d+1-m$) | Resulting Track |
| :--- | :---: | :---: | :---: | :--- |
| **2D Critical Point** | 2 | 3 ($2D+T$) | 1 | **1D Curve (Line)** |
| **3D Critical Point** | 3 | 4 ($3D+T$) | 1 | **1D Curve (Line)** |
| **2D Parallel Vectors** | 1 | 3 ($2D+T$) | 2 | **2D Surface (Sheet)** |
| **3D Parallel Vectors** | 2 | 4 ($3D+T$) | 2 | **2D Surface (Sheet)** |
| **Magnetic Flux Vortex** | 2 | 4 ($3D+T$) | 2 | **2D Surface (Sheet)** |
| **3D Isosurface** | 1 | 4 ($3D+T$) | 3 | **3D Volume** |

### 2.2 Connecting Features Across Spacetime
Tracking is the transformation of a **Spacetime Mesh ($M$)** into a **Feature Simplicial Complex ($C$)**.
*   **Discovery**: Identifying $m$-simplices that contain features (these become the vertices of $C$).
*   **Connecting**: Linking $m$-simplices through their shared $(m+1)$-dimensional cofaces (these become the connectivity of $C$).
*   **Distributed Parallelism**: Future integration of asynchronous label propagation for extreme-scale tracking (arXiv:2003.02351).

## 3. Predicate Layer (Mathematical Kernels)

Predicates are stateless, `__host__ __device__` functors that perform the local math.

### 3.1 Universal Zero-Crossing Solver
Solves $f(x)=0$ on a $k$-simplex.
1.  **Input**: $k+1$ values at vertices, each of dimension $k$.
2.  **Logic**: Solve the $(k+1) \times (k+1)$ linear system for barycentric coordinates $\lambda_i$.
3.  **Output**: If $0 \le \lambda_i \le 1$ for all $i$, a feature exists at $x = \sum \lambda_i v_i$.

### 3.2 Exact Analytical Parallel Vectors (ExactPV)
Implements the cubic rational solver from arXiv:2107.02708.
*   **Input**: Two vector fields $V_1, V_2$ on a $3$-simplex.
*   **Logic**: Finds intersections of $V_1 \times V_2 = 0$ using the **ExactPV** analytical formulation.
*   **Multi-Crossing**: Can return multiple line segments per simplex, each tracked independently by the engine.

## 4. Data Abstraction Layer (`ndarray`)

FTK2 is built around the `ftk::ndarray` library for high-performance IO.

*   **`ftk::ndarray_stream<T>`**: Manages a sliding window of time steps, loading only necessary buffers into memory.
*   **Zero-Copy Design**: Algorithms operate on `ndarray` views to avoid expensive memory allocations.
*   **Domain Decomposition**: Native support for ghost cells and MPI-based partitioning.

## 5. GPU Acceleration Strategy

FTK2 avoids code duplication by using **Static Polymorphism**:
*   **Kernel Templates**: A single `__global__` kernel iterates over the mesh, templated on the `MeshType` and `PredicateType`.
*   **Device-Only Mesh Structs**: Lightweight PODs passed to kernels for fast coordinate lookups.
*   **Parallel CCL**: High-performance Union-Find implementation on the GPU.

## 6. Slicing and Visualization

The resulting "Tracks" are spacetime manifolds.
*   **Slicing**: The manifold (dimension $k$) is intersected with the hyperplane $t=T$ to produce $(k-1)$-dimensional features.
*   **ParaView Integration**: (High Priority) Server-side filters and client-side plugins for interactive exploration.
