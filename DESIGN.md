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
For unstructured meshes (e.g., VTK tetrahedra, XGC, MPAS), FTK2 uses a hybrid explicit-implicit approach.
*   **Explicit Spatial Topology**: The base mesh stores raw connectivity and coordinate tables.
*   **Implicit Temporal Extrusion**: Spacetime connectivity is computed on-the-fly to minimize memory overhead. A spatial $d$-simplex is extruded into $d+1$ spacetime $(d+1)$-simplices.
*   **Unstructured GPU Strategy**: CUDA kernels utilize Read-Only caches for spatial connectivity while performing temporal extrusion entirely in-thread.
*   **Deforming Spacetime**: Vertex coordinates can change at each layer while maintaining fixed topology, as detailed in arXiv:2309.02677.

## 2. The Unified Simplicial Engine

The engine unifies extraction and tracking into a single pass that transforms a **Spacetime Mesh ($M$)** into a **Feature Simplicial Complex ($C$)**.

### 2.1 Tracing Intersections in Any Dimension
Features are defined by a set of $m$ equations. The engine searches for these features in all $m$-dimensional simplices.

| Feature Type | Plain Language | Equations ($m$) | Spacetime Dim ($d+1$) | Manifold Dim ($k = d+1-m$) | Resulting Track |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Critical Point** | Zeros of a vector field | $d$ | $d+1$ | 1 | **1D Curve (Line)** |
| **Levelset** | Points where field = constant | 1 | $d+1$ | $d$ | **$d$-D Manifold** |
| **Fiber** | Intersection of two levelsets | 2 | $d+1$ | $d-1$ | **$(d-1)$-D Manifold** |

**Specific Examples:**
*   **2D Levelset Track**: A moving curve in 2D space traces a **2D Surface** in 3D spacetime.
*   **3D Levelset Track**: A moving surface in 3D space traces a **3D Volume** in 4D spacetime.
*   **3D Fiber Track**: The intersection of two moving surfaces in 3D (a moving curve) traces a **2D Surface** in 4D spacetime.

### 2.2 Connecting Features Across Spacetime
*   **Extraction ($m$):** Nodes are extracted from all $m$-dimensional simplices of the mesh.
*   **Manifold Construction ($m+1, m+2, ...$):** Feature nodes are connected to form edges, triangles, and higher-order simplices based strictly on the topology of the mesh.
*   **Distributed Parallelism**: Future integration of asynchronous label propagation for extreme-scale tracking (arXiv:2003.02351).

## 3. Predicate Layer (Mathematical Kernels)

### 3.1 Universal Zero-Crossing Solver
Solves $f(x)=0$ on a $k$-simplex using linear interpolation and robust **Simulation of Simplicity (SoS)** tie-breaking.

### 3.2 Exact Analytical Parallel Vectors (ExactPV)
Analytical cubic rational solver for exact parallel vector tracking (arXiv:2107.02708).

## 4. Data Abstraction Layer (`ndarray`)
Zero-copy data flow leveraging `ftk::ndarray_stream`.

## 5. GPU Acceleration Strategy
Static polymorphism and lightweight device-mesh structs for CUDA/HIP.

## 6. Slicing and Visualization
*   **Slicing**: Intersection of spacetime manifolds with $t=T$.
*   **ParaView Integration**: Export to `.vtu` and `.vtp` files.
