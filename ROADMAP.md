# FTK2 Development Roadmap

This document outlines the planned milestones and future features for the FTK2 project.

## Milestone 1: Core Simplicial Engine (Current Focus)
*   [ ] Implementation of `Mesh` abstractions (Regular, Extruded).
*   [ ] Unified `SimplicialEngine` for extraction and tracking.
*   [ ] **Universal Zero-Crossing Engine**: Handle **Critical Points**, **Levelsets**, and **Fibers** ($m=1, 2, ...$).

## Milestone 2: GPU Acceleration (CUDA)
*   [ ] **CUDA Integration**: Build system support. (Done)
*   [ ] **Device Meshes**: CUDA-compatible POD structs. (Done)
*   [ ] **Parallel Extraction**: Discovery of feature nodes on GPU. (Done)
*   [ ] **GPU Manifold Construction**: Port Marching Simplices logic. (Done)
*   [ ] **Streamed Data Management**: Sliding-window processing for CPU and GPU (2-timestep limit).
*   [ ] **Unstructured GPU Acceleration**: Support explicit meshes on NVIDIA GPUs.

## Milestone 3: Distributed-Parallel Tracking (High Priority)
Integrate the distributed-parallel CCL algorithm from arXiv:2003.02351.
*   [ ] Implement the asynchronous label propagation and merging logic.
*   [ ] Integrate with MPI for cross-node tracking.

## Milestone 4: Advanced Mesh & Feature Types
*   [ ] **Consistent Manifold Orientation**: Implement a rigorous orientation scheme for simplices to ensure consistent normals during slicing and visualization.
*   [ ] **Robust Fixed-Point Arithmetic**: Replace hard-coded SoS quantization with a unified `FixedPoint` data type.
*   [ ] **ExactPV Predicate**: Analytical cubic rational solver (arXiv:2107.02708).
*   [ ] **Magnetic Flux Vortices**: Phase-angle based extraction on 2-simplices (IEEE TVCG 2017).
*   [ ] **Stable Feature Flow Fields (Stable FFF)**: High-precision tracking via stable streamline integration (IEEE TVCG 2010).
*   [ ] **Lagrangian Particle Tracer**: Pathline integration across mesh types.
*   [ ] **Deforming Spacetime Mesh**: Non-linear extrusion (arXiv:2309.02677).

## Milestone 5: ParaView & Ecosystem
*   [ ] **ParaView Plugins (High Priority)**: Integration with ParaView/VTK for interactive analysis.
*   [ ] **Python Bindings (PyFTK2)**: High-level Python interface for data science workflows.
