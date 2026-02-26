# ExactPV Research Questions and TODO

## Degeneracy Handling (Research Question)

**Issue**: When vector fields are parallel everywhere (or over extended regions), the ExactPV solver encounters degenerate polynomials where all coefficients vanish or nearly vanish.

**Example**:
- Field configuration: V = k*U everywhere (constant scalar multiple)
- Result: Entire volume is a parallel vector surface
- Current behavior: Solver may return 0 features or numerical instability

**Research Questions**:
1. How to detect and distinguish:
   - Volumetric degeneracy (entire region is PV surface)
   - Surface degeneracy (2D patches are PV surfaces)
   - Isolated features (discrete puncture points/curves)

2. How to meaningfully report degenerate regions:
   - Should we output "entire mesh is degenerate"?
   - Should we detect boundaries of degenerate regions?
   - How to handle mixed cases (some simplices degenerate, others not)?

3. Numerical precision concerns:
   - Near-degenerate cases (vectors nearly parallel)
   - Tolerance for declaring degeneracy
   - Robustness of polynomial solvers

**Current Status**:
- Triangle solver (`solve_pv_triangle`) returns INT_MAX for degenerate cases
- Tetrahedron solver (`solve_pv_tetrahedron`) needs similar handling
- Engine doesn't distinguish degenerate vs no-feature cases

**Potential Solutions**:
- Add explicit degeneracy detection before polynomial solving
- Return a special feature type for degenerate simplices
- Implement hierarchical analysis (check degeneracy at coarse level first)
- Use perturbation analysis for near-degenerate cases

---

## Codimension Naming Confusion

**Issue**: The base `Predicate<M, T>` template uses M to mean "number of data components", but misleadingly calls it "codimension" in the base class.

**Example**:
- `ExactPVPredicate : public Predicate<6, T>` - M=6 means 6 data components
- But actual mathematical codimension = 2 (set separately)

**Fix**: This is a framework-level naming issue. Consider:
- Renaming M to "NumComponents" or "DataDimension"
- Keeping "codimension" separate as it currently is
- Adding clarifying comments

---

## Future Work

- [ ] Implement degeneracy detection and handling
- [ ] Add tetrahedron extraction to engine (currently only triangles)
- [ ] CUDA implementation of ExactPV extraction
- [ ] Adaptive sampling of parametric curves for visualization
- [ ] Critical point detection on PV curves
- [ ] High-level API integration
- [ ] VTK output for curves and puncture points
- [ ] Performance testing on large meshes
- [ ] Add unit tests for degenerate cases
- [ ] Documentation with examples

---

## Notes

- First working integration completed: 2025-01-XX
- Example: `examples/exact_pv_simple.cpp`
- Successfully finds puncture points on triangular meshes
- Lambda parameter stored in `scalar` field of FeatureElement
