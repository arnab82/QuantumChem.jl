# Project #25: RHF Geometry Optimization

Code: `src/optimization.jl`

Tests: `test/runtests.jl`, testset `Project #25 RHF geometry optimization`

## Goal

Use the RHF analytic gradient from Project #24 to relax Cartesian nuclear
coordinates:

```julia
rhf = run_rhf()
opt = run_rhf_geometry_optimization(rhf; maxiter=20)
opt.converged
opt.energies
opt.coordinates
```

Coordinates are stored and returned in Bohr.  The returned `opt.atoms` string is
ready to pass back into PySCF-backed routines.

## Algorithm

The optimizer uses a simple, inspectable Cartesian procedure:

1. Build the RHF wavefunction at the current geometry.
2. Compute the analytic RHF gradient.
3. Choose a search direction using either BFGS or steepest descent.
4. Remove net translation from the Cartesian step direction.
5. Cap the largest per-atom displacement with `max_step`.
6. Backtrack until the RHF total energy satisfies a downhill Armijo check.
7. Recompute the gradient and update the inverse-Hessian approximation.

The default method is `:bfgs`; `method=:steepest_descent` is available for
debugging.

## Important Controls

```julia
run_rhf_geometry_optimization(
    rhf;
    method=:bfgs,
    maxiter=20,
    gradient_tol=3e-4,
    rms_gradient_tol=2e-4,
    initial_step=0.5,
    max_step=0.2,
)
```

`max_step` is in Bohr and limits the largest Cartesian displacement of any atom
in a single accepted iteration.

## Checks

The test suite verifies one optimization step from the default STO-3G water
geometry:

- The RHF total energy decreases.
- The maximum gradient decreases.
- The accepted displacement obeys `max_step`.
- The optimized atom string round-trips through `format_atoms_bohr`.
