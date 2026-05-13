# Project #8: DIIS for SCF

Code: `src/rhf.jl`

Tests: `test/runtests.jl`, testset `Project #8 RHF DIIS`

## Goal

Accelerate RHF convergence using Pulay DIIS:

```julia
rhf = run_rhf(diis=true)
```

## Error Vector

The SCF DIIS error matrix is:

```text
e = F D S - S D F
```

At convergence, the Fock and density matrices commute in the AO metric, so this
error approaches zero.

## DIIS Procedure

1. Store recent Fock matrices and error matrices.
2. Build the Pulay B matrix from error inner products.
3. Solve the constrained linear system so the coefficients sum to one.
4. Extrapolate a new Fock matrix as a linear combination of stored Focks.

## Local Helpers

- `diis_error_matrix`
- `diis_coefficients`
- `extrapolate_fock`

The test verifies that DIIS reaches the RHF reference energy with fewer
iterations than the plain SCF path.

