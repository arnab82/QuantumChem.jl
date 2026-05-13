# Project #13: Davidson-Liu CIS

Code: `src/excited_states.jl`

Tests: `test/runtests.jl`, testset `Project #13 Davidson-Liu CIS`

## Goal

Compute a few lowest spin-adapted singlet CIS roots without diagonalizing the
full CIS Hamiltonian:

```julia
davidson = run_davidson_cis(rhf, mp2; nroots=5)
```

## Davidson-Liu Ingredients

- Guess vectors: unit vectors corresponding to the lowest diagonal elements.
- Sigma build: matrix-free `sigma = H * c`.
- Subspace Hamiltonian: `G = B' * H * B`.
- Ritz vectors: current approximate eigenvectors.
- Residuals: `r_k = H c_k - omega_k c_k`.
- Preconditioner: diagonal CIS approximation.
- Subspace collapse: when needed, keep current best Ritz vectors.

## Local Helpers

- `cis_singlet_sigma`
- `cis_singlet_diagonal`
- `davidson_liu`
- `run_davidson_cis`

## Test Strategy

The tests compare the Davidson roots against exact spin-adapted singlet CIS
roots from Project #12 and verify the matrix-free sigma equation against an
explicit matrix-vector product.

