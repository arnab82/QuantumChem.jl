# Project #3: Restricted Hartree-Fock SCF

Code: `src/rhf.jl`

Tests: `test/runtests.jl`, testset `RHF`

## Goal

Implement closed-shell restricted Hartree-Fock:

```julia
rhf = run_rhf()
```

The default system is STO-3G water in Bohr.

## SCF Loop

1. Build the overlap matrix `S`, core Hamiltonian `h`, two-electron integrals
   `(mu nu | lambda sigma)`, and nuclear repulsion using PySCF.
2. Form the symmetric orthogonalizer `X = S^(-1/2)`.
3. Diagonalize the core Hamiltonian guess in the orthonormal AO basis.
4. Build the density from occupied orbitals.
5. Build the Fock matrix:

   ```text
   F[mu,nu] = h[mu,nu] + sum_lambda_sigma D[lambda,sigma]
              * (2 (mu nu | lambda sigma) - (mu lambda | nu sigma))
   ```

6. Diagonalize the Fock matrix and iterate until the energy change is small.

## Returned Data

`run_rhf` returns total energy, orbital energies, MO coefficients, Fock matrix,
AO integrals, overlap matrix, basis size, electron count, and iteration data.

The reference RHF total energy is `-74.942079928192 Eh`.

