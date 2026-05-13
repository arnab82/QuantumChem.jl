# Project #15: Unrestricted Hartree-Fock

Code: `src/uhf.jl`

Tests: `test/runtests.jl`, testset `Project #15 unrestricted Hartree-Fock`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#15 UHF`.

## Goal

Generalize RHF to independent alpha and beta spin orbitals:

```julia
uhf = run_uhf()
oh = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
```

The default system is closed-shell STO-3G water, so it should agree with RHF.
Open-shell systems use PySCF's spin convention:

```text
spin = n_alpha - n_beta
```

## UHF Equations

The spin densities are built separately:

```text
D_alpha[mu,nu] = sum_i C_alpha[mu,i] C_alpha[nu,i]
D_beta[mu,nu]  = sum_i C_beta[mu,i]  C_beta[nu,i]
```

The alpha Fock matrix is:

```text
F_alpha[mu,nu] = h[mu,nu] + sum_lambda_sigma (
    (D_alpha + D_beta)[lambda,sigma] (mu nu | lambda sigma)
  - D_alpha[lambda,sigma] (mu lambda | nu sigma)
)
```

The beta Fock matrix has the same Coulomb term but uses beta exchange.

The electronic energy is:

```text
E_UHF = 1/2 sum_mu_nu D_alpha[mu,nu] (h[mu,nu] + F_alpha[mu,nu])
      + 1/2 sum_mu_nu D_beta[mu,nu]  (h[mu,nu] + F_beta[mu,nu])
```

## Implementation Notes

1. Build AO integrals with the same PySCF helpers used by RHF.
2. Form `S^(-1/2)` and diagonalize the core Hamiltonian for the initial guess.
3. Build separate alpha and beta densities.
4. Build separate alpha and beta Fock matrices.
5. Optionally apply DIIS to the paired alpha/beta commutator errors.
6. Diagonalize the two Fock matrices independently.
7. Report total energy, orbital energies, densities, Fock matrices, and `<S^2>`.

## Spin Diagnostic

For a UHF determinant:

```text
<S^2> = Sz(Sz + 1) + n_beta - sum_ij |<phi_i_alpha | phi_j_beta>|^2
```

Closed-shell water gives `<S^2> = 0`.  The OH radical test gives
`E_UHF = -74.36040547332438 Eh` and `<S^2> = 0.7529381425680679`.

