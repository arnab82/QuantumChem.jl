# Project #19: CISD

Code: `src/cisd.jl`

Tests: `test/runtests.jl`, testset `Project #19 CISD`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#19 CISD`.

## Goal

Build and diagonalize the configuration-interaction Hamiltonian in the space of
the Hartree-Fock reference determinant plus all single and double spin-orbital
excitations:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
cisd = run_cisd(rhf, mp2)
```

## Determinant Space

For STO-3G water:

```text
nspin = 14
nelec = 10
dimension = 1 reference + 10*4 singles + binomial(10,2)*binomial(4,2) doubles
          = 311
```

Determinants are stored as bit strings.  Orbital `p` corresponds to bit
`p - 1`.  The reference determinant occupies spin orbitals `1:nelec`.

## Hamiltonian

The electronic Hamiltonian is generated directly with fermionic creation and
annihilation operators:

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```

The sign is determined by the number of occupied orbitals below each
creation/annihilation index.  The resulting matrix is symmetrized before
diagonalization.

## Implementation Notes

1. Transform the AO core Hamiltonian into the RHF MO basis.
2. Lift the spatial one-electron matrix into the interleaved alpha/beta
   spin-orbital basis.
3. Reuse `mo_to_aso` for antisymmetrized spin-orbital ERIs.
4. Generate reference, single, and double determinants.
5. Build the CISD Hamiltonian by applying the one- and two-body operators.
6. Diagonalize the Hamiltonian and add nuclear repulsion to total energies.

## Reference

For default STO-3G water, the local CISD result matches PySCF:

```text
E_CISD corr  = -0.0691430716170365 Eh
E_CISD total = -75.01122299980942 Eh
```

The HF determinant weight in the CISD ground-state vector is about `0.955`.

