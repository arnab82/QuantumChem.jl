# Project #22: Full Configuration Interaction

Code: `src/fci.jl`

Tests: `test/runtests.jl`, testset `Project #22 full CI`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#22 FCI`.

## Goal

Diagonalize the electronic Hamiltonian in the complete fixed-electron
determinant space for the current spin-orbital basis:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
fci = run_fci(rhf, mp2)
```

FCI is exact within the one-particle basis.  For STO-3G water, the spin-orbital
space has:

```text
nspin = 14
nelec = 10
dimension = binomial(14, 10) = 1001
```

## Determinant Basis

`fci_determinants(nspin, nelec)` enumerates every determinant with `nelec`
occupied spin orbitals.  The first determinant is the canonical Hartree-Fock
reference:

```text
|1 2 3 ... nelec>
```

Labels store excitation rank relative to the reference.  For STO-3G water:

```text
rank 0:   1
rank 1:  40
rank 2: 270
rank 3: 480
rank 4: 210
```

## Hamiltonian

The FCI implementation reuses the selected-CI Slater-Condon-style matrix
element evaluator:

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```

Only determinant pairs differing by zero, one, or two spin orbitals have
nonzero matrix elements.

## Reference

For default STO-3G water:

```text
E_FCI corr  = -0.07090027025027723 Eh
E_FCI total = -75.01298019844313 Eh
```

This is slightly above the default HCI+PT2 estimate because PT2 is not
variational, and below the HCI variational energy as expected.
