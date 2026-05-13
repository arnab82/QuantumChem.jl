# Project #23: Density Matrix Renormalization Group

Code: `src/custom_dmrg.jl`, `src/fci_dmrg.jl`

Tests: `test/runtests.jl`, testset `Project #23 DMRG/MPS`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#23 DMRG`.

## Goal

Run DMRG directly from the Hartree-Fock molecular-orbital basis:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
dmrg = run_dmrg(rhf, mp2; maxdim=[50, 100, 200, 400], cutoff=1e-8)
```

The production path is implemented inside this repo.  It:

1. Runs RHF and MP2 integral transformation in the canonical MO basis.
2. Converts the spatial MO one-electron integrals and ERIs to spin-orbital
   integrals.
3. Builds the fermionic second-quantized Hamiltonian:
   `sum h_pq Cdag_p C_q + 1/4 sum <pq||rs> Cdag_p Cdag_q C_s C_r`.
4. Converts that Hamiltonian to a Jordan-Wigner product-term MPO.
5. Starts from the Hartree-Fock occupation string.
6. Optimizes the MPS with two-site DMRG sweeps.

The older exact FCI-compression path is still available as `run_fci_dmrg`.  It
is only meant as a small-system reference for testing MPS/MPO tensor
contractions:

```julia
fci = run_fci(rhf, mp2)
reference = run_fci_dmrg(rhf, mp2; fci_result=fci)
```

For default STO-3G water:

```text
nspin = 14
nelec = 10
FCI dimension = 1001
Fock-vector dimension = 2^14 = 16384
```

## MPS Layout

The direct DMRG path and the reference path both store plain Julia MPS tensors
with shape:

```text
(left_bond, physical_dimension, right_bond)
```

The physical dimension is 2 because each spin orbital is either unoccupied or
occupied.  The determinant bit ordering is preserved, so orbital 1 is the
fastest-changing local index in the full Fock vector.

For the untruncated default water FCI reference, the retained bond dimensions
are:

```text
2, 4, 8, 16, 24, 29, 25, 19, 22, 16, 8, 4, 2
```

## MPO And Two-Site Contraction

The direct path creates a custom Jordan-Wigner product-term MPO from:

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```

For default water this gives:

```text
one-body MPO terms = 54
two-body MPO terms = 4172
```

The exact reference path also has plain-Julia two-site helpers exposing the core
contraction used in a DMRG sweep:

```julia
theta = two_site_tensor(reference.tensors, reference.two_site_site)
sigma = two_site_tensor_contract(reference.mpo, reference.tensors,
                                 reference.two_site_site, theta)
energy = two_site_energy(reference.mpo, reference.tensors,
                         reference.two_site_site, theta)
```

Internally, the surrounding MPS tensors define the left and right environments.
The two-site center tensor is projected to determinant coefficients, the MPO is
applied, and the result is projected back into the two-site tensor shape.  This
keeps the implementation explicit and easy to inspect while preserving the
standard DMRG idea of an effective Hamiltonian acting on two neighboring sites.

## Truncation

The direct DMRG path controls bond dimension with custom sweep parameters:

```julia
dmrg = run_dmrg(rhf, mp2; maxdim=[50, 100, 200, 400, 800], cutoff=1e-9)
```

The exact reference MPS helper accepts either an absolute singular-value
`cutoff` or a maximum bond dimension:

```julia
compressed = run_fci_dmrg(rhf, mp2; fci_result=fci, max_bond=8)
compressed.discarded_weight
compressed.total_energy
```

The exact reference path keeps all singular values above `1e-12`, which is
effectively exact for the small reference calculation.  A reduced `max_bond`
gives a compact approximation and reports the discarded weight and
reconstruction error.

## Reference

For the default STO-3G water exact reference:

```text
E_DMRG/MPS corr  = -0.07090027025022039 Eh
E_DMRG/MPS total = -75.01298019844307 Eh
max bond dimension = 29
reconstruction error = 6.1e-15
two-site electronic energy = -83.01534726025385 Eh
```

These agree with FCI to numerical precision in the untruncated reference MPS.
