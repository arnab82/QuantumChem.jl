# Project #24: RHF Analytic Nuclear Gradients

Code: `src/gradients.jl`

Tests: `test/runtests.jl`, testset `Project #24 RHF analytic gradients`

## Goal

Evaluate the derivative of the restricted Hartree-Fock total energy with
respect to nuclear coordinates:

```julia
rhf = run_rhf()
grad = run_rhf_gradient(rhf)
grad.gradient      # natom x 3, in Eh/Bohr
grad.max_force
```

## Formula

For a closed-shell RHF wavefunction the total gradient is:

```text
dE/dA = dE_elec/dA + dE_nuc/dA
```

The electronic part is assembled in the AO basis from:

```text
P_mn = 2 sum_i C_mi C_ni
W_mn = 2 sum_i epsilon_i C_mi C_ni
```

where `P` is the spin-summed density and `W` is the energy-weighted density.
The terms are:

```text
sum_mn P_mn h_mn^A
+ two-electron derivative contribution
- Pulay overlap term from W and S^A
```

PySCF supplies the AO derivative integrals; the package assembles the density
matrices, atom blocks, electronic gradient, nuclear repulsion gradient, and
summary diagnostics in Julia.

## Useful Checks

For a molecule with no external field, the total gradient should be
translationally invariant:

```julia
sum(grad.gradient; dims=1)
```

For the default STO-3G water geometry, the reference gradient is:

```text
 O   0.0000000000  -0.0974413804   0.0000000000
 H   0.0863000587   0.0487206902   0.0000000000
 H  -0.0863000587   0.0487206902   0.0000000000
```

The force on each atom is the negative of this gradient.
