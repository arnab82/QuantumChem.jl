# Project #14: EOM-CCSD

Code: `src/eom_ccsd.jl`

Tests: `test/runtests.jl`, testset `Project #14 EOM-CCSD`

## Status of Crawford Spec

The Crawford Project #14 directory currently only contains the title,
`Excited Electronic States: EOM-CCSD`, without detailed reference outputs.
This repo therefore implements a compact educational EOM-CCSD route and tests
internal consistency rather than claiming official Crawford reference energies.

## Goal

Compute excitation roots from the linearized CCSD amplitude residual equations:

```julia
ccsd = run_ccsd(rhf, mp2; diis=true)
eom = run_eom_ccsd(rhf, mp2, ccsd; nroots=5)
```

## Implementation

1. Use converged CCSD amplitudes `T1` and `T2`.
2. Build the CCSD residual equations.
3. Linearize the residual equations by finite differences.
4. Pack the EOM space into singles plus unique antisymmetric doubles.
5. Diagonalize the resulting Jacobian.
6. Return positive real excitation roots.

## Local Helpers

- `eom_double_indices`
- `eom_dimension`
- `pack_eom_amplitudes`
- `unpack_eom_amplitudes`
- `ccsd_residuals`
- `build_eom_ccsd_jacobian`
- `run_eom_ccsd`

## Notes

For default STO-3G water, the EOM space has:

```text
40 singles + 270 unique doubles = 310 dimensions
```

The tests verify dimension, residual size at converged CCSD amplitudes, positive
roots, and pack/unpack antisymmetry behavior.

