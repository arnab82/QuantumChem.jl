# Project #10: DIIS for CC Amplitudes

Code: `src/ccsd.jl`

Tests: `test/runtests.jl`, testset `Project #10 CCSD amplitude DIIS`

## Goal

Accelerate CCSD amplitude convergence:

```julia
ccsd = run_ccsd(rhf, mp2; diis=true)
```

## Error Vector

The implementation uses flattened amplitude residuals:

```text
e = T_new(raw) - T_current
```

The raw update is stored before extrapolation, then DIIS forms a linear
combination of previous raw amplitude sets.

## Local Helpers

- `cc_amplitude_error`
- `cc_diis_coefficients`
- `extrapolate_amplitudes`
- `ccsd_energy`

## Notes

The default DIIS subspace size is eight and extrapolation starts after three
iterations.  For the STO-3G water test case, DIIS converges to the same CCSD
energy in fewer iterations than plain CCSD.

