# Project #2: Harmonic Vibrational Analysis

Code: `src/vibrations.jl`

Tests: `test/runtests.jl`, testset `Project #2 harmonic vibrations`

## Goal

Compute harmonic vibrational frequencies from a Cartesian Hessian.

## Main API

```julia
hessian_data = read_hessian(hessian_string)
vib = run_vibrational_analysis(mol, hessian_data.hessian)
```

## Algorithm

1. Read the Cartesian Hessian.
2. Mass-weight the Hessian using atomic masses.
3. Diagonalize the mass-weighted Hessian.
4. Convert eigenvalues into harmonic frequencies.

## Notes

For nonlinear molecules, six near-zero modes correspond to translations and
rotations.  The current tests use the Crawford water Hessian and compare the
reported vibrational eigenvalues and frequencies against reference values.

