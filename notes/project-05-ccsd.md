# Project #5: CCSD

Code: `src/ccsd.jl`

Tests: `test/runtests.jl`, testsets `CCSD correlation energy`,
`CCSD total energy`, `T1 and T2 amplitude shapes`, and `T2 antisymmetry`

## Goal

Solve coupled-cluster singles and doubles amplitude equations in the
antisymmetrized spin-orbital basis.

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
ccsd = run_ccsd(rhf, mp2)
```

## Key Objects

- `ts[i,a]`: singles amplitudes
- `td[i,j,a,b]`: doubles amplitudes
- `mo_to_aso`: converts spatial MO ERIs into antisymmetrized spin-orbital ERIs
- `ccsd_scf`: iterative CCSD amplitude solver

## Algorithm Notes

The implementation follows the Stanton-style CCSD working equations using
intermediates such as `Fae`, `Fmi`, `Fme`, `Wmnij`, `Wabef`, and `Wmbej`.
The initial doubles amplitudes are MP2 amplitudes and singles start at zero.

Convergence is checked using both correlation-energy change and amplitude
change.

## Reference

For STO-3G water:

```text
E_CCSD corr = -0.070680088376 Eh
E_CCSD total = -75.012760016568 Eh
```

