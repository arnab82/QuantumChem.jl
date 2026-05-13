# Project #26: Integral-Direct RHF

Code: `src/direct_scf.jl`, `src/rhf.jl`

Tests: `test/runtests.jl`, testset `Project #26 integral-direct RHF`

## Goal

Run RHF without materializing the four-index AO electron-repulsion integral
tensor:

```julia
rhf = run_rhf(direct=true)
direct = run_direct_rhf()
```

The direct path keeps the same SCF machinery as the normal RHF driver, but
replaces the Fock builder.  Given this package's closed-shell density convention

```text
D = C_occ C_occ'
```

the direct Fock builder evaluates:

```text
F = h + 2J[D] - K[D]
```

where `J` and `K` are requested from PySCF's direct J/K contraction routines.

## Why This Matters

The in-core RHF path stores `(mu nu | lambda sigma)` as a dense rank-4 tensor,
which scales as `O(nbasis^4)` memory.  The direct path rebuilds J/K from the
current density during each SCF iteration and can therefore handle larger basis
sets without keeping all AO ERIs in Julia memory.

## API

```julia
rhf = run_rhf(direct=true)
rhf.eri        # nothing unless return_eri=true
rhf.direct     # true

D = make_density(rhf.mo_coeffs, rhf.n_elec ÷ 2)
jk = direct_jk(rhf.mol, D)
F = make_fock_direct(D, rhf.h1e, rhf.mol)
```

`direct=true` is mutually exclusive with `outcore=true` and the current
symmetry-blocked SCF path.  Use `return_eri=true` if a downstream post-HF method
needs the AO ERI tensor after the direct SCF is complete.

## Checks

For default STO-3G water, direct RHF reproduces the in-core RHF energy and Fock
matrix to numerical precision, while leaving `rhf.eri === nothing`.
