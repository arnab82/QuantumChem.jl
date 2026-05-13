# Project #17: Unrestricted CCSD

Code: `src/uccsd.jl`, reusing the spin-orbital CCSD solver in `src/ccsd.jl`

Tests: `test/runtests.jl`, testset `Project #17 unrestricted CCSD`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#17 UCCSD`.

## Goal

Run CCSD from a UHF reference, where alpha and beta orbitals can have different
MO coefficients and orbital energies:

```julia
uhf = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
uccsd = run_uccsd(uhf)
```

## Spin-Orbital Ordering

The existing CCSD solver expects all occupied spin orbitals first.  UCCSD uses:

```text
occupied alpha
occupied beta
virtual alpha
virtual beta
```

For the OH radical in STO-3G, this gives:

```text
o = 9 occupied spin orbitals
v = 3 virtual spin orbitals
T1 shape = (9, 3)
T2 shape = (9, 9, 3, 3)
```

## Integral Build

First build a spin-orbital AO coefficient matrix from UHF orbitals.  Each
column is either an alpha or beta MO.  Transform AO ERIs with that coefficient
matrix, then apply spin selection rules:

```text
<pq||rs> = (p r | q s) delta(spin_p, spin_r) delta(spin_q, spin_s)
         - (p s | q r) delta(spin_p, spin_s) delta(spin_q, spin_r)
```

This produces the antisymmetrized spin-orbital ERI tensor used by the same CCSD
equations as Project #5.

## Denominators

RHF CCSD can infer paired alpha/beta orbital energies from spatial orbital
energies.  UCCSD cannot, so Project #17 adds `make_td_spinorbital`, which builds
initial MP2 doubles from explicit occupied and virtual spin-orbital energies.

## Reference

For the OH radical in STO-3G at `R = 1.8 Bohr`, the local UCCSD result matches
PySCF:

```text
E_UHF corr from UCCSD = -0.02343230248645602 Eh
E_UCCSD total         = -74.38383777581083 Eh
```

For closed-shell water, `run_uccsd(run_uhf())` reduces to the same correlation
energy as the RHF-based CCSD path:

```text
E_CCSD corr = -0.070680088376 Eh
```

