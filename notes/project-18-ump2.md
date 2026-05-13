# Project #18: Unrestricted MP2

Code: `src/ump2.jl`, reusing UHF spin-orbital helpers from `src/uccsd.jl`

Tests: `test/runtests.jl`, testset `Project #18 unrestricted MP2`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#18 U-MP2`.

## Goal

Evaluate the MP2 correlation correction from a UHF reference:

```julia
uhf = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
ump2 = run_ump2(uhf)
```

## Spin-Orbital Form

UMP2 uses the same occupied-first spin-orbital order as UCCSD:

```text
occupied alpha
occupied beta
virtual alpha
virtual beta
```

The doubles amplitudes are:

```text
t_ij^ab = <ij||ab> / (eps_i + eps_j - eps_a - eps_b)
```

The energy is:

```text
E_UMP2 = 1/4 sum_ijab <ij||ab> t_ij^ab
```

Because the alpha and beta orbitals can have different energies, UMP2 uses
explicit occupied and virtual spin-orbital energies rather than the paired
closed-shell spatial energies used by RHF-MP2.

## Implementation Notes

1. Build UHF spin-orbital coefficients and spin labels.
2. Transform AO ERIs into that unrestricted spin-orbital MO basis.
3. Antisymmetrize the spin-orbital ERIs.
4. Slice the OOVV block.
5. Build MP2 doubles with explicit spin-orbital denominators.
6. Evaluate the spin-orbital MP2 energy.

## Reference

For closed-shell water, UMP2 reduces to the existing RHF-MP2 correction:

```text
E_MP2 = -0.049149636120 Eh
```

For the OH radical in STO-3G at `R = 1.8 Bohr`, the local UMP2 result matches
PySCF:

```text
E_UMP2 corr  = -0.015217583817377001 Eh
E_UMP2 total = -74.37562305714175 Eh
```

