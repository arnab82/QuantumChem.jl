# Project #6: CCSD(T)

Code: `src/ccsd_t.jl`

Tests: `test/runtests.jl`, testsets `CCSD(T) triples correction` and
`CCSD(T) total energy`

## Goal

Add a perturbative triples correction to converged CCSD.

```julia
ccsdt = run_ccsd_t(rhf, mp2, ccsd)
```

## Idea

CCSD includes connected single and double excitations.  CCSD(T) estimates the
effect of connected triples perturbatively using the converged CCSD amplitudes
and orbital-energy denominators.

## Reference

For STO-3G water:

```text
E(T) = -0.000099877272 Eh
E_CCSD(T) total = -75.012859893840 Eh
```

## Practical Note

The triples step is much more expensive than MP2 and RHF-style operations
because it loops over triples of occupied and virtual spin orbitals.

