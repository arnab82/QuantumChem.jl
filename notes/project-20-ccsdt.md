# Project #20: CCSDT

Code: `src/ccsdt.jl`

Generator: `tools/generate_ccsdt_wicked.py`

Tests: `test/runtests.jl`, testset `Project #20 CCSDT`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#20 CCSDT`.

## Goal

Solve the connected coupled-cluster equations through triple excitations:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
ccsdt = run_ccsdt(rhf, mp2)
```

The method uses the RHF canonical spin-orbital basis, with amplitudes:

```text
T = T1 + T2 + T3
t_i^a
t_ij^ab
t_ijk^abc
```

## Wicked Equation Generation

The residual equations are generated from the similarity-transformed
Hamiltonian:

```text
Hbar = exp(-T) H exp(T)
```

For a two-body Hamiltonian, the BCH expansion truncates after the fourth nested
commutator.  Wick&d generates the spin-orbital residual equations:

```text
R0:  3 terms
R1: 15 terms
R2: 37 terms
R3: 47 terms
```

To regenerate the equation source:

```shell
WICKED_PATH=/private/tmp/wicked /usr/bin/python3 tools/generate_ccsdt_wicked.py --summary
```

## Iteration

The solver starts from MP2 doubles, zero singles, and zero triples:

```text
t_i^a       = 0
t_ij^ab     = <ij||ab> / (eps_i + eps_j - eps_a - eps_b)
t_ijk^abc   = 0
```

Each iteration evaluates the Wicked residuals, antisymmetrizes the doubles and
triples residuals, applies a diagonal Jacobi update, and optionally accelerates
the amplitudes with DIIS.

## Reference

For default STO-3G water:

```text
E_CCSDT corr  = -0.07081280769760294 Eh
E_CCSDT total = -75.01289273589046 Eh
iterations    = 14 with DIIS
```

The full CCSDT energy is lower than both CCSD and perturbative CCSD(T) for this
small test system.
