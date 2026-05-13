# Project #16: Restricted Open-Shell Hartree-Fock

Code: `src/rohf.jl`

Tests: `test/runtests.jl`, testset `Project #16 restricted open-shell Hartree-Fock`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#16 ROHF`.

## Goal

Implement restricted open-shell Hartree-Fock: one shared spatial orbital set,
with closed-shell orbitals doubly occupied and open-shell orbitals singly
occupied.

```julia
rohf = run_rohf()
oh = run_rohf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
```

The default system is closed-shell STO-3G water, so it reduces to RHF.

## Occupations

ROHF uses PySCF's spin convention:

```text
spin = n_alpha - n_beta
```

For high-spin ROHF:

```text
n_closed = n_beta
n_open   = n_alpha - n_beta
```

The OH radical test has 9 electrons and `spin = 1`, giving:

```text
n_closed = 4
n_open   = 1
occupations = [2, 2, 2, 2, 1, 0]
```

## Roothaan Effective Fock

The alpha and beta Fock matrices are built as in UHF, but the orbitals are
updated from Roothaan's effective Fock.  In closed/open/virtual blocks:

```text
           closed   open   virtual
closed       Fc      Fb      Fc
open         Fb      Fc      Fa
virtual      Fc      Fa      Fc
```

where:

```text
Fc = (Fa + Fb) / 2
```

This keeps one orbital coefficient matrix while allowing the exchange operator
to distinguish closed and open spin occupancy.

## Implementation Notes

1. Build AO integrals with the shared RHF/PySCF helpers.
2. Convert `n_elec` and `spin` into closed/open-shell counts.
3. Build closed, open, alpha, and beta densities from one `C` matrix.
4. Build alpha and beta Fock matrices with the UHF helper.
5. Build the Roothaan effective Fock and diagonalize it.
6. Optionally apply DIIS using the Roothaan Fock and total density.
7. Return energy, occupations, densities, Fock matrices, and spin diagnostics.

## Reference

For the OH radical in STO-3G at `R = 1.8 Bohr`, the local ROHF result matches
PySCF:

```text
E_ROHF = -74.35932208774248 Eh
<S^2>  = 0.75
```

The UHF energy is slightly lower for the same system because UHF has more
variational freedom.

