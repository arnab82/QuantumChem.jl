# Project #11: Out-of-Core SCF

Code: `src/outcore.jl`, `src/rhf.jl`

Tests: `test/runtests.jl`, testset `Project #11 out-of-core SCF`

## Goal

Reduce memory pressure by storing only permutationally unique AO electron
repulsion integrals on disk and streaming them during Fock builds:

```julia
rhf = run_rhf(outcore=true)
```

## Main Objects

```julia
struct ERIFile
    path::String
    nbasis::Int
    nrecords::Int
    cutoff::Float64
end
```

## Workflow

1. Generate or receive AO ERIs.
2. Write unique integrals `(ij|kl)` to a compact binary file.
3. During each Fock build, read the file in batches.
4. Expand all symmetry-related integral permutations on the fly.
5. Accumulate Coulomb and exchange contributions into the Fock matrix.

## Main API

- `write_unique_eri_file`
- `read_eri_file_header`
- `make_fock_outcore`
- `run_rhf(outcore=true)`

## Test Strategy

The tests compare the streamed out-of-core Fock matrix against the original
in-core Fock matrix and check that out-of-core RHF reproduces the reference
RHF total energy.

