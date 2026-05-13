# Project #9: Symmetry in SCF

Code: `src/symmetry.jl`, `src/rhf.jl`

Tests: `test/runtests.jl`, testset `Project #9 symmetry in SCF`

## Goal

Use symmetry-adapted orbitals to block the RHF problem.

```julia
run_rhf(symmetry=true)
run_rhf(symmetry="Cs")
run_rhf(symmetry="C1")
```

## Implementation

The general path asks PySCF to build symmetry orbitals, then converts PySCF's
symmetry blocks into a local `SymmetryInfo` object:

```julia
struct SymmetryInfo
    transform::Matrix{Float64}
    irreps::Vector{Symbol}
    occupations::Vector{Int}
end
```

## Important Helpers

- `pyscf_symmetry_info`
- `block_ranges`
- `sopi`
- `symmetry_transform_1e`
- `symmetry_transform_eri`
- `symmetry_blocks`
- `block_eigen`
- `make_density_symmetry`
- `make_fock_symmetry`

## Notes

The code supports PySCF-detected point groups and requested subgroups.  The
tests check both default water `C2v` and forced `Cs`, including block sizes,
occupations, energies, and off-block Fock norms.

