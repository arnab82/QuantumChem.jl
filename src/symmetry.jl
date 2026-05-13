"""
Point-group symmetry helpers for Crawford Programming Project #9.

The implementation targets the STO-3G water example used throughout the
Crawford projects.  Symmetry orbitals are represented by columns of an
AO-to-SO transformation matrix, grouped contiguously by irreducible
representation.
"""

using LinearAlgebra
using TensorOperations
using Base.Threads: @threads

"""
    SymmetryInfo

Symmetry-orbital metadata used by block SCF routines.

Fields:
- `transform`: AO-to-symmetry-orbital coefficient matrix `T`.
- `irreps`: irrep label for each symmetry orbital, grouped contiguously.
- `occupations`: number of occupied spatial orbitals assigned to each irrep
  block.
"""
struct SymmetryInfo
    transform::Matrix{Float64}
    irreps::Vector{Symbol}
    occupations::Vector{Int}
end

pyscf_symmetry_setting(symmetry::Symbol) = String(symmetry)
pyscf_symmetry_setting(symmetry::AbstractString) = String(symmetry)
pyscf_symmetry_setting(symmetry::Bool) = symmetry
pyscf_symmetry_setting(symmetry) = symmetry

"""
    block_ranges(irreps) -> Vector{UnitRange{Int}}

Return contiguous index ranges for a symmetry-orbital ordering grouped by
irrep label.
"""
function block_ranges(irreps::Vector{Symbol})
    isempty(irreps) && return UnitRange{Int}[]

    ranges = UnitRange{Int}[]
    start = 1
    current = irreps[1]
    for idx in 2:length(irreps)
        if irreps[idx] != current
            push!(ranges, start:(idx - 1))
            start = idx
            current = irreps[idx]
        end
    end
    push!(ranges, start:length(irreps))
    return ranges
end

"""Return one contiguous block range per irrep in `info`."""
block_ranges(info::SymmetryInfo) = block_ranges(info.irreps)

"""Return symmetry orbitals per irrep (SOPI), i.e. block sizes."""
sopi(info::SymmetryInfo) = length.(block_ranges(info))

"""Return the irrep label for each contiguous block in `info`."""
irrep_block_names(info::SymmetryInfo) = [info.irreps[first(range)] for range in block_ranges(info)]

"""
    water_c2v_sto3g_symmetry() -> SymmetryInfo

Return the AO-to-SO transformation for the default STO-3G water orientation
used by `run_rhf`.  The SO order is grouped as A1, B1, B2 with block sizes
4, 1, and 2.  The hydrogen 1s symmetry orbitals include the 1/sqrt(2)
normalization from Crawford Project #9.
"""
function water_c2v_sto3g_symmetry()
    invsqrt2 = inv(sqrt(2.0))
    transform = zeros(Float64, 7, 7)

    # A1: O 1s, O 2s, O 2p along the C2 axis, symmetric H 1s combination
    transform[1,1] = 1.0
    transform[2,2] = 1.0
    transform[4,3] = 1.0
    transform[6,4] = invsqrt2
    transform[7,4] = invsqrt2

    # B1: O 2p perpendicular to the molecular plane
    transform[5,5] = 1.0

    # B2: O 2p across the H-H axis, antisymmetric H 1s combination
    transform[3,6] = 1.0
    transform[6,7] = invsqrt2
    transform[7,7] = -invsqrt2

    return SymmetryInfo(transform, [:A1, :A1, :A1, :A1, :B1, :B2, :B2], [3, 1, 1])
end

"""
    pyscf_symmetry_orbitals(mol) -> SymmetryInfo

Read PySCF's symmetry-adapted AO blocks from a molecule built with symmetry
enabled and concatenate them into this package's `SymmetryInfo` representation.
"""
function pyscf_symmetry_orbitals(mol)
    getattr = pybuiltin("getattr")
    as_list = pybuiltin("list")
    symm_orb_obj = pycall(getattr, PyObject, mol, "symm_orb")
    irrep_name_obj = pycall(getattr, PyObject, mol, "irrep_name")
    _keep_pyscf!(getattr, as_list, symm_orb_obj, irrep_name_obj)

    raw_blocks = pycall(as_list, PyAny, symm_orb_obj)
    _keep_pyscf!(raw_blocks...)
    blocks = [Matrix{Float64}(Array(block)) for block in raw_blocks]
    isempty(blocks) && throw(ArgumentError("PySCF molecule does not contain symmetry orbital blocks"))

    raw_names = pycall(as_list, PyAny, irrep_name_obj)
    _keep_pyscf!(raw_names...)
    irrep_names = [Symbol(String(name)) for name in raw_names]
    length(blocks) == length(irrep_names) ||
        throw(ArgumentError("PySCF symmetry orbital blocks and irrep names have different lengths"))

    transform = hcat(blocks...)
    irreps = Symbol[]
    for (name, block) in zip(irrep_names, blocks)
        append!(irreps, fill(name, size(block, 2)))
    end

    return SymmetryInfo(transform, irreps, Int[])
end

"""
    normalize_symmetry_occupations(occupations, info) -> Vector{Int}

Accept either a vector ordered by symmetry block or a dictionary keyed by irrep
name and return one closed-shell occupied-orbital count per block.
"""
function normalize_symmetry_occupations(occupations, info::SymmetryInfo)
    names = irrep_block_names(info)
    if occupations isa AbstractVector
        length(occupations) == length(names) ||
            throw(ArgumentError("Occupation vector must have one entry per symmetry block"))
        return Int.(occupations)
    elseif occupations isa AbstractDict
        return [Int(get(occupations, name, get(occupations, String(name), 0))) for name in names]
    else
        throw(ArgumentError("Symmetry occupations must be a vector or dictionary"))
    end
end

"""
    infer_symmetry_occupations(info, h1e, S, n_elec) -> Vector{Int}

Infer closed-shell irrep occupations by diagonalizing the core Hamiltonian in
each symmetry block and filling the lowest `n_elec/2` orbital energies.
"""
function infer_symmetry_occupations(info::SymmetryInfo, h1e, S, n_elec)
    iseven(n_elec) || throw(ArgumentError("Closed-shell symmetry occupation inference requires an even electron count"))
    nocc = n_elec ÷ 2

    h_so = symmetry_project(symmetry_transform_1e(h1e, info), info)
    S_so = symmetry_project(symmetry_transform_1e(S, info), info)
    candidates = Tuple{Float64,Int}[]

    for (block_index, range) in enumerate(block_ranges(info))
        values, vectors = eigen(Symmetric((S_so[range, range] .+ S_so[range, range]') ./ 2))
        X = vectors * Diagonal(values .^ -0.5) * vectors'
        F = X * h_so[range, range] * X'
        orbital_energies = eigvals(Symmetric((F .+ F') ./ 2))
        append!(candidates, [(energy, block_index) for energy in orbital_energies])
    end

    sort!(candidates; by=first)
    length(candidates) >= nocc ||
        throw(ArgumentError("Not enough symmetry orbitals to place $nocc occupied orbitals"))

    occupations = zeros(Int, length(block_ranges(info)))
    for (_, block_index) in candidates[1:nocc]
        occupations[block_index] += 1
    end
    return occupations
end

"""
    pyscf_symmetry_info(mol; h1e=nothing, S=nothing, n_elec=nothing, occupations=nothing)

Build `SymmetryInfo` from PySCF symmetry orbitals and either use explicit
occupations or infer them from a one-electron Hamiltonian and overlap.
"""
function pyscf_symmetry_info(mol; h1e=nothing, S=nothing, n_elec=nothing, occupations=nothing)
    info = pyscf_symmetry_orbitals(mol)
    resolved_occupations =
        if occupations !== nothing
            normalize_symmetry_occupations(occupations, info)
        elseif n_elec !== nothing && h1e !== nothing && S !== nothing
            infer_symmetry_occupations(info, h1e, S, n_elec)
        else
            Int[]
        end
    return SymmetryInfo(info.transform, info.irreps, resolved_occupations)
end

"""
    symmetry_transform_1e(matrix, info)

Transform a one-electron AO matrix to the symmetry-orbital basis:

```text
A_SO = T' A_AO T
```
"""
symmetry_transform_1e(matrix, info::SymmetryInfo) = info.transform' * matrix * info.transform

"""
    symmetry_backtransform_1e(matrix, info)

Transform a one-electron symmetry-orbital matrix back to the AO basis:

```text
A_AO = T A_SO T'
```
"""
symmetry_backtransform_1e(matrix, info::SymmetryInfo) = info.transform * matrix * info.transform'

"""
    symmetry_transform_eri(eri, info)

Transform AO ERIs to the symmetry-orbital basis:

```text
(pq|rs)_SO = sum_mnls T_mp T_nq (mn|ls)_AO T_lr T_ls
```
"""
function symmetry_transform_eri(eri, info::SymmetryInfo)
    T = info.transform
    @tensoropt eri_so[p,q,r,s] := T[i,p] * T[j,q] * eri[i,j,k,l] * T[k,r] * T[l,s]
    return eri_so
end

"""Extract diagonal symmetry blocks from a block-ordered matrix."""
function symmetry_blocks(matrix, info::SymmetryInfo)
    return [copy(matrix[range, range]) for range in block_ranges(info)]
end

"""Assemble diagonal blocks into one block-diagonal matrix."""
function assemble_symmetry_blocks(blocks)
    dim = sum(size(block, 1) for block in blocks)
    matrix = zeros(Float64, dim, dim)
    offset = 0
    for block in blocks
        n = size(block, 1)
        matrix[(offset + 1):(offset + n), (offset + 1):(offset + n)] .= block
        offset += n
    end
    return matrix
end

"""Zero all off-block matrix elements according to `info`."""
function symmetry_project(matrix, info::SymmetryInfo)
    projected = zeros(Float64, size(matrix))
    for range in block_ranges(info)
        projected[range, range] .= matrix[range, range]
    end
    return projected
end

"""Return the Frobenius norm of all off-block matrix elements."""
function symmetry_offblock_norm(matrix, info::SymmetryInfo)
    offblock = matrix .- symmetry_project(matrix, info)
    return norm(offblock)
end

"""Multiply matching block matrices pairwise."""
block_multiply(left_blocks, right_blocks) = [A * B for (A, B) in zip(left_blocks, right_blocks)]

"""
    block_s_half(S, info)

Build the block-diagonal symmetric orthogonalizer `S^{-1/2}` by diagonalizing
each symmetry block independently.
"""
function block_s_half(S, info::SymmetryInfo)
    X = zeros(Float64, size(S))
    for range in block_ranges(info)
        values, vectors = eigen(Symmetric((S[range, range] .+ S[range, range]') ./ 2))
        X[range, range] .= vectors * Diagonal(values .^ -0.5) * vectors'
    end
    return X
end

"""Diagonalize each symmetry block independently and assemble eigenvectors."""
function block_eigen(matrix, info::SymmetryInfo)
    values = zeros(Float64, size(matrix, 1))
    vectors = zeros(Float64, size(matrix))
    for range in block_ranges(info)
        block_values, block_vectors = eigen(Symmetric((matrix[range, range] .+ matrix[range, range]') ./ 2))
        values[range] .= block_values
        vectors[range, range] .= block_vectors
    end
    return values, vectors
end

"""Return occupied column indices implied by `info.occupations`."""
function occupied_columns(info::SymmetryInfo)
    ranges = block_ranges(info)
    length(info.occupations) == length(ranges) ||
        throw(ArgumentError("Occupation count must match the number of symmetry blocks"))

    columns = Int[]
    for (range, occupation) in zip(ranges, info.occupations)
        occupation <= length(range) ||
            throw(ArgumentError("Occupation $occupation exceeds block size $(length(range))"))
        append!(columns, first(range):(first(range) + occupation - 1))
    end
    return columns
end

"""
    make_density_symmetry(c, info)

Build a closed-shell density in the symmetry-orbital basis:

```text
D = C_occ C_occ'
```

and project away numerical off-block noise.
"""
function make_density_symmetry(c, info::SymmetryInfo)
    cols = occupied_columns(info)
    c_occ = @view c[:, cols]
    return symmetry_project(c_occ * c_occ', info)
end

"""
    make_fock_symmetry(D, h1e, eri, info)

Build the closed-shell Fock matrix in the symmetry-orbital basis:

```text
F_ij = h_ij + sum_kl D_kl [2 (ij|kl) - (ik|jl)]
```

Only symmetry-allowed block pairs are contracted.
"""
function make_fock_symmetry(D, h1e, eri, info::SymmetryInfo)
    nbasis = size(D, 1)
    ranges = block_ranges(info)
    F = zeros(Float64, nbasis, nbasis)

    for ij_range in ranges
        @threads for idx in 1:(length(ij_range) * length(ij_range))
            local_i = ((idx - 1) % length(ij_range)) + 1
            local_j = ((idx - 1) ÷ length(ij_range)) + 1
            i = ij_range[local_i]
            j = ij_range[local_j]
            value = 0.0
            @inbounds for kl_range in ranges
                for k in kl_range, l in kl_range
                    value += D[k,l] * (2 * eri[i,j,k,l] - eri[i,k,j,l])
                end
            end
            @inbounds F[i,j] = h1e[i,j] + value
        end
    end

    return symmetry_project(F, info)
end
