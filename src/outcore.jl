"""
Out-of-core AO electron-repulsion integral storage and Fock builds.

Project #11 stores only permutationally unique Mulliken integrals on disk and
streams them during each SCF Fock build.
"""

const ERI_FILE_MAGIC = UInt32(0x51434552)  # "QCER"
const ERI_FILE_VERSION = UInt32(1)

"""
    ERIFile

Metadata for a disk-backed unique-ERI file.

Fields:
- `path`: binary file path.
- `nbasis`: AO basis dimension.
- `nrecords`: number of stored unique integral records.
- `cutoff`: absolute-value threshold used when the file was written.
"""
struct ERIFile
    path::String
    nbasis::Int
    nrecords::Int
    cutoff::Float64
end

"""
    compound_index(i, j) -> Int

One-based packed index for an unordered AO pair:

```text
ij = max(i,j) * (max(i,j) - 1) / 2 + min(i,j).
```
This is the Mulliken pair index used to identify permutationally unique ERIs.
"""
compound_index(i::Integer, j::Integer) =
    i >= j ? (i * (i - 1)) ÷ 2 + j : (j * (j - 1)) ÷ 2 + i

function _pair_indices(nbasis::Integer)
    pairs = Vector{NTuple{2,Int}}(undef, nbasis * (nbasis + 1) ÷ 2)
    idx = 0
    for i in 1:nbasis, j in 1:i
        idx += 1
        pairs[idx] = (i, j)
    end
    return pairs
end

_is_unique_eri_index(i, j, k, l) =
    i >= j && k >= l && compound_index(i, j) >= compound_index(k, l)

function _write_eri_header(io, nbasis, nrecords, cutoff)
    write(io, ERI_FILE_MAGIC)
    write(io, ERI_FILE_VERSION)
    write(io, Int64(nbasis))
    write(io, Int64(nrecords))
    write(io, Float64(cutoff))
    return nothing
end

function _read_eri_header(io, path)
    magic = read(io, UInt32)
    version = read(io, UInt32)
    magic == ERI_FILE_MAGIC ||
        throw(ArgumentError("$path is not a QuantumChem ERI file"))
    version == ERI_FILE_VERSION ||
        throw(ArgumentError("Unsupported ERI file version $version"))

    nbasis = Int(read(io, Int64))
    nrecords = Int(read(io, Int64))
    cutoff = read(io, Float64)
    return ERIFile(path, nbasis, nrecords, cutoff)
end

"""
    read_eri_file_header(path) -> ERIFile

Read metadata for a disk-backed unique-ERI file.
"""
function read_eri_file_header(path::AbstractString)
    open(path, "r") do io
        return _read_eri_header(io, String(path))
    end
end

function _write_eri_record(io, i, j, k, l, value)
    write(io, Int32(i))
    write(io, Int32(j))
    write(io, Int32(k))
    write(io, Int32(l))
    write(io, Float64(value))
    return nothing
end

function _write_unique_eri_file_from_array(eri, path, cutoff)
    nbasis = size(eri, 1)
    size(eri) == (nbasis, nbasis, nbasis, nbasis) ||
        throw(DimensionMismatch("ERI tensor must be a square rank-4 AO tensor"))

    pairs = _pair_indices(nbasis)
    nrecords = 0
    open(path, "w") do io
        _write_eri_header(io, nbasis, 0, cutoff)
        for ij in eachindex(pairs)
            i, j = pairs[ij]
            for kl in 1:ij
                k, l = pairs[kl]
                value = eri[i,j,k,l]
                abs(value) <= cutoff && continue
                _write_eri_record(io, i, j, k, l, value)
                nrecords += 1
            end
        end
        seekstart(io)
        _write_eri_header(io, nbasis, nrecords, cutoff)
    end
    return ERIFile(String(path), nbasis, nrecords, Float64(cutoff))
end

function _write_unique_eri_file_from_mol(mol::PyObject, path, cutoff)
    nbasis = Int(mol.nao_nr())
    nshell = Int(mol.nbas)
    ao_loc = Int.(mol.ao_loc_nr())
    intor_by_shell = mol.intor_by_shell
    _keep_pyscf!(intor_by_shell)

    nrecords = 0
    open(path, "w") do io
        _write_eri_header(io, nbasis, 0, cutoff)
        for si in 0:(nshell - 1), sj in 0:(nshell - 1),
            sk in 0:(nshell - 1), sl in 0:(nshell - 1)

            raw_block = pycall(intor_by_shell, PyObject, "int2e", (si, sj, sk, sl))
            pyblock = PyArray(raw_block)
            _keep_pyscf!(raw_block, pyblock)
            block = Array{Float64,4}(Array(pyblock))
            for a in axes(block, 1), b in axes(block, 2),
                c in axes(block, 3), d in axes(block, 4)

                i = ao_loc[si + 1] + a
                j = ao_loc[sj + 1] + b
                k = ao_loc[sk + 1] + c
                l = ao_loc[sl + 1] + d
                _is_unique_eri_index(i, j, k, l) || continue

                value = block[a,b,c,d]
                abs(value) <= cutoff && continue
                _write_eri_record(io, i, j, k, l, value)
                nrecords += 1
            end
        end
        seekstart(io)
        _write_eri_header(io, nbasis, nrecords, cutoff)
    end
    return ERIFile(String(path), nbasis, nrecords, Float64(cutoff))
end

"""
    write_unique_eri_file(eri_or_mol; path=tempname(), cutoff=1e-14) -> ERIFile

Write the permutationally unique AO ERIs to a compact binary file.  The source
can be either a full four-index AO tensor or a PySCF molecule, in which case
shell quartets are requested one at a time.
"""
function write_unique_eri_file(eri::AbstractArray{<:Real,4}; path=tempname(), cutoff=1e-14)
    return _write_unique_eri_file_from_array(eri, String(path), Float64(cutoff))
end

function write_unique_eri_file(mol::PyObject; path=tempname(), cutoff=1e-14)
    return _write_unique_eri_file_from_mol(mol, String(path), Float64(cutoff))
end

function _eri_permutations(i, j, k, l)
    candidates = NTuple{4,Int}[
        (i, j, k, l),
        (j, i, k, l),
        (i, j, l, k),
        (j, i, l, k),
        (k, l, i, j),
        (l, k, i, j),
        (k, l, j, i),
        (l, k, j, i),
    ]
    return unique(candidates)
end

function _accumulate_fock_integral!(F, D, i, j, k, l, value)
    for (p, q, r, s) in _eri_permutations(i, j, k, l)
        @inbounds begin
            F[p,q] += 2.0 * D[r,s] * value
            F[p,r] -= D[q,s] * value
        end
    end
    return nothing
end

"""
    make_fock_outcore(D, h1e, eri_file; batch_size=4096) -> Matrix{Float64}

Build the closed-shell AO Fock matrix by streaming unique ERI records from disk.
For each restored integral permutation, the code accumulates

```text
F_pq += 2 D_rs (pq|rs)
F_pr -=   D_qs (pq|rs),
```

which gives the RHF expression `F = h + 2J - K` under this package's
closed-shell density convention.
"""
function make_fock_outcore(D, h1e, eri_file::ERIFile; batch_size=4096)
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    nbasis = size(D, 1)
    size(D) == (nbasis, nbasis) || throw(DimensionMismatch("Density matrix must be square"))
    size(h1e) == (nbasis, nbasis) || throw(DimensionMismatch("Core Hamiltonian size does not match density"))
    eri_file.nbasis == nbasis ||
        throw(DimensionMismatch("ERI file basis size $(eri_file.nbasis) does not match matrix size $nbasis"))

    F = copy(h1e)
    open(eri_file.path, "r") do io
        header = _read_eri_header(io, eri_file.path)
        header.nbasis == eri_file.nbasis && header.nrecords == eri_file.nrecords ||
            throw(ArgumentError("ERI file metadata changed since it was opened"))

        remaining = header.nrecords
        while remaining > 0
            count = min(batch_size, remaining)
            for _ in 1:count
                i = Int(read(io, Int32))
                j = Int(read(io, Int32))
                k = Int(read(io, Int32))
                l = Int(read(io, Int32))
                value = read(io, Float64)
                _accumulate_fock_integral!(F, D, i, j, k, l, value)
            end
            remaining -= count
        end
    end
    return (F .+ F') ./ 2
end

make_fock_outcore(D, h1e, path::AbstractString; batch_size=4096) =
    make_fock_outcore(D, h1e, read_eri_file_header(path); batch_size)
