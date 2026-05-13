"""
Configuration interaction with singles and doubles (CISD).

Project #19 builds a determinant-space Hamiltonian from the Hartree-Fock
reference plus all single and double spin-orbital excitations.
"""

using LinearAlgebra, Printf
using Base.Threads: @threads

const CISDDet = UInt64

function _orbital_bit(orbital::Integer)
    1 <= orbital <= 63 ||
        throw(ArgumentError("CISD determinant bit representation supports orbitals 1:63"))
    return CISDDet(1) << (orbital - 1)
end

"""
    determinant_from_orbitals(orbitals) -> CISDDet

Encode a Slater determinant as a bit string.  Orbital `p` occupies bit
`p - 1`, so the determinant for occupied spin orbitals `I` is

```text
|I>  <=>  sum_{p in I} 2^(p - 1).
```
"""
function determinant_from_orbitals(orbitals)
    det = CISDDet(0)
    for orbital in orbitals
        bit = _orbital_bit(orbital)
        det & bit == 0 ||
            throw(ArgumentError("Orbital $orbital appears more than once"))
        det |= bit
    end
    return det
end

"""
    occupied_orbitals(det, nspin) -> Vector{Int}

Decode the occupied spin-orbital labels in a bit-string determinant.
"""
function occupied_orbitals(det::CISDDet, nspin::Integer)
    return [orbital for orbital in 1:nspin if det & _orbital_bit(orbital) != 0]
end

function _fermion_phase(det::CISDDet, orbital::Integer)
    mask = _orbital_bit(orbital) - CISDDet(1)
    return isodd(count_ones(det & mask)) ? -1.0 : 1.0
end

function _annihilate(det::CISDDet, orbital::Integer)
    bit = _orbital_bit(orbital)
    det & bit == 0 && return (ok=false, det=det, phase=0.0)
    return (ok=true, det=det ⊻ bit, phase=_fermion_phase(det, orbital))
end

function _create(det::CISDDet, orbital::Integer)
    bit = _orbital_bit(orbital)
    det & bit != 0 && return (ok=false, det=det, phase=0.0)
    return (ok=true, det=det | bit, phase=_fermion_phase(det, orbital))
end

function _apply_one_body(det::CISDDet, p::Integer, q::Integer)
    step1 = _annihilate(det, q)
    step1.ok || return (ok=false, det=det, phase=0.0)
    step2 = _create(step1.det, p)
    step2.ok || return (ok=false, det=det, phase=0.0)
    return (ok=true, det=step2.det, phase=step1.phase * step2.phase)
end

function _apply_two_body(det::CISDDet, p::Integer, q::Integer, r::Integer, s::Integer)
    step1 = _annihilate(det, r)
    step1.ok || return (ok=false, det=det, phase=0.0)
    step2 = _annihilate(step1.det, s)
    step2.ok || return (ok=false, det=det, phase=0.0)
    step3 = _create(step2.det, q)
    step3.ok || return (ok=false, det=det, phase=0.0)
    step4 = _create(step3.det, p)
    step4.ok || return (ok=false, det=det, phase=0.0)
    return (ok=true,
            det=step4.det,
            phase=step1.phase * step2.phase * step3.phase * step4.phase)
end

"""
    cisd_determinants(nspin, n_elec) -> (determinants, labels)

Generate the reference determinant and all spin-orbital single and double
excitations from the canonical occupied/virtual partition:

```text
|Phi_i^a>   = a_a^+ a_i |Phi_0>
|Phi_ij^ab> = a_a^+ a_b^+ a_j a_i |Phi_0>.
```
"""
function cisd_determinants(nspin::Integer, n_elec::Integer)
    0 <= n_elec <= nspin ||
        throw(ArgumentError("n_elec must be between 0 and nspin"))

    occupied = collect(1:n_elec)
    virtual = collect((n_elec + 1):nspin)
    reference = determinant_from_orbitals(occupied)

    determinants = CISDDet[reference]
    labels = NamedTuple{(:rank, :holes, :particles), Tuple{Int, Tuple, Tuple}}[
        (rank=0, holes=(), particles=())
    ]

    for i in occupied, a in virtual
        det = (reference ⊻ _orbital_bit(i)) | _orbital_bit(a)
        push!(determinants, det)
        push!(labels, (rank=1, holes=(i,), particles=(a,)))
    end

    for ix in 1:(length(occupied) - 1), jx in (ix + 1):length(occupied)
        i = occupied[ix]
        j = occupied[jx]
        for ax in 1:(length(virtual) - 1), bx in (ax + 1):length(virtual)
            a = virtual[ax]
            b = virtual[bx]
            det = reference
            det ⊻= _orbital_bit(i)
            det ⊻= _orbital_bit(j)
            det |= _orbital_bit(a)
            det |= _orbital_bit(b)
            push!(determinants, det)
            push!(labels, (rank=2, holes=(i, j), particles=(a, b)))
        end
    end

    return (determinants=determinants, labels=labels)
end

"""
    spatial_to_spinorbital_1e(h_mo) -> Matrix{Float64}

Lift a spatial one-electron matrix to the interleaved alpha/beta spin-orbital
basis used by the RHF spin-orbital code:

```text
h_{p alpha,q alpha} = h_{p beta,q beta} = h_pq
h_{p alpha,q beta} = h_{p beta,q alpha} = 0.
```
"""
function spatial_to_spinorbital_1e(h_mo)
    nbasis = size(h_mo, 1)
    size(h_mo, 2) == nbasis ||
        throw(DimensionMismatch("One-electron matrix must be square"))

    h_so = zeros(Float64, 2 * nbasis, 2 * nbasis)
    @threads for idx in 1:(nbasis * nbasis)
        p = ((idx - 1) % nbasis) + 1
        q = ((idx - 1) ÷ nbasis) + 1
        @inbounds begin
            h_so[2p - 1, 2q - 1] = h_mo[p,q]
            h_so[2p, 2q] = h_mo[p,q]
        end
    end
    return h_so
end

"""
    build_cisd_hamiltonian(h_so, aso, determinants; integral_cutoff=1e-14)

Build the electronic Hamiltonian matrix in a determinant basis:

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```
"""
function build_cisd_hamiltonian(h_so, aso, determinants; integral_cutoff=1e-14)
    nspin = size(h_so, 1)
    size(h_so, 2) == nspin ||
        throw(DimensionMismatch("Spin-orbital one-electron matrix must be square"))
    size(aso) == (nspin, nspin, nspin, nspin) ||
        throw(DimensionMismatch("ASO tensor size does not match one-electron matrix"))

    ndet = length(determinants)
    det_index = Dict(det => idx for (idx, det) in pairs(determinants))
    H = zeros(Float64, ndet, ndet)

    for (col, det) in pairs(determinants)
        for q in 1:nspin, p in 1:nspin
            abs(h_so[p,q]) <= integral_cutoff && continue
            applied = _apply_one_body(det, p, q)
            applied.ok || continue
            row = get(det_index, applied.det, 0)
            row == 0 && continue
            H[row,col] += h_so[p,q] * applied.phase
        end

        for r in 1:nspin, s in 1:nspin, q in 1:nspin, p in 1:nspin
            value = 0.25 * aso[p,q,r,s]
            abs(value) <= integral_cutoff && continue
            applied = _apply_two_body(det, p, q, r, s)
            applied.ok || continue
            row = get(det_index, applied.det, 0)
            row == 0 && continue
            H[row,col] += value * applied.phase
        end
    end

    return (H .+ H') ./ 2
end

"""
    run_cisd(rhf, mp2_result; nroots=5, verbose=true)

Run determinant-space CISD from an RHF reference and MO integrals.  The
Hamiltonian is diagonalized in the reference plus singles/doubles determinant
space,

```text
H C_k = E_k C_k
E_corr = E_0 + E_nuc - E_RHF.
```
"""
function run_cisd(rhf, mp2_result; nroots=5, verbose=true)
    nspin = 2 * rhf.nbasis
    n_elec = rhf.n_elec
    dets = cisd_determinants(nspin, n_elec)

    h_mo = Mat_aotoMat_mo(rhf.mo_coeffs, rhf.h1e)
    h_so = spatial_to_spinorbital_1e(h_mo)
    aso = mo_to_aso(mp2_result.new_eri)
    H = build_cisd_hamiltonian(h_so, aso, dets.determinants)

    eig = eigen(Symmetric(H))
    roots = min(nroots, length(eig.values))
    nuclear_repulsion = rhf.total_energy - rhf.energy
    total_energies = eig.values[1:roots] .+ nuclear_repulsion
    correlation_energies = total_energies .- rhf.total_energy

    singles = count(label -> label.rank == 1, dets.labels)
    doubles = count(label -> label.rank == 2, dets.labels)
    verbose && begin
        @printf("CISD dimension          = %d (%d singles, %d doubles + reference)\n",
                length(dets.determinants), singles, doubles)
        @printf("CISD correlation energy = %20.12f Eh\n", correlation_energies[1])
        @printf("CISD total energy       = %20.12f Eh\n", total_energies[1])
    end

    return (
        E_cisd = correlation_energies[1],
        total_energy = total_energies[1],
        electronic_energy = eig.values[1],
        electronic_energies = eig.values[1:roots],
        total_energies = total_energies,
        correlation_energies = correlation_energies,
        ci_vector = eig.vectors[:, 1],
        ci_vectors = eig.vectors[:, 1:roots],
        reference_weight = eig.vectors[1, 1]^2,
        hamiltonian = H,
        determinants = dets.determinants,
        labels = dets.labels,
        dimension = length(dets.determinants),
        singles = singles,
        doubles = doubles,
        h_so = h_so,
        aso = aso,
    )
end
