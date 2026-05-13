"""
FCI-reference DMRG/MPS helpers.

This file owns the exact small-system reference path that compresses an FCI
vector into an MPS for checking tensor contractions.  The custom HF-basis DMRG
optimizer lives in `custom_dmrg.jl`.
"""

using LinearAlgebra, Printf

"""
    OneBodyMPOTerm

One fermionic one-body Hamiltonian term `h_pq a_p^+ a_q` used by the
determinant-space `FermionMPO` path.
"""
struct OneBodyMPOTerm
    p::Int
    q::Int
    value::Float64
end

"""
    TwoBodyMPOTerm

One antisymmetrized two-body Hamiltonian term
`(1/4) <pq||rs> a_p^+ a_q^+ a_s a_r` used by the determinant-space
`FermionMPO` path.
"""
struct TwoBodyMPOTerm
    p::Int
    q::Int
    r::Int
    s::Int
    value::Float64
end

"""
    FermionMPO

Sparse second-quantized MPO representation used by the pedagogical DMRG code.
The Hamiltonian is stored as one- and two-body fermion strings.  When built from
an FCI result, the determinant-space Hamiltonian is also cached so two-site
effective contractions can use fast BLAS matrix products.
"""
struct FermionMPO
    nspin::Int
    n_elec::Int
    determinants::Vector{CISDDet}
    one_body::Vector{OneBodyMPOTerm}
    two_body::Vector{TwoBodyMPOTerm}
    hamiltonian::Union{Nothing, Matrix{Float64}}
end

function _explicit_fock_dimension(nspin::Integer)
    0 <= nspin <= 24 ||
        throw(ArgumentError("Explicit occupation-vector embedding supports at most 24 spin orbitals"))
    return 1 << Int(nspin)
end

function _determinant_index(determinants)
    return Dict(det => idx for (idx, det) in pairs(determinants))
end

function _build_mpo_terms(h_so, aso; integral_cutoff=1e-14)
    nspin = size(h_so, 1)
    size(h_so, 2) == nspin ||
        throw(DimensionMismatch("Spin-orbital one-electron matrix must be square"))
    size(aso) == (nspin, nspin, nspin, nspin) ||
        throw(DimensionMismatch("ASO tensor size does not match one-electron matrix"))

    one_body = OneBodyMPOTerm[]
    two_body = TwoBodyMPOTerm[]

    for q in 1:nspin, p in 1:nspin
        value = h_so[p,q]
        abs(value) <= integral_cutoff && continue
        push!(one_body, OneBodyMPOTerm(p, q, value))
    end

    for r in 1:nspin, s in 1:nspin, q in 1:nspin, p in 1:nspin
        value = 0.25 * aso[p,q,r,s]
        abs(value) <= integral_cutoff && continue
        push!(two_body, TwoBodyMPOTerm(p, q, r, s, value))
    end

    return (one_body=one_body, two_body=two_body)
end

"""
    build_dmrg_mpo(fci_result; nspin=size(fci_result.h_so, 1), n_elec=nothing)

Build the sparse fermionic MPO used by the DMRG helpers from an FCI result.
"""
function build_dmrg_mpo(fci_result; nspin=size(fci_result.h_so, 1),
                        n_elec=nothing, integral_cutoff=1e-14)
    determinants = CISDDet.(fci_result.determinants)
    isempty(determinants) && throw(ArgumentError("MPO needs at least one determinant"))
    electron_count = n_elec === nothing ? count_ones(determinants[1]) : Int(n_elec)
    terms = _build_mpo_terms(fci_result.h_so, fci_result.aso; integral_cutoff)

    return FermionMPO(
        Int(nspin),
        electron_count,
        determinants,
        terms.one_body,
        terms.two_body,
        Matrix{Float64}(fci_result.hamiltonian),
    )
end

"""
    mpo_apply(mpo, coefficients; use_cache=true)

Apply the sparse fermionic MPO to determinant-basis coefficients.
"""
function mpo_apply(mpo::FermionMPO, coefficients::AbstractVector; use_cache=true)
    length(coefficients) == length(mpo.determinants) ||
        throw(DimensionMismatch("Coefficient vector length must match MPO determinant count"))

    if use_cache && mpo.hamiltonian !== nothing
        return mpo.hamiltonian * coefficients
    end

    det_index = _determinant_index(mpo.determinants)
    result = zeros(Float64, length(mpo.determinants))

    for (col, det) in pairs(mpo.determinants)
        coefficient = coefficients[col]
        coefficient == 0.0 && continue

        for term in mpo.one_body
            applied = _apply_one_body(det, term.p, term.q)
            applied.ok || continue
            row = get(det_index, applied.det, 0)
            row == 0 && continue
            @inbounds result[row] += term.value * applied.phase * coefficient
        end

        for term in mpo.two_body
            applied = _apply_two_body(det, term.p, term.q, term.r, term.s)
            applied.ok || continue
            row = get(det_index, applied.det, 0)
            row == 0 && continue
            @inbounds result[row] += term.value * applied.phase * coefficient
        end
    end

    return result
end

function _bond_entropy(singular_values)
    norm2 = sum(abs2, singular_values)
    norm2 <= eps(Float64) && return 0.0
    entropy = 0.0
    for value in singular_values
        probability = value^2 / norm2
        probability <= eps(Float64) && continue
        entropy -= probability * log(probability)
    end
    return entropy
end

"""
    determinants_to_fock_vector(coefficients, determinants, nspin)

Embed fixed-electron determinant coefficients into the full occupation-number
Fock vector with local physical dimension 2 on each spin orbital.
"""
function determinants_to_fock_vector(coefficients, determinants, nspin::Integer)
    length(coefficients) == length(determinants) ||
        throw(DimensionMismatch("Number of coefficients must match determinant count"))

    dimension = _explicit_fock_dimension(nspin)
    state = zeros(Float64, dimension)
    highest_allowed = CISDDet(dimension - 1)

    for (coefficient, det) in zip(coefficients, determinants)
        det <= highest_allowed ||
            throw(ArgumentError("Determinant contains orbitals outside the requested spin-orbital space"))
        @inbounds state[Int(det) + 1] = coefficient
    end
    return state
end

"""
    fock_vector_to_determinants(state, determinants)

Project an occupation-number Fock vector back onto a determinant list.
"""
function fock_vector_to_determinants(state::AbstractVector, determinants)
    coefficients = zeros(Float64, length(determinants))
    for (idx, det) in pairs(determinants)
        state_index = Int(det) + 1
        state_index <= length(state) ||
            throw(DimensionMismatch("State vector is too short for determinant bit pattern"))
        @inbounds coefficients[idx] = state[state_index]
    end
    return coefficients
end

"""
    mps_decompose(state, nsites; max_bond=nothing, cutoff=1e-12)

Factor an occupation-number vector into left-canonical MPS tensors with shape
`(left_bond, 2, right_bond)` using sequential SVD.  Singular values below
`cutoff` and, if supplied, values beyond `max_bond` are discarded.

At each bond the reshaped state is factorized as

```text
Psi_(left),(right) = U S V^T.
```
"""
function mps_decompose(state::AbstractVector, nsites::Integer; max_bond=nothing, cutoff=1e-12)
    nsites >= 1 || throw(ArgumentError("MPS needs at least one site"))
    length(state) == 1 << Int(nsites) ||
        throw(DimensionMismatch("State length must equal 2^nsites"))
    cutoff >= 0.0 || throw(ArgumentError("cutoff must be nonnegative"))

    if max_bond !== nothing
        max_bond >= 1 || throw(ArgumentError("max_bond must be positive"))
        max_bond = Int(max_bond)
    end

    tensors = Array{Float64, 3}[]
    singular_values = Vector{Float64}[]
    discarded_weights = Float64[]

    left_bond = 1
    remainder = reshape(copy(Vector{Float64}(state)), 2, :)

    for site in 1:(Int(nsites) - 1)
        matrix = reshape(remainder, left_bond * 2, :)
        factorization = svd(matrix)

        cutoff_keep = count(value -> value > cutoff, factorization.S)
        keep = max(cutoff_keep, 1)
        if max_bond !== nothing
            keep = min(keep, max_bond)
        end
        keep = min(keep, length(factorization.S))

        kept_singular_values = factorization.S[1:keep]
        discarded = keep < length(factorization.S) ? sum(abs2, factorization.S[(keep + 1):end]) : 0.0

        push!(singular_values, copy(kept_singular_values))
        push!(discarded_weights, discarded)
        push!(tensors, reshape(factorization.U[:, 1:keep], left_bond, 2, keep))

        remainder = Diagonal(kept_singular_values) * factorization.Vt[1:keep, :]
        left_bond = keep
    end

    push!(tensors, reshape(remainder, left_bond, 2, 1))
    bond_dimensions = [size(tensor, 3) for tensor in tensors[1:(end - 1)]]
    entropies = [_bond_entropy(values) for values in singular_values]

    return (
        tensors = tensors,
        bond_dimensions = bond_dimensions,
        max_bond_dimension = isempty(bond_dimensions) ? 1 : maximum(bond_dimensions),
        singular_values = singular_values,
        discarded_weights = discarded_weights,
        discarded_weight = sum(discarded_weights),
        entanglement_entropies = entropies,
    )
end

"""
    mps_reconstruct(tensors)

Contract MPS tensors back to the full occupation-number state vector:

```text
Psi[n_1...n_L] = sum_{a_1...a_{L-1}}
    A_1[1,n_1,a_1] A_2[a_1,n_2,a_2] ... A_L[a_{L-1},n_L,1].
```
"""
function mps_reconstruct(tensors::AbstractVector)
    isempty(tensors) && throw(ArgumentError("At least one MPS tensor is required"))

    first_tensor = tensors[1]
    size(first_tensor, 1) == 1 ||
        throw(DimensionMismatch("First MPS tensor must have left bond dimension 1"))
    size(first_tensor, 2) == 2 ||
        throw(DimensionMismatch("MPS tensors must have physical dimension 2"))

    state = reshape(first_tensor[1, :, :], 2, size(first_tensor, 3))

    for site in 2:length(tensors)
        tensor = tensors[site]
        left_bond, physical_dim, right_bond = size(tensor)
        physical_dim == 2 ||
            throw(DimensionMismatch("MPS tensors must have physical dimension 2"))
        size(state, 2) == left_bond ||
            throw(DimensionMismatch("Adjacent MPS bond dimensions do not match"))

        previous_dim = size(state, 1)
        next_state = zeros(Float64, previous_dim * physical_dim, right_bond)
        @inbounds for basis in 1:previous_dim, left in 1:left_bond
            amplitude = state[basis, left]
            amplitude == 0.0 && continue
            for occ in 1:physical_dim, right in 1:right_bond
                next_state[basis + (occ - 1) * previous_dim, right] +=
                    amplitude * tensor[left, occ, right]
            end
        end
        state = next_state
    end

    size(state, 2) == 1 ||
        throw(DimensionMismatch("Last MPS tensor must have right bond dimension 1"))
    return vec(state)
end

function _check_two_site(tensors, site::Integer)
    1 <= site < length(tensors) ||
        throw(ArgumentError("site must select the first tensor in an adjacent two-site block"))
    size(tensors[site], 2) == 2 && size(tensors[site + 1], 2) == 2 ||
        throw(DimensionMismatch("MPS tensors must have physical dimension 2"))
    size(tensors[site], 3) == size(tensors[site + 1], 1) ||
        throw(DimensionMismatch("Adjacent MPS bond dimensions do not match"))
    return Int(site)
end

"""
    two_site_tensor(tensors, site)

Merge neighboring MPS tensors at `site` and `site + 1` into a two-site center
tensor with shape `(left_bond, 2, 2, right_bond)`:

```text
Theta[l,s1,s2,r] = sum_m A_site[l,s1,m] A_{site+1}[m,s2,r].
```
"""
function two_site_tensor(tensors::AbstractVector, site::Integer)
    site = _check_two_site(tensors, site)
    left_tensor = tensors[site]
    right_tensor = tensors[site + 1]
    left_bond, _, shared_bond = size(left_tensor)
    _, _, right_bond = size(right_tensor)

    theta = zeros(Float64, left_bond, 2, 2, right_bond)
    @inbounds for left in 1:left_bond, occ1 in 1:2, middle in 1:shared_bond,
                  occ2 in 1:2, right in 1:right_bond
        theta[left, occ1, occ2, right] +=
            left_tensor[left, occ1, middle] * right_tensor[middle, occ2, right]
    end
    return theta
end

"""
    split_two_site_tensor(theta; max_bond=nothing, cutoff=1e-12, direction=:right)

Split a two-site center tensor back into two MPS tensors by SVD.  `direction`
controls where the singular values are absorbed: `:right` for left-to-right
sweeps and `:left` for right-to-left sweeps.

```text
Theta_(l s1),(s2 r) = U S V^T.
```
"""
function split_two_site_tensor(theta::Array{Float64, 4}; max_bond=nothing,
                               cutoff=1e-12, direction=:right)
    cutoff >= 0.0 || throw(ArgumentError("cutoff must be nonnegative"))
    if max_bond !== nothing
        max_bond >= 1 || throw(ArgumentError("max_bond must be positive"))
        max_bond = Int(max_bond)
    end

    left_bond, phys1, phys2, right_bond = size(theta)
    matrix = reshape(theta, left_bond * phys1, phys2 * right_bond)
    factorization = svd(matrix)
    cutoff_keep = count(value -> value > cutoff, factorization.S)
    keep = max(cutoff_keep, 1)
    if max_bond !== nothing
        keep = min(keep, max_bond)
    end
    keep = min(keep, length(factorization.S))

    kept_singular_values = factorization.S[1:keep]
    discarded = keep < length(factorization.S) ? sum(abs2, factorization.S[(keep + 1):end]) : 0.0
    if direction === :right
        left_tensor = reshape(factorization.U[:, 1:keep], left_bond, phys1, keep)
        right_tensor = reshape(Diagonal(kept_singular_values) * factorization.Vt[1:keep, :],
                               keep, phys2, right_bond)
    elseif direction === :left
        left_tensor = reshape(factorization.U[:, 1:keep] * Diagonal(kept_singular_values),
                              left_bond, phys1, keep)
        right_tensor = reshape(factorization.Vt[1:keep, :], keep, phys2, right_bond)
    else
        throw(ArgumentError("direction must be :right or :left"))
    end

    return (
        left_tensor = left_tensor,
        right_tensor = right_tensor,
        singular_values = copy(kept_singular_values),
        discarded_weight = discarded,
    )
end

"""
    replace_two_site_tensor(tensors, site, theta; max_bond=nothing, cutoff=1e-12)

Return a copy of `tensors` where the adjacent two-site block has been replaced
by the SVD split of `theta`.
"""
function replace_two_site_tensor(tensors::AbstractVector, site::Integer,
                                 theta::Array{Float64, 4};
                                 max_bond=nothing, cutoff=1e-12, direction=:right)
    site = _check_two_site(tensors, site)
    split = split_two_site_tensor(theta; max_bond, cutoff, direction)
    updated = copy(tensors)
    updated[site] = split.left_tensor
    updated[site + 1] = split.right_tensor
    return (tensors=updated,
            singular_values=split.singular_values,
            discarded_weight=split.discarded_weight)
end

function _physical_index(det::CISDDet, site::Integer)
    return det & _orbital_bit(site) == 0 ? 1 : 2
end

function _left_block_vector(tensors, det::CISDDet, site::Integer)
    vector = [1.0]
    for tensor_site in 1:(site - 1)
        tensor = tensors[tensor_site]
        occ = _physical_index(det, tensor_site)
        next_vector = zeros(Float64, size(tensor, 3))
        @inbounds for left in 1:size(tensor, 1), right in 1:size(tensor, 3)
            next_vector[right] += vector[left] * tensor[left, occ, right]
        end
        vector = next_vector
    end
    return vector
end

function _right_block_vector(tensors, det::CISDDet, site::Integer)
    vector = [1.0]
    for tensor_site in length(tensors):-1:(site + 2)
        tensor = tensors[tensor_site]
        occ = _physical_index(det, tensor_site)
        next_vector = zeros(Float64, size(tensor, 1))
        @inbounds for left in 1:size(tensor, 1), right in 1:size(tensor, 3)
            next_vector[left] += tensor[left, occ, right] * vector[right]
        end
        vector = next_vector
    end
    return vector
end

"""
    two_site_projection_matrix(tensors, determinants, site)

Build the linear map from the flattened two-site tensor at `site, site + 1` to
the determinant coefficients represented by the surrounding MPS environments:

```text
C_I = sum_x P_{I x} theta_x.
```
"""
function two_site_projection_matrix(tensors::AbstractVector, determinants, site::Integer)
    site = _check_two_site(tensors, site)
    left_bond = size(tensors[site], 1)
    right_bond = size(tensors[site + 1], 3)
    center_dimension = left_bond * 2 * 2 * right_bond
    projection = zeros(Float64, length(determinants), center_dimension)

    @inbounds for (row, det) in pairs(determinants)
        left_vector = _left_block_vector(tensors, det, site)
        right_vector = _right_block_vector(tensors, det, site)
        occ1 = _physical_index(det, site)
        occ2 = _physical_index(det, site + 1)

        for left in 1:left_bond, right in 1:right_bond
            col = left +
                  (occ1 - 1) * left_bond +
                  (occ2 - 1) * left_bond * 2 +
                  (right - 1) * left_bond * 2 * 2
            projection[row, col] = left_vector[left] * right_vector[right]
        end
    end

    return projection
end

"""
    two_site_tensor_contract(mpo, tensors, site, theta=two_site_tensor(tensors, site))

Apply the MPO effective Hamiltonian to a two-site tensor.  The surrounding MPS
blocks define the left and right environments, and the contraction is returned
with the same shape as `theta`:

```text
sigma = P' H P theta.
```
"""
function two_site_tensor_contract(mpo::FermionMPO, tensors::AbstractVector,
                                  site::Integer,
                                  theta::Array{Float64, 4}=two_site_tensor(tensors, site);
                                  use_cache=true)
    site = _check_two_site(tensors, site)
    size(theta) == (size(tensors[site], 1), 2, 2, size(tensors[site + 1], 3)) ||
        throw(DimensionMismatch("Two-site tensor shape does not match selected MPS block"))

    projection = two_site_projection_matrix(tensors, mpo.determinants, site)
    coefficients = projection * vec(theta)
    sigma_coefficients = mpo_apply(mpo, coefficients; use_cache)
    return reshape(projection' * sigma_coefficients, size(theta))
end

"""
    two_site_energy(mpo, tensors, site, theta=two_site_tensor(tensors, site))

Evaluate the electronic energy of a two-site tensor in its current MPS
environment:

```text
E = (theta' P' H P theta) / (theta' P' P theta).
```
"""
function two_site_energy(mpo::FermionMPO, tensors::AbstractVector, site::Integer,
                         theta::Array{Float64, 4}=two_site_tensor(tensors, site);
                         use_cache=true)
    site = _check_two_site(tensors, site)
    projection = two_site_projection_matrix(tensors, mpo.determinants, site)
    coefficients = projection * vec(theta)
    norm2 = dot(coefficients, coefficients)
    norm2 > eps(Float64) ||
        throw(ArgumentError("Two-site tensor has zero norm in the determinant projection"))
    sigma_coefficients = mpo_apply(mpo, coefficients; use_cache)
    return dot(coefficients, sigma_coefficients) / norm2
end

"""
    mpo_expectation(mpo, coefficients; use_cache=true) -> Float64

Evaluate a determinant-space expectation value:

```text
<H> = (c' H c) / (c' c).
```
"""
function mpo_expectation(mpo::FermionMPO, coefficients::AbstractVector{<:Real}; use_cache=true)
    norm2 = dot(coefficients, coefficients)
    norm2 > eps(Float64) ||
        throw(ArgumentError("Coefficient vector has zero norm"))
    return dot(coefficients, mpo_apply(mpo, coefficients; use_cache)) / norm2
end

"""
    mpo_expectation(mpo, tensors; use_cache=true) -> Float64

Reconstruct determinant coefficients from an MPS and evaluate
`(c' H c)/(c' c)` with the determinant-space MPO.
"""
function mpo_expectation(mpo::FermionMPO, tensors::AbstractVector{<:AbstractArray{<:Real, 3}};
                         use_cache=true)
    coefficients = fock_vector_to_determinants(mps_reconstruct(tensors), mpo.determinants)
    return mpo_expectation(mpo, coefficients; use_cache)
end

"""
    run_fci_dmrg(rhf, mp2_result; fci_result=nothing, max_bond=nothing, cutoff=1e-12)

Build an MPS representation of the FCI wavefunction and evaluate the energy of
the reconstructed fixed-electron wavefunction.  Supplying an existing
`fci_result` avoids rebuilding and rediagonalizing the determinant Hamiltonian.
This is an exact small-system reference, not the production DMRG route.
"""
function run_fci_dmrg(rhf, mp2_result; fci_result=nothing, max_bond=nothing,
                      cutoff=1e-12, integral_cutoff=1e-14, verbose=true)
    fci = fci_result === nothing ?
        run_fci(rhf, mp2_result; nroots=1, integral_cutoff, verbose=false) :
        fci_result

    nspin = 2 * rhf.nbasis
    n_elec = rhf.n_elec

    fock_state = determinants_to_fock_vector(fci.ci_vector, fci.determinants, nspin)
    mps = mps_decompose(fock_state, nspin; max_bond, cutoff)
    reconstructed = mps_reconstruct(mps.tensors)

    projected_coefficients = fock_vector_to_determinants(reconstructed, fci.determinants)
    projected_norm2 = dot(projected_coefficients, projected_coefficients)
    projected_norm2 > eps(Float64) ||
        throw(ArgumentError("Reconstructed MPS has no weight in the target electron-number sector"))
    normalized_coefficients = projected_coefficients ./ sqrt(projected_norm2)

    mpo = build_dmrg_mpo(fci; nspin, n_elec, integral_cutoff)
    electronic_energy = mpo_expectation(mpo, normalized_coefficients)
    nuclear_repulsion = rhf.total_energy - rhf.energy
    total_energy = electronic_energy + nuclear_repulsion
    correlation_energy = total_energy - rhf.total_energy

    reconstructed_norm2 = dot(reconstructed, reconstructed)
    sector_weight = projected_norm2 / max(reconstructed_norm2, eps(Float64))
    reconstruction_error = norm(reconstructed - fock_state)
    two_site_site = max(1, min(nspin - 1, nspin ÷ 2))
    theta = two_site_tensor(mps.tensors, two_site_site)
    two_site_electronic_energy = two_site_energy(mpo, mps.tensors, two_site_site, theta)

    verbose && begin
        @printf("DMRG/MPS sites            = %d\n", nspin)
        @printf("DMRG/MPS max bond         = %d\n", mps.max_bond_dimension)
        @printf("DMRG discarded weight     = %.6e\n", mps.discarded_weight)
        @printf("DMRG reconstruction error = %.6e\n", reconstruction_error)
        @printf("DMRG two-site contraction = site %d  E=%20.12f Eh\n",
                two_site_site, two_site_electronic_energy)
        @printf("DMRG correlation energy   = %20.12f Eh\n", correlation_energy)
        @printf("DMRG total energy         = %20.12f Eh\n", total_energy)
    end

    return (
        E_dmrg = correlation_energy,
        total_energy = total_energy,
        electronic_energy = electronic_energy,
        ci_vector = normalized_coefficients,
        determinant_coefficients = projected_coefficients,
        fock_state = fock_state,
        reconstructed_state = reconstructed,
        tensors = mps.tensors,
        mps = mps.tensors,
        bond_dimensions = mps.bond_dimensions,
        max_bond_dimension = mps.max_bond_dimension,
        singular_values = mps.singular_values,
        entanglement_entropies = mps.entanglement_entropies,
        discarded_weights = mps.discarded_weights,
        discarded_weight = mps.discarded_weight,
        reconstruction_error = reconstruction_error,
        sector_weight = sector_weight,
        projected_norm = sqrt(projected_norm2),
        mpo = mpo,
        two_site_site = two_site_site,
        two_site_tensor = theta,
        two_site_electronic_energy = two_site_electronic_energy,
        fci_energy = fci.E_fci,
        fci_total_energy = fci.total_energy,
        determinants = fci.determinants,
        hamiltonian = fci.hamiltonian,
        dimension = fci.dimension,
        nspin = nspin,
        n_elec = n_elec,
        cutoff = cutoff,
        max_bond = max_bond,
    )
end
