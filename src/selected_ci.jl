"""
Selected configuration interaction with a heat-bath-style determinant
selection criterion.

Project #21 builds a variational determinant space iteratively.  Starting from
the Hartree-Fock determinant, connected single and double excitations are added
when `abs(H_ai * c_i) > epsilon1` for at least one determinant already in the
variational space.  A deterministic Epstein-Nesbet PT2 correction can then be
evaluated over the remaining connected external determinants.
"""

using LinearAlgebra, Printf
using Base.Threads: @threads

function _orbitals_from_bits(bits::CISDDet, nspin::Integer)
    return [orbital for orbital in 1:nspin if bits & _orbital_bit(orbital) != 0]
end

function _excitation_between(bra::CISDDet, ket::CISDDet, nspin::Integer)
    holes = _orbitals_from_bits(ket & ~bra, nspin)
    particles = _orbitals_from_bits(bra & ~ket, nspin)
    return (rank=length(holes), holes=holes, particles=particles)
end

"""
    connected_determinants(det, nspin) -> Vector{CISDDet}

Generate all determinants connected to `det` by the one- and two-body
electronic Hamiltonian.  The Hamiltonian

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```

can only connect determinants that differ by zero, one, or two spin orbitals.
"""
function connected_determinants(det::CISDDet, nspin::Integer)
    occupied = occupied_orbitals(det, nspin)
    occupied_set = Set(occupied)
    virtual = [orbital for orbital in 1:nspin if !(orbital in occupied_set)]

    determinants = CISDDet[]
    for i in occupied, a in virtual
        push!(determinants, (det ⊻ _orbital_bit(i)) | _orbital_bit(a))
    end

    for ix in 1:(length(occupied) - 1), jx in (ix + 1):length(occupied)
        i = occupied[ix]
        j = occupied[jx]
        for ax in 1:(length(virtual) - 1), bx in (ax + 1):length(virtual)
            a = virtual[ax]
            b = virtual[bx]
            push!(determinants,
                  (((det ⊻ _orbital_bit(i)) ⊻ _orbital_bit(j)) |
                   _orbital_bit(a) | _orbital_bit(b)))
        end
    end
    return determinants
end

function _diagonal_hamiltonian_element(h_so, aso, det::CISDDet, nspin::Integer)
    occ = occupied_orbitals(det, nspin)
    value = 0.0
    @inbounds for i in occ
        value += h_so[i,i]
    end
    @inbounds for i in occ, j in occ
        value += 0.5 * aso[i,j,i,j]
    end
    return value
end

"""
    hamiltonian_element(h_so, aso, bra, ket) -> Float64

Evaluate `<bra|H|ket>` for determinants represented as bit strings.  Only
determinants differing by zero, one, or two spin orbitals can couple through
the electronic Hamiltonian.  The implemented Slater-Condon forms are:

```text
<I|H|I>       = sum_i h_ii + 1/2 sum_ij <ij||ij>
<I_i^a|H|I>   = h_ai + sum_j <aj||ij>
<I_ij^ab|H|I> = <ab||ij>.
```

Fermionic creation/annihilation phases are applied by the bit-string operator
helpers.
"""
function hamiltonian_element(h_so, aso, bra::CISDDet, ket::CISDDet; integral_cutoff=1e-14)
    nspin = size(h_so, 1)
    excitation = _excitation_between(bra, ket, nspin)
    excitation.rank > 2 && return 0.0

    if excitation.rank == 0
        return _diagonal_hamiltonian_element(h_so, aso, ket, nspin)
    elseif excitation.rank == 1
        p = excitation.particles[1]
        q = excitation.holes[1]
        applied = _apply_one_body(ket, p, q)
        applied.ok && applied.det == bra || return 0.0

        value = h_so[p,q] * applied.phase
        common = occupied_orbitals(bra & ket, nspin)
        @inbounds for j in common
            value += aso[p,j,q,j] * applied.phase
        end
        abs(value) <= integral_cutoff && return 0.0
        return value
    else
        particles = excitation.particles
        holes = excitation.holes
        value = 0.0
        @inbounds for p in particles, q in particles, r in holes, s in holes
            integral = 0.25 * aso[p,q,r,s]
            abs(integral) <= integral_cutoff && continue
            applied = _apply_two_body(ket, p, q, r, s)
            applied.ok && applied.det == bra || continue
            value += integral * applied.phase
        end
        abs(value) <= integral_cutoff && return 0.0
        return value
    end
end

"""
    build_selected_ci_hamiltonian(h_so, aso, determinants)

Build a determinant-space Hamiltonian for any fixed-electron determinant list.
The returned matrix has elements `H_IJ = <I|H|J>` and is symmetrized before it
is returned.
"""
function build_selected_ci_hamiltonian(h_so, aso, determinants; integral_cutoff=1e-14)
    ndet = length(determinants)
    H = zeros(Float64, ndet, ndet)
    @threads for idx in eachindex(H)
        row = ((idx - 1) % ndet) + 1
        col = ((idx - 1) ÷ ndet) + 1
        @inbounds H[row,col] = hamiltonian_element(
            h_so, aso, determinants[row], determinants[col]; integral_cutoff
        )
    end
    return (H .+ H') ./ 2
end

function _hci_candidate_scores(h_so, aso, determinants, coefficients, nspin;
                               epsilon1=1e-3, coefficient_cutoff=1e-12,
                               integral_cutoff=1e-14)
    selected = Set(determinants)
    scores = Dict{CISDDet, Float64}()

    for (col, det) in pairs(determinants)
        coefficient = coefficients[col]
        abs(coefficient) <= coefficient_cutoff && continue
        for candidate in connected_determinants(det, nspin)
            candidate in selected && continue
            hij = hamiltonian_element(h_so, aso, candidate, det; integral_cutoff)
            score = abs(hij * coefficient)
            score > epsilon1 || continue
            scores[candidate] = max(get(scores, candidate, 0.0), score)
        end
    end
    return scores
end

function _sorted_hci_candidates(scores::Dict{CISDDet, Float64})
    candidates = collect(keys(scores))
    sort!(candidates; by = det -> (-scores[det], det))
    return candidates
end

"""
    hci_pt2_correction(h_so, aso, determinants, coefficients, electronic_energy, nspin; kwargs...)

Evaluate the deterministic Epstein-Nesbet PT2 correction over connected
external determinants that pass the heat-bath threshold:

```text
E_PT2 = sum_a (sum_i H_ai c_i)^2 / (E_0 - H_aa).
```

Here `i` runs over the selected variational determinants and `a` runs over
external determinants not currently in the variational space.
"""
function hci_pt2_correction(h_so, aso, determinants, coefficients, electronic_energy, nspin;
                            epsilon2=0.0, coefficient_cutoff=1e-12,
                            integral_cutoff=1e-14)
    selected = Set(determinants)
    couplings = Dict{CISDDet, Float64}()
    max_scores = Dict{CISDDet, Float64}()

    for (col, det) in pairs(determinants)
        coefficient = coefficients[col]
        abs(coefficient) <= coefficient_cutoff && continue
        for candidate in connected_determinants(det, nspin)
            candidate in selected && continue
            hij = hamiltonian_element(h_so, aso, candidate, det; integral_cutoff)
            contribution = hij * coefficient
            score = abs(contribution)
            score > epsilon2 || continue
            couplings[candidate] = get(couplings, candidate, 0.0) + contribution
            max_scores[candidate] = max(get(max_scores, candidate, 0.0), score)
        end
    end

    correction = 0.0
    for (candidate, coupling) in couplings
        diagonal = hamiltonian_element(h_so, aso, candidate, candidate; integral_cutoff)
        denominator = electronic_energy - diagonal
        abs(denominator) <= eps(Float64) && continue
        correction += coupling^2 / denominator
    end

    return (
        correction = correction,
        n_external = length(couplings),
        max_external_score = isempty(max_scores) ? 0.0 : maximum(values(max_scores)),
    )
end

"""
    run_hci(rhf, mp2_result; epsilon1=1e-3, epsilon2=0.0)

Run a deterministic heat-bath selected CI calculation from an RHF reference.
The returned `total_energy` includes the deterministic PT2 correction; the
pure variational result is available as `variational_total_energy`.

Selection adds a candidate determinant `a` when at least one selected
determinant `i` satisfies

```text
|H_ai c_i| > epsilon1.
```
"""
function run_hci(rhf, mp2_result;
                 epsilon1=1e-3, epsilon2=0.0, maxiter=20, nroots=5,
                 coefficient_cutoff=1e-12, integral_cutoff=1e-14,
                 verbose=true)
    nspin = 2 * rhf.nbasis
    n_elec = rhf.n_elec
    reference = determinant_from_orbitals(1:n_elec)

    h_mo = Mat_aotoMat_mo(rhf.mo_coeffs, rhf.h1e)
    h_so = spatial_to_spinorbital_1e(h_mo)
    aso = mo_to_aso(mp2_result.new_eri)

    determinants = CISDDet[reference]
    electronic_energy = Inf
    ci_vector = [1.0]
    H = zeros(Float64, 1, 1)
    converged = false
    iterations = 0
    added_per_iteration = Int[]

    for iteration in 1:maxiter
        iterations = iteration
        H = build_selected_ci_hamiltonian(h_so, aso, determinants; integral_cutoff)
        eig = eigen(Symmetric(H))
        electronic_energy = eig.values[1]
        ci_vector = eig.vectors[:, 1]

        scores = _hci_candidate_scores(h_so, aso, determinants, ci_vector, nspin;
                                       epsilon1, coefficient_cutoff, integral_cutoff)
        candidates = _sorted_hci_candidates(scores)
        push!(added_per_iteration, length(candidates))

        verbose && @printf("HCI iter %3d  dim=%5d  E_elec=%20.12f  add=%5d\n",
                           iteration, length(determinants), electronic_energy, length(candidates))

        if isempty(candidates)
            converged = true
            break
        end

        append!(determinants, candidates)
    end

    if !converged
        H = build_selected_ci_hamiltonian(h_so, aso, determinants; integral_cutoff)
        eig = eigen(Symmetric(H))
        electronic_energy = eig.values[1]
        ci_vector = eig.vectors[:, 1]
        scores = _hci_candidate_scores(h_so, aso, determinants, ci_vector, nspin;
                                       epsilon1, coefficient_cutoff, integral_cutoff)
        converged = isempty(scores)
    end

    roots = min(nroots, size(H, 1))
    eig = eigen(Symmetric(H))
    electronic_energies = eig.values[1:roots]
    ci_vectors = eig.vectors[:, 1:roots]
    electronic_energy = electronic_energies[1]
    ci_vector = ci_vectors[:, 1]

    pt2 = hci_pt2_correction(h_so, aso, determinants, ci_vector, electronic_energy, nspin;
                             epsilon2, coefficient_cutoff, integral_cutoff)

    nuclear_repulsion = rhf.total_energy - rhf.energy
    variational_total_energy = electronic_energy + nuclear_repulsion
    pt2_total_energy = variational_total_energy + pt2.correction
    variational_correlation = variational_total_energy - rhf.total_energy
    total_correlation = pt2_total_energy - rhf.total_energy

    verbose && begin
        @printf("HCI variational dimension = %d\n", length(determinants))
        @printf("HCI variational corr      = %20.12f Eh\n", variational_correlation)
        @printf("HCI PT2 correction        = %20.12f Eh (%d external)\n",
                pt2.correction, pt2.n_external)
        @printf("HCI+PT2 total             = %20.12f Eh\n", pt2_total_energy)
    end

    return (
        E_hci = variational_correlation,
        E_pt2 = pt2.correction,
        E_hci_pt2 = total_correlation,
        total_energy = pt2_total_energy,
        variational_total_energy = variational_total_energy,
        pt2_total_energy = pt2_total_energy,
        electronic_energy = electronic_energy,
        electronic_energies = electronic_energies,
        total_energies = electronic_energies .+ nuclear_repulsion,
        ci_vector = ci_vector,
        ci_vectors = ci_vectors,
        reference_weight = ci_vector[1]^2,
        determinants = determinants,
        hamiltonian = H,
        dimension = length(determinants),
        iterations = iterations,
        converged = converged,
        added_per_iteration = added_per_iteration,
        n_external = pt2.n_external,
        max_external_score = pt2.max_external_score,
        epsilon1 = epsilon1,
        epsilon2 = epsilon2,
        h_so = h_so,
        aso = aso,
    )
end
