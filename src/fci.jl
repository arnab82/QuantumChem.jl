"""
Full configuration interaction (FCI).

Project #22 diagonalizes the electronic Hamiltonian in the complete fixed
electron-number determinant space for the chosen spin-orbital basis.  This is
the exact answer within the one-particle basis used by RHF.
"""

using LinearAlgebra, Printf

function _determinant_combinations(values::AbstractVector{<:Integer}, k::Integer)
    k == 0 && return [Tuple{}()]
    k > length(values) && return Tuple[]

    combinations = Tuple[]
    current = Int[]
    function visit(start, remaining)
        if remaining == 0
            push!(combinations, Tuple(current))
            return nothing
        end
        last_start = length(values) - remaining + 1
        for idx in start:last_start
            push!(current, Int(values[idx]))
            visit(idx + 1, remaining - 1)
            pop!(current)
        end
        return nothing
    end
    visit(1, k)
    return combinations
end

"""
    fci_dimension(nspin, n_elec)

Return the number of fixed-electron Slater determinants, `binomial(nspin,
n_elec)`:

```text
dim_FCI = C(nspin, n_elec) = nspin! / (n_elec! (nspin - n_elec)!).
```
"""
function fci_dimension(nspin::Integer, n_elec::Integer)
    0 <= n_elec <= nspin ||
        throw(ArgumentError("n_elec must be between 0 and nspin"))
    return binomial(Int(nspin), Int(n_elec))
end

"""
    fci_determinants(nspin, n_elec) -> (determinants, labels)

Generate every determinant with `n_elec` occupied spin orbitals among `nspin`
spin orbitals.  Labels record the excitation rank relative to the canonical
Hartree-Fock reference determinant.
"""
function fci_determinants(nspin::Integer, n_elec::Integer)
    0 <= n_elec <= nspin ||
        throw(ArgumentError("n_elec must be between 0 and nspin"))
    nspin <= 63 ||
        throw(ArgumentError("FCI determinant bit representation supports orbitals 1:63"))

    reference = determinant_from_orbitals(1:n_elec)
    determinants = CISDDet[]
    labels = NamedTuple{(:rank, :holes, :particles), Tuple{Int, Tuple, Tuple}}[]

    for orbitals in _determinant_combinations(collect(1:nspin), n_elec)
        det = determinant_from_orbitals(orbitals)
        excitation = _excitation_between(det, reference, nspin)
        push!(determinants, det)
        push!(labels, (
            rank = excitation.rank,
            holes = Tuple(excitation.holes),
            particles = Tuple(excitation.particles),
        ))
    end

    return (determinants=determinants, labels=labels)
end

"""
    run_fci(rhf, mp2_result; nroots=5, verbose=true)

Run full CI in the RHF canonical spin-orbital basis.  The complete
fixed-electron determinant Hamiltonian is built and diagonalized:

```text
H C_k = E_k C_k
E_total,k = E_k + E_nuc
E_corr,k = E_total,k - E_RHF.
```
"""
function run_fci(rhf, mp2_result; nroots=5, integral_cutoff=1e-14, verbose=true)
    nspin = 2 * rhf.nbasis
    n_elec = rhf.n_elec
    dets = fci_determinants(nspin, n_elec)

    h_mo = Mat_aotoMat_mo(rhf.mo_coeffs, rhf.h1e)
    h_so = spatial_to_spinorbital_1e(h_mo)
    aso = mo_to_aso(mp2_result.new_eri)
    H = build_selected_ci_hamiltonian(h_so, aso, dets.determinants; integral_cutoff)

    eig = eigen(Symmetric(H))
    roots = min(nroots, length(eig.values))
    nuclear_repulsion = rhf.total_energy - rhf.energy
    total_energies = eig.values[1:roots] .+ nuclear_repulsion
    correlation_energies = total_energies .- rhf.total_energy

    rank_counts = Dict{Int, Int}()
    for label in dets.labels
        rank_counts[label.rank] = get(rank_counts, label.rank, 0) + 1
    end

    verbose && begin
        @printf("FCI dimension              = %d\n", length(dets.determinants))
        @printf("FCI max excitation rank    = %d\n", maximum(keys(rank_counts)))
        @printf("FCI correlation energy     = %20.12f Eh\n", correlation_energies[1])
        @printf("FCI total energy           = %20.12f Eh\n", total_energies[1])
    end

    return (
        E_fci = correlation_energies[1],
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
        rank_counts = rank_counts,
        dimension = length(dets.determinants),
        h_so = h_so,
        aso = aso,
    )
end
