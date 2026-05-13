"""
Unrestricted coupled-cluster singles and doubles (UCCSD).

Project #17 reuses the spin-orbital CCSD equations with orbitals from a UHF
reference determinant.
"""

using LinearAlgebra
using Base.Threads: @threads

"""
    uhf_spinorbital_order(uhf) -> Vector

Return the UHF spin-orbital order used by `run_uccsd`: occupied alpha, occupied
beta, virtual alpha, virtual beta.  This puts the occupied block first for the
existing spin-orbital CCSD solver.
"""
function uhf_spinorbital_order(uhf)
    nbasis = uhf.nbasis
    n_alpha = uhf.n_alpha
    n_beta = uhf.n_beta

    order = NamedTuple{(:spin, :orbital, :occupied), Tuple{Symbol, Int, Bool}}[]
    for p in 1:n_alpha
        push!(order, (spin=:alpha, orbital=p, occupied=true))
    end
    for p in 1:n_beta
        push!(order, (spin=:beta, orbital=p, occupied=true))
    end
    for p in (n_alpha + 1):nbasis
        push!(order, (spin=:alpha, orbital=p, occupied=false))
    end
    for p in (n_beta + 1):nbasis
        push!(order, (spin=:beta, orbital=p, occupied=false))
    end
    return order
end

"""
    uhf_spinorbital_coefficients(uhf)

Build the AO coefficient matrix, spin labels, orbital energies, and order
metadata for UHF spin orbitals.
"""
function uhf_spinorbital_coefficients(uhf)
    order = uhf_spinorbital_order(uhf)
    nbasis = uhf.nbasis
    nso = 2 * nbasis

    coefficients = zeros(Float64, nbasis, nso)
    spins = Vector{Symbol}(undef, nso)
    orbital_energies = zeros(Float64, nso)

    for (idx, item) in pairs(order)
        spins[idx] = item.spin
        if item.spin == :alpha
            coefficients[:, idx] .= uhf.mo_coeffs_alpha[:, item.orbital]
            orbital_energies[idx] = uhf.orbital_energies_alpha[item.orbital]
        else
            coefficients[:, idx] .= uhf.mo_coeffs_beta[:, item.orbital]
            orbital_energies[idx] = uhf.orbital_energies_beta[item.orbital]
        end
    end

    return (
        coefficients = coefficients,
        spins = spins,
        orbital_energies = orbital_energies,
        order = order,
    )
end

"""
    antisymmetrize_spinorbital_eri(mo_eri, spins) -> Array{Float64,4}

Convert unrestricted MO integrals to antisymmetrized spin-orbital integrals:

```text
<pq||rs> = (p r | q s) delta(spin_p, spin_r) delta(spin_q, spin_s)
         - (p s | q r) delta(spin_p, spin_s) delta(spin_q, spin_r)
```
"""
function antisymmetrize_spinorbital_eri(mo_eri, spins)
    nso = length(spins)
    size(mo_eri) == (nso, nso, nso, nso) ||
        throw(DimensionMismatch("MO ERI tensor size must match spin labels"))

    aso = zeros(Float64, nso, nso, nso, nso)
    @threads for idx in eachindex(aso)
        x = idx - 1
        p = (x % nso) + 1; x ÷= nso
        q = (x % nso) + 1; x ÷= nso
        r = (x % nso) + 1; x ÷= nso
        s = (x % nso) + 1

        value = 0.0
        @inbounds begin
            if spins[p] == spins[r] && spins[q] == spins[s]
                value += mo_eri[p,r,q,s]
            end
            if spins[p] == spins[s] && spins[q] == spins[r]
                value -= mo_eri[p,s,q,r]
            end
            aso[p,q,r,s] = value
        end
    end
    return aso
end

"""
    build_uccsd_spinorbital_inputs(uhf)

Build the ASO integral tensor, diagonal spin-orbital Fock matrix, and
spin-orbital energies for UCCSD.
"""
function build_uccsd_spinorbital_inputs(uhf)
    basis = uhf_spinorbital_coefficients(uhf)
    mo_eri = transform_eri(basis.coefficients, uhf.eri)
    aso = antisymmetrize_spinorbital_eri(mo_eri, basis.spins)
    F_so = Matrix(Diagonal(basis.orbital_energies))
    return merge(basis, (mo_eri = mo_eri, aso = aso, fock = F_so))
end

"""
    run_uccsd(uhf; maxiter, e_tol, amp_tol, diis, diis_size, diis_start, verbose)

Run unrestricted CCSD from a UHF reference.  The input should be the NamedTuple
returned by `run_uhf`.
"""
function run_uccsd(uhf;
                   maxiter=500, e_tol=1e-10, amp_tol=1e-10,
                   diis=true, diis_size=8, diis_start=3, verbose=true)
    o = uhf.n_elec
    v = 2 * uhf.nbasis - o
    inputs = build_uccsd_spinorbital_inputs(uhf)

    result = ccsd_scf(inputs.aso, inputs.fock, inputs.orbital_energies, o, v, o;
                      maxiter, e_tol, amp_tol, diis, diis_size, diis_start, verbose)

    total = result.E_ccsd + uhf.total_energy
    verbose && @printf("\nUCCSD correlation energy = %20.12f Eh\nUCCSD total energy       = %20.12f Eh\n",
                       result.E_ccsd, total)

    return merge(result, (
        total_energy = total,
        reference_energy = uhf.total_energy,
        orbital_order = inputs.order,
        orbital_energies = inputs.orbital_energies,
        spins = inputs.spins,
        aso = inputs.aso,
        fock = inputs.fock,
    ))
end

