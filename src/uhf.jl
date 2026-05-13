"""
Unrestricted Hartree-Fock (UHF) using PySCF for one- and two-electron integrals.

Project #15 extends the SCF machinery to separate alpha and beta spin spaces.
"""

using LinearAlgebra, Statistics, Printf
using Base.Threads: @threads, maxthreadid, threadid

"""
    uhf_electron_counts(n_elec, spin) -> (n_alpha, n_beta)

Convert the PySCF spin convention, `spin = n_alpha - n_beta`, into alpha and
beta electron counts.
"""
function uhf_electron_counts(n_elec::Integer, spin::Integer)
    n_elec >= 0 || throw(ArgumentError("n_elec must be nonnegative"))
    abs(spin) <= n_elec || throw(ArgumentError("|spin| cannot exceed n_elec"))
    iseven(n_elec + spin) && iseven(n_elec - spin) ||
        throw(ArgumentError("n_elec and spin have inconsistent parity"))

    n_alpha = (n_elec + spin) ÷ 2
    n_beta = (n_elec - spin) ÷ 2
    n_alpha >= 0 && n_beta >= 0 ||
        throw(ArgumentError("Invalid alpha/beta electron counts"))
    return (Int(n_alpha), Int(n_beta))
end

function _resolve_uhf_counts(n_elec::Integer, spin::Integer, n_alpha, n_beta)
    if n_alpha !== nothing || n_beta !== nothing
        n_alpha !== nothing && n_beta !== nothing ||
            throw(ArgumentError("Provide both n_alpha and n_beta, or provide neither"))

        na = Int(n_alpha)
        nb = Int(n_beta)
        na >= 0 && nb >= 0 ||
            throw(ArgumentError("n_alpha and n_beta must be nonnegative"))
        na + nb == n_elec ||
            throw(ArgumentError("n_alpha + n_beta must equal n_elec"))
        na - nb == spin ||
            throw(ArgumentError("n_alpha - n_beta must equal spin"))
        return (na, nb)
    end

    return uhf_electron_counts(n_elec, spin)
end

function _uhf_molecule_spin(spin::Integer, n_alpha, n_beta)
    if n_alpha !== nothing && n_beta !== nothing
        derived_spin = Int(n_alpha) - Int(n_beta)
        spin != 0 && spin != derived_spin &&
            throw(ArgumentError("spin is inconsistent with n_alpha - n_beta"))
        return derived_spin
    end
    return Int(spin)
end

"""
    make_uhf_density(c_alpha, c_beta, n_alpha, n_beta)

Build the separate AO alpha and beta density matrices:

```text
D^alpha = C^alpha_occ (C^alpha_occ)'
D^beta  = C^beta_occ  (C^beta_occ)'.
```
"""
function make_uhf_density(c_alpha, c_beta, n_alpha::Integer, n_beta::Integer)
    return (
        alpha = make_density(c_alpha, n_alpha),
        beta = make_density(c_beta, n_beta),
    )
end

"""
    make_uhf_fock(Dalpha, Dbeta, h1e, eri) -> (Falpha, Fbeta)

Build UHF AO Fock matrices:

```text
F_alpha[mu,nu] = h[mu,nu] + sum[lambda,sigma] (
    (D_alpha + D_beta)[lambda,sigma] (mu nu | lambda sigma)
  - D_alpha[lambda,sigma] (mu lambda | nu sigma)
)
```

and similarly for beta exchange.
"""
function make_uhf_fock(Dalpha, Dbeta, h1e, eri)
    size(Dalpha) == size(Dbeta) == size(h1e) ||
        throw(DimensionMismatch("UHF densities and h1e must have the same shape"))

    nbasis = size(h1e, 1)
    size(eri) == (nbasis, nbasis, nbasis, nbasis) ||
        throw(DimensionMismatch("ERI tensor size does not match h1e"))

    Falpha = zeros(Float64, nbasis, nbasis)
    Fbeta = zeros(Float64, nbasis, nbasis)
    @threads for idx in 1:(nbasis * nbasis)
        mu = ((idx - 1) % nbasis) + 1
        nu = ((idx - 1) ÷ nbasis) + 1
        value_alpha = 0.0
        value_beta = 0.0
        @inbounds for lambda in 1:nbasis, sigma in 1:nbasis
            density_total = Dalpha[lambda,sigma] + Dbeta[lambda,sigma]
            coulomb = density_total * eri[mu,nu,lambda,sigma]
            value_alpha += coulomb - Dalpha[lambda,sigma] * eri[mu,lambda,nu,sigma]
            value_beta += coulomb - Dbeta[lambda,sigma] * eri[mu,lambda,nu,sigma]
        end
        @inbounds begin
            Falpha[mu,nu] = h1e[mu,nu] + value_alpha
            Fbeta[mu,nu] = h1e[mu,nu] + value_beta
        end
    end
    return (Falpha, Fbeta)
end

"""
    uhf_energy(Dalpha, Dbeta, Falpha, Fbeta, h1e) -> Float64

Compute the UHF electronic energy from alpha/beta densities and Fock matrices.

```text
E_UHF = 1/2 Tr[D^alpha (h + F^alpha)]
      + 1/2 Tr[D^beta  (h + F^beta)].
```
"""
function uhf_energy(Dalpha, Dbeta, Falpha, Fbeta, h1e)
    partial = zeros(Float64, maxthreadid())
    @threads for idx in eachindex(h1e)
        @inbounds partial[threadid()] += 0.5 * (
            Dalpha[idx] * (h1e[idx] + Falpha[idx]) +
            Dbeta[idx] * (h1e[idx] + Fbeta[idx])
        )
    end
    return sum(partial)
end

function _diagonalize_ao_fock(fock, X)
    F_orth = X * fock * X'
    F_orth = (F_orth .+ F_orth') ./ 2
    orbital_energies, c_orth = eigen(Symmetric(F_orth))
    return orbital_energies, X * c_orth
end

function _combined_uhf_rms(Dalpha, Dbeta, Dalpha_old, Dbeta_old)
    return sqrt(0.5 * (mean((Dalpha .- Dalpha_old).^2) +
                       mean((Dbeta .- Dbeta_old).^2)))
end

"""
    spin_square(c_alpha, c_beta, S, n_alpha, n_beta) -> Float64

Return the UHF expectation value `<S^2>`:

```text
<S^2> = S_z(S_z + 1) + n_beta
      - sum_ij | <phi_i^alpha | phi_j^beta> |^2,
S_z = (n_alpha - n_beta) / 2.
```
"""
function spin_square(c_alpha, c_beta, S, n_alpha::Integer, n_beta::Integer)
    c_alpha_occ = @view c_alpha[:, 1:n_alpha]
    c_beta_occ = @view c_beta[:, 1:n_beta]
    spin_z = 0.5 * (n_alpha - n_beta)
    overlap = c_alpha_occ' * S * c_beta_occ
    return spin_z * (spin_z + 1.0) + n_beta - sum(abs2, overlap)
end

"""
    scf_uhf(mol, h1e, eri, S, n_alpha, n_beta; kwargs...) -> NamedTuple

Run the unrestricted Hartree-Fock SCF loop.  Each iteration builds separate
alpha and beta Fock matrices, solves

```text
F^sigma C^sigma = S C^sigma epsilon^sigma,
```

updates the spin densities, and checks energy/density convergence.
"""
function scf_uhf(mol, h1e, eri, S, n_alpha::Integer, n_beta::Integer;
                 maxiter=100, e_tol=1e-12, d_tol=0.0,
                 diis=false, diis_size=6, diis_start=2,
                 verbose=true)
    nuclear_repulsion = pyscf_nucr(mol)
    nbasis = size(h1e, 1)
    n_alpha <= nbasis && n_beta <= nbasis ||
        throw(ArgumentError("Electron counts exceed the number of basis functions"))

    X = make_s_half(S)
    orbital_energies_alpha, c_alpha = _diagonalize_ao_fock(h1e, X)
    orbital_energies_beta, c_beta = _diagonalize_ao_fock(h1e, X)

    D = make_uhf_density(c_alpha, c_beta, n_alpha, n_beta)
    Dalpha = D.alpha
    Dbeta = D.beta

    Falpha = zeros(Float64, nbasis, nbasis)
    Fbeta = zeros(Float64, nbasis, nbasis)
    E_prev = 0.0
    E_elec = 0.0
    iterations = maxiter
    rms_D = Inf

    fock_alpha_history = Matrix{Float64}[]
    fock_beta_history = Matrix{Float64}[]
    error_history = Matrix{Float64}[]

    for n in 1:maxiter
        Dalpha_old = Dalpha
        Dbeta_old = Dbeta
        Falpha_built, Fbeta_built = make_uhf_fock(Dalpha_old, Dbeta_old, h1e, eri)
        Falpha = Falpha_built
        Fbeta = Fbeta_built

        if diis
            push!(fock_alpha_history, copy(Falpha_built))
            push!(fock_beta_history, copy(Fbeta_built))
            push!(error_history, vcat(
                diis_error_matrix(Falpha_built, Dalpha_old, S),
                diis_error_matrix(Fbeta_built, Dbeta_old, S),
            ))

            if length(fock_alpha_history) > diis_size
                popfirst!(fock_alpha_history)
                popfirst!(fock_beta_history)
                popfirst!(error_history)
            end

            if length(fock_alpha_history) >= diis_start
                coefficients = diis_coefficients(error_history)
                Falpha = extrapolate_fock(fock_alpha_history, coefficients)
                Fbeta = extrapolate_fock(fock_beta_history, coefficients)
            end
        end

        orbital_energies_alpha, c_alpha = _diagonalize_ao_fock(Falpha, X)
        orbital_energies_beta, c_beta = _diagonalize_ao_fock(Fbeta, X)
        D = make_uhf_density(c_alpha, c_beta, n_alpha, n_beta)
        Dalpha = D.alpha
        Dbeta = D.beta

        E_elec = uhf_energy(Dalpha, Dbeta, Falpha_built, Fbeta_built, h1e)
        delta = abs(E_elec - E_prev)
        rms_D = _combined_uhf_rms(Dalpha, Dbeta, Dalpha_old, Dbeta_old)

        if verbose
            @printf("  iter= %3d  E= %20.12f  ΔE= %12.4e  RMS(D)= %12.4e\n",
                    n, E_elec + nuclear_repulsion, delta, rms_D)
        end
        E_prev = E_elec

        if delta < e_tol && (d_tol <= 0 || rms_D < d_tol)
            iterations = n
            Falpha, Fbeta = make_uhf_fock(Dalpha, Dbeta, h1e, eri)
            orbital_energies_alpha, c_alpha = _diagonalize_ao_fock(Falpha, X)
            orbital_energies_beta, c_beta = _diagonalize_ao_fock(Fbeta, X)
            D = make_uhf_density(c_alpha, c_beta, n_alpha, n_beta)
            Dalpha = D.alpha
            Dbeta = D.beta
            Falpha, Fbeta = make_uhf_fock(Dalpha, Dbeta, h1e, eri)
            E_elec = uhf_energy(Dalpha, Dbeta, Falpha, Fbeta, h1e)
            break
        end
    end

    s2 = spin_square(c_alpha, c_beta, S, n_alpha, n_beta)
    return (
        energy = E_elec,
        total_energy = E_elec + nuclear_repulsion,
        orbital_energies_alpha = orbital_energies_alpha,
        orbital_energies_beta = orbital_energies_beta,
        mo_coeffs_alpha = c_alpha,
        mo_coeffs_beta = c_beta,
        fock_alpha = Falpha,
        fock_beta = Fbeta,
        density_alpha = Dalpha,
        density_beta = Dbeta,
        nbasis = nbasis,
        n_alpha = Int(n_alpha),
        n_beta = Int(n_beta),
        n_elec = Int(n_alpha + n_beta),
        spin_square = s2,
        spin_multiplicity = sqrt(4.0 * s2 + 1.0),
        iterations = iterations,
        rms_density = rms_D,
    )
end

"""
    run_uhf(; atoms, basis, charge, spin, unit, n_elec, n_alpha, n_beta, verbose)

Build the molecule and run unrestricted Hartree-Fock.  By default this uses the
same closed-shell water system as `run_rhf`; set `spin` or explicit alpha/beta
counts for open-shell systems.
"""
function run_uhf(;
    atoms  = "O 0.000000000000 -0.143225816552 0.000000000000;" *
             "H 1.638036840407  1.136548822547 -0.000000000000;" *
             "H -1.638036840407 1.136548822547 -0.000000000000",
    basis  = "sto-3g",
    charge = 0,
    spin   = 0,
    unit   = "Bohr",
    n_elec = nothing,
    n_alpha = nothing,
    n_beta = nothing,
    maxiter = 100,
    e_tol = 1e-12,
    d_tol = 0.0,
    diis = false,
    diis_size = 6,
    diis_start = 2,
    verbose = true,
)
    spin_for_mol = _uhf_molecule_spin(Int(spin), n_alpha, n_beta)
    mol = make_molecule(atoms, basis, charge, spin_for_mol, unit)
    mol_n_elec = pyscf_getattr(mol, "nelectron", Int)
    electron_count = isnothing(n_elec) ? mol_n_elec : Int(n_elec)
    electron_count == mol_n_elec ||
        throw(ArgumentError("n_elec=$electron_count does not match PySCF molecule electron count $mol_n_elec"))

    nalpha, nbeta = _resolve_uhf_counts(electron_count, spin_for_mol, n_alpha, n_beta)
    h1e = pyscf_1e(mol)
    S = pyscf_overlap(mol)
    eri = pyscf_2e(mol)

    result = scf_uhf(mol, h1e, eri, S, nalpha, nbeta;
                     maxiter, e_tol, d_tol, diis, diis_size, diis_start, verbose)
    verbose && @printf("\nUHF total energy = %20.12f Eh   <S^2> = %.8f\n",
                       result.total_energy, result.spin_square)

    return merge(result, (
        h1e = h1e,
        eri = eri,
        S = S,
        charge = Int(charge),
        spin = spin_for_mol,
    ))
end
