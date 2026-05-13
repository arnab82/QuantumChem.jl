"""
Restricted open-shell Hartree-Fock (ROHF).

Project #16 keeps one shared set of spatial orbitals while allowing closed
shells plus high-spin singly occupied open shells.
"""

using LinearAlgebra, Statistics, Printf

"""
    rohf_occupations(nbasis, n_closed, n_open) -> Vector{Float64}

Return spatial-orbital occupations: `2` for closed shells, `1` for open shells,
and `0` for virtual orbitals.
"""
function rohf_occupations(nbasis::Integer, n_closed::Integer, n_open::Integer)
    n_closed >= 0 && n_open >= 0 ||
        throw(ArgumentError("ROHF closed/open occupation counts must be nonnegative"))
    n_closed + n_open <= nbasis ||
        throw(ArgumentError("ROHF occupations exceed the number of basis functions"))

    occ = zeros(Float64, nbasis)
    occ[1:n_closed] .= 2.0
    occ[(n_closed + 1):(n_closed + n_open)] .= 1.0
    return occ
end

"""
    rohf_shell_counts(n_elec, spin) -> (n_closed, n_open, n_alpha, n_beta)

Convert electron count and PySCF spin convention (`spin = n_alpha - n_beta`) to
ROHF closed/open shell counts.  This implementation follows the usual high-spin
ROHF convention, so `spin` must be nonnegative.
"""
function rohf_shell_counts(n_elec::Integer, spin::Integer)
    spin >= 0 || throw(ArgumentError("ROHF expects nonnegative high-spin states"))
    n_alpha, n_beta = uhf_electron_counts(n_elec, spin)
    n_alpha >= n_beta ||
        throw(ArgumentError("ROHF expects n_alpha >= n_beta"))
    return (n_closed = n_beta,
            n_open = n_alpha - n_beta,
            n_alpha = n_alpha,
            n_beta = n_beta)
end

"""
    make_rohf_density(c, n_closed, n_open)

Build closed-shell, open-shell, alpha, and beta AO densities from one shared
spatial MO coefficient matrix:

```text
D_closed = C_closed C_closed'
D_open   = C_open C_open'
D_alpha  = D_closed + D_open
D_beta   = D_closed.
```
"""
function make_rohf_density(c, n_closed::Integer, n_open::Integer)
    n_closed >= 0 && n_open >= 0 ||
        throw(ArgumentError("ROHF closed/open occupation counts must be nonnegative"))

    D_closed = make_density(c, n_closed)
    D_open = n_open == 0 ?
        zeros(Float64, size(c, 1), size(c, 1)) :
        make_density(c[:, (n_closed + 1):(n_closed + n_open)], n_open)
    return (
        closed = D_closed,
        open = D_open,
        alpha = D_closed + D_open,
        beta = D_closed,
    )
end

"""
    roothaan_fock(Falpha, Fbeta, Dalpha, Dbeta, S) -> Matrix{Float64}

Build Roothaan's effective ROHF Fock matrix.  In the closed/open/virtual block
structure it uses:

```text
           closed   open   virtual
closed       Fc      Fb      Fc
open         Fb      Fc      Fa
virtual      Fc      Fa      Fc
```

where `Fc = (Falpha + Fbeta) / 2`.
"""
function roothaan_fock(Falpha, Fbeta, Dalpha, Dbeta, S)
    size(Falpha) == size(Fbeta) == size(Dalpha) == size(Dbeta) == size(S) ||
        throw(DimensionMismatch("ROHF Fock, density, and overlap matrices must have the same shape"))

    nbasis = size(S, 1)
    Fc = (Falpha .+ Fbeta) ./ 2
    Pc = Dbeta * S
    Po = (Dalpha .- Dbeta) * S
    Pv = Matrix{Float64}(I, nbasis, nbasis) - Dalpha * S

    F = 0.5 .* (Pc' * Fc * Pc)
    F .+= 0.5 .* (Po' * Fc * Po)
    F .+= 0.5 .* (Pv' * Fc * Pv)
    F .+= Po' * Fbeta * Pc
    F .+= Po' * Falpha * Pv
    F .+= Pv' * Fc * Pc
    F .+= F'
    return (F .+ F') ./ 2
end

function _combined_rohf_rms(Dalpha, Dbeta, Dalpha_old, Dbeta_old)
    return sqrt(0.5 * (mean((Dalpha .- Dalpha_old).^2) +
                       mean((Dbeta .- Dbeta_old).^2)))
end

"""
    scf_rohf(mol, h1e, eri, S, n_closed, n_open; kwargs...) -> NamedTuple

Run a restricted open-shell Hartree-Fock SCF loop.  The alpha/beta Fock
matrices are first built from the ROHF spin densities, then Roothaan's
effective Fock matrix is diagonalized:

```text
F_ROHF C = S C epsilon.
```
"""
function scf_rohf(mol, h1e, eri, S, n_closed::Integer, n_open::Integer;
                  maxiter=100, e_tol=1e-12, d_tol=0.0,
                  diis=false, diis_size=6, diis_start=2,
                  verbose=true)
    nuclear_repulsion = pyscf_nucr(mol)
    nbasis = size(h1e, 1)
    n_closed + n_open <= nbasis ||
        throw(ArgumentError("ROHF occupations exceed the number of basis functions"))

    X = make_s_half(S)
    orbital_energies, c = _diagonalize_ao_fock(h1e, X)
    D = make_rohf_density(c, n_closed, n_open)

    Falpha = zeros(Float64, nbasis, nbasis)
    Fbeta = zeros(Float64, nbasis, nbasis)
    F_roothaan = zeros(Float64, nbasis, nbasis)
    E_prev = 0.0
    E_elec = 0.0
    iterations = maxiter
    rms_D = Inf

    fock_history = Matrix{Float64}[]
    error_history = Matrix{Float64}[]

    for n in 1:maxiter
        Dalpha_old = D.alpha
        Dbeta_old = D.beta

        Falpha_built, Fbeta_built = make_uhf_fock(Dalpha_old, Dbeta_old, h1e, eri)
        F_roothaan_built = roothaan_fock(Falpha_built, Fbeta_built,
                                         Dalpha_old, Dbeta_old, S)
        Falpha = Falpha_built
        Fbeta = Fbeta_built
        F_roothaan = F_roothaan_built

        if diis
            push!(fock_history, copy(F_roothaan_built))
            push!(error_history,
                  diis_error_matrix(F_roothaan_built, Dalpha_old + Dbeta_old, S))

            if length(fock_history) > diis_size
                popfirst!(fock_history)
                popfirst!(error_history)
            end

            if length(fock_history) >= diis_start
                coefficients = diis_coefficients(error_history)
                F_roothaan = extrapolate_fock(fock_history, coefficients)
            end
        end

        orbital_energies, c = _diagonalize_ao_fock(F_roothaan, X)
        D = make_rohf_density(c, n_closed, n_open)

        E_elec = uhf_energy(D.alpha, D.beta, Falpha_built, Fbeta_built, h1e)
        delta = abs(E_elec - E_prev)
        rms_D = _combined_rohf_rms(D.alpha, D.beta, Dalpha_old, Dbeta_old)

        if verbose
            @printf("  iter= %3d  E= %20.12f  ΔE= %12.4e  RMS(D)= %12.4e\n",
                    n, E_elec + nuclear_repulsion, delta, rms_D)
        end
        E_prev = E_elec

        if delta < e_tol && (d_tol <= 0 || rms_D < d_tol)
            iterations = n
            Falpha, Fbeta = make_uhf_fock(D.alpha, D.beta, h1e, eri)
            F_roothaan = roothaan_fock(Falpha, Fbeta, D.alpha, D.beta, S)
            orbital_energies, c = _diagonalize_ao_fock(F_roothaan, X)
            D = make_rohf_density(c, n_closed, n_open)
            Falpha, Fbeta = make_uhf_fock(D.alpha, D.beta, h1e, eri)
            F_roothaan = roothaan_fock(Falpha, Fbeta, D.alpha, D.beta, S)
            E_elec = uhf_energy(D.alpha, D.beta, Falpha, Fbeta, h1e)
            break
        end
    end

    n_alpha = n_closed + n_open
    n_beta = n_closed
    s2 = 0.5 * n_open * (0.5 * n_open + 1.0)
    return (
        energy = E_elec,
        total_energy = E_elec + nuclear_repulsion,
        orbital_energies = orbital_energies,
        mo_coeffs = c,
        occupations = rohf_occupations(nbasis, n_closed, n_open),
        fock = F_roothaan,
        fock_alpha = Falpha,
        fock_beta = Fbeta,
        density_closed = D.closed,
        density_open = D.open,
        density_alpha = D.alpha,
        density_beta = D.beta,
        nbasis = nbasis,
        n_closed = Int(n_closed),
        n_open = Int(n_open),
        n_alpha = Int(n_alpha),
        n_beta = Int(n_beta),
        n_elec = Int(n_alpha + n_beta),
        spin_square = s2,
        spin_multiplicity = 2.0 * (0.5 * n_open) + 1.0,
        iterations = iterations,
        rms_density = rms_D,
    )
end

"""
    run_rohf(; atoms, basis, charge, spin, unit, n_elec, verbose)

Build the molecule and run restricted open-shell Hartree-Fock.  Closed-shell
systems reduce to RHF; open-shell systems use high-spin alpha open shells.
"""
function run_rohf(;
    atoms  = "O 0.000000000000 -0.143225816552 0.000000000000;" *
             "H 1.638036840407  1.136548822547 -0.000000000000;" *
             "H -1.638036840407 1.136548822547 -0.000000000000",
    basis  = "sto-3g",
    charge = 0,
    spin   = 0,
    unit   = "Bohr",
    n_elec = nothing,
    maxiter = 100,
    e_tol = 1e-12,
    d_tol = 0.0,
    diis = false,
    diis_size = 6,
    diis_start = 2,
    verbose = true,
)
    spin_for_mol = Int(spin)
    mol = make_molecule(atoms, basis, charge, spin_for_mol, unit)
    mol_n_elec = pyscf_getattr(mol, "nelectron", Int)
    electron_count = isnothing(n_elec) ? mol_n_elec : Int(n_elec)
    electron_count == mol_n_elec ||
        throw(ArgumentError("n_elec=$electron_count does not match PySCF molecule electron count $mol_n_elec"))

    counts = rohf_shell_counts(electron_count, spin_for_mol)
    h1e = pyscf_1e(mol)
    S = pyscf_overlap(mol)
    eri = pyscf_2e(mol)

    result = scf_rohf(mol, h1e, eri, S, counts.n_closed, counts.n_open;
                      maxiter, e_tol, d_tol, diis, diis_size, diis_start, verbose)
    verbose && @printf("\nROHF total energy = %20.12f Eh   <S^2> = %.8f\n",
                       result.total_energy, result.spin_square)

    return merge(result, (
        h1e = h1e,
        eri = eri,
        S = S,
        charge = Int(charge),
        spin = spin_for_mol,
    ))
end
