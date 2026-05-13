"""
Restricted Hartree-Fock (RHF) using PySCF for one- and two-electron integrals.

Exported entry point: `run_rhf(; atoms, basis, charge, spin, unit, n_elec)`
"""

using PyCall
using LinearAlgebra, Statistics, Printf
using Base.Threads: @threads, maxthreadid, threadid

# ── Molecule and integral helpers ──────────────────────────────────────────────

# PyCall + newer Python runtimes can finalize PySCF objects without the GIL
# during Julia package tests.  The numerical arrays are copied into Julia, but
# retaining the tiny PySCF molecule wrappers avoids mid-run finalizer crashes.
const _PYSCF_KEEPALIVE = Any[]

function _keep_pyscf!(objects...)
    for object in objects
        (object isa PyObject || object isa PyArray) && push!(_PYSCF_KEEPALIVE, object)
    end
    return nothing
end

function pyscf_getattr(mol, name::AbstractString, ::Type{T}) where T
    getattr = pybuiltin("getattr")
    value = pycall(getattr, PyObject, mol, name)
    _keep_pyscf!(getattr, value)
    return convert(T, value)
end

"""
    make_molecule(atoms, basis, charge, spin, unit) -> PyObject

Build and return a PySCF Mole object.
"""
function make_molecule(atoms::String, basis::String, charge::Int, spin::Int, unit::String; symmetry=false)
    pyscf = pyimport("pyscf")
    gto = pyscf.gto
    molecule_constructor = gto.M
    _keep_pyscf!(pyscf, gto, molecule_constructor)
    mol = pycall(molecule_constructor, PyObject)
    mol.charge = charge
    mol.unit   = unit
    mol.spin   = spin
    mol.symmetry = pyscf_symmetry_setting(symmetry)
    build = mol.build
    built = pycall(build, PyObject; atom=atoms, basis=basis)
    _keep_pyscf!(mol, build, built)
    return mol
end

function pyscf_intor(mol, name::AbstractString, ::Val{N}) where N
    intor = mol.intor
    values = pycall(intor, PyObject, name)
    pyarray = PyArray(values)
    _keep_pyscf!(intor, values, pyarray)
    return Array{Float64,N}(Array(pyarray))
end

"""Return the core Hamiltonian (T + V_ne) in the AO basis."""
pyscf_1e(mol) = pyscf_intor(mol, "int1e_kin", Val(2)) .+
                pyscf_intor(mol, "int1e_nuc", Val(2))

"""Return the AO overlap matrix."""
pyscf_overlap(mol) = pyscf_intor(mol, "int1e_ovlp", Val(2))

"""Return the 4-index AO electron-repulsion integrals."""
pyscf_2e(mol) = pyscf_intor(mol, "int2e", Val(4))

"""Return the nuclear-repulsion energy."""
function pyscf_nucr(mol)
    energy_nuc = mol.energy_nuc
    _keep_pyscf!(energy_nuc)
    return pycall(energy_nuc, Float64)
end

# ── SCF utilities ──────────────────────────────────────────────────────────────

"""
    scf_energy(D, fock, h1e) -> Float64

Compute the one-electron + two-electron HF energy:
    E = ∑_{μν} D_{μν} (F_{μν} + h_{μν})
"""
function scf_energy(D, fock, h1e)
    partial = zeros(Float64, maxthreadid())
    @threads for idx in eachindex(D)
        @inbounds partial[threadid()] += D[idx] * (fock[idx] + h1e[idx])
    end
    return sum(partial)
end

"""
    make_fock(D, h1e, eri) -> Matrix{Float64}

Build the AO Fock matrix:
    F_{μν} = h_{μν} + ∑_{λσ} D_{λσ} (2 (μν|λσ) − (μλ|νσ))
"""
function make_fock(D, h1e, eri)
    nbasis = size(D, 1)
    F = zeros(Float64, nbasis, nbasis)
    @threads for idx in 1:(nbasis * nbasis)
        i = ((idx - 1) % nbasis) + 1
        j = ((idx - 1) ÷ nbasis) + 1
        value = 0.0
        @inbounds for k in 1:nbasis, l in 1:nbasis
            value += D[k,l] * (2*eri[i,j,k,l] - eri[i,k,j,l])
        end
        @inbounds F[i,j] = h1e[i,j] + value
    end
    return F
end

"""
    make_density(c, nocc) -> Matrix{Float64}

Build the AO density matrix from the first `nocc` MO columns of `c`:
    D_{μν} = ∑_{i=1}^{nocc} C_{μi} C_{νi}
"""
function make_density(c, nocc)
    c_occ = @view c[:, 1:nocc]
    return c_occ * c_occ'
end

"""
    diis_error_matrix(fock, D, S) -> Matrix{Float64}

Pulay DIIS error matrix in the AO basis:
    e = F D S - S D F
"""
diis_error_matrix(fock, D, S) = fock * D * S - S * D * fock

"""
    diis_coefficients(errors) -> Vector{Float64}

Solve the Pulay DIIS constrained least-squares equations for the coefficients
that minimize the linear combination of error matrices.
"""
function diis_coefficients(errors)
    n = length(errors)
    n >= 2 || throw(ArgumentError("DIIS needs at least two error matrices"))

    B = zeros(Float64, n + 1, n + 1)
    for i in 1:n, j in 1:i
        value = dot(errors[i], errors[j])
        B[i,j] = value
        B[j,i] = value
    end
    B[1:n,n+1] .= -1.0
    B[n+1,1:n] .= -1.0

    rhs = zeros(Float64, n + 1)
    rhs[n+1] = -1.0
    return (B \ rhs)[1:n]
end

"""
    extrapolate_fock(focks, coefficients) -> Matrix{Float64}

Form the Pulay DIIS extrapolated Fock matrix:

```text
F_DIIS = sum_i c_i F_i,   sum_i c_i = 1.
```
"""
function extrapolate_fock(focks, coefficients)
    length(focks) == length(coefficients) ||
        throw(ArgumentError("Fock and DIIS coefficient counts differ"))

    extrapolated = zeros(Float64, size(focks[end]))
    for (coefficient, fock) in zip(coefficients, focks)
        extrapolated .+= coefficient .* fock
    end
    return extrapolated
end

"""
    make_s_half(S) -> Matrix{Float64}

Compute S^{-1/2} via symmetric orthogonalization.
"""
function make_s_half(S)
    s = (S .+ S') ./ 2
    q, L = eigen(Symmetric(s))
    q_half = Diagonal([qi^(-0.5) for qi in q])
    return L * q_half * L'
end

# ── SCF driver ─────────────────────────────────────────────────────────────────

"""
    scf(mol, h1e, eri, S, n_elec; maxiter, e_tol, verbose) -> NamedTuple

Run the closed-shell RHF SCF loop.

Returns `(energy, total_energy, orbital_energies, mo_coeffs, fock, nbasis)`:
- `energy`          : electronic energy (without nuclear repulsion)
- `total_energy`    : E_elec + E_nuc
- `orbital_energies`: MO eigenvalues ε
- `mo_coeffs`       : MO coefficient matrix C (AO × MO)
- `fock`            : converged AO Fock matrix
- `nbasis`          : number of AO basis functions
"""
function scf(mol, h1e, eri, S, n_elec;
             maxiter=100, e_tol=1e-12, d_tol=0.0,
             diis=false, diis_size=6, diis_start=2,
             fock_builder=nothing,
             verbose=true)
    nuclear_repulsion = pyscf_nucr(mol)
    nocc   = n_elec ÷ 2
    nbasis = size(h1e, 1)
    X      = make_s_half(S)          # S^{-1/2}
    build_fock = isnothing(fock_builder) ?
        (D -> make_fock(D, h1e, eri)) :
        fock_builder

    # Initial Fock from core Hamiltonian
    F0 = X * h1e * X'
    F0 = (F0 .+ F0') ./ 2
    _, c0 = eigen(Symmetric(F0))
    c  = X * c0

    fock     = zeros(Float64, nbasis, nbasis)
    E_prev   = 0.0
    E_elec   = 0.0
    orb_energies = Float64[]

    fock_history = Matrix{Float64}[]
    error_history = Matrix{Float64}[]
    iterations = maxiter
    rms_D = Inf

    for n in 1:maxiter
        D_old = make_density(c, nocc)
        fock_built = build_fock(D_old)
        fock = fock_built

        if diis
            push!(fock_history, copy(fock_built))
            push!(error_history, diis_error_matrix(fock_built, D_old, S))

            if length(fock_history) > diis_size
                popfirst!(fock_history)
                popfirst!(error_history)
            end

            if length(fock_history) >= diis_start
                coefficients = diis_coefficients(error_history)
                fock = extrapolate_fock(fock_history, coefficients)
            end
        end

        F_orth    = X * fock * X'
        F_orth    = (F_orth .+ F_orth') ./ 2
        eps, c_orth = eigen(Symmetric(F_orth))
        orb_energies = eps
        c            = X * c_orth
        D            = make_density(c, nocc)

        E_elec = scf_energy(D, fock_built, h1e)
        delta  = abs(E_elec - E_prev)
        rms_D = sqrt(mean((D .- D_old).^2))

        if verbose
            @printf("  iter= %3d  E= %20.12f  ΔE= %12.4e  RMS(D)= %12.4e\n",
                    n, E_elec + nuclear_repulsion, delta, rms_D)
        end
        E_prev = E_elec
        if delta < e_tol && (d_tol <= 0 || rms_D < d_tol)
            iterations = n
            fock = build_fock(D)
            F_orth = X * fock * X'
            F_orth = (F_orth .+ F_orth') ./ 2
            orb_energies, c_orth = eigen(Symmetric(F_orth))
            c = X * c_orth
            E_elec = scf_energy(make_density(c, nocc), fock, h1e)
            break
        end
    end

    return (
        energy           = E_elec,
        total_energy     = E_elec + nuclear_repulsion,
        orbital_energies = orb_energies,
        mo_coeffs        = c,
        fock             = fock,
        nbasis           = nbasis,
        iterations       = iterations,
        rms_density      = rms_D,
    )
end

function scf_symmetry(mol, h1e, eri, S, n_elec, info::SymmetryInfo;
                      maxiter=100, e_tol=1e-12, d_tol=0.0,
                      diis=false, diis_size=6, diis_start=2,
                      verbose=true)
    sum(info.occupations) == n_elec ÷ 2 ||
        throw(ArgumentError("Symmetry occupations must sum to the number of occupied orbitals"))

    nuclear_repulsion = pyscf_nucr(mol)
    nbasis = size(h1e, 1)

    h_so = symmetry_project(symmetry_transform_1e(h1e, info), info)
    S_so = symmetry_project(symmetry_transform_1e(S, info), info)
    eri_so = symmetry_transform_eri(eri, info)
    X = block_s_half(S_so, info)

    F0 = symmetry_project(X * h_so * X', info)
    eps, c0 = block_eigen(F0, info)
    c = X * c0

    fock = zeros(Float64, nbasis, nbasis)
    E_prev = 0.0
    E_elec = 0.0
    orb_energies = eps
    iterations = maxiter
    rms_D = Inf

    fock_history = Matrix{Float64}[]
    error_history = Matrix{Float64}[]

    for n in 1:maxiter
        D_old = make_density_symmetry(c, info)
        fock_built = make_fock_symmetry(D_old, h_so, eri_so, info)
        fock = fock_built

        if diis
            push!(fock_history, copy(fock_built))
            push!(error_history, symmetry_project(diis_error_matrix(fock_built, D_old, S_so), info))

            if length(fock_history) > diis_size
                popfirst!(fock_history)
                popfirst!(error_history)
            end

            if length(fock_history) >= diis_start
                coefficients = diis_coefficients(error_history)
                fock = symmetry_project(extrapolate_fock(fock_history, coefficients), info)
            end
        end

        F_orth = symmetry_project(X * fock * X', info)
        eps, c_orth = block_eigen(F_orth, info)
        orb_energies = eps
        c = X * c_orth
        D = make_density_symmetry(c, info)

        E_elec = scf_energy(D, fock_built, h_so)
        delta = abs(E_elec - E_prev)
        rms_D = sqrt(mean((D .- D_old).^2))

        if verbose
            @printf("  iter= %3d  E= %20.12f  ΔE= %12.4e  RMS(D)= %12.4e\n",
                    n, E_elec + nuclear_repulsion, delta, rms_D)
        end
        E_prev = E_elec

        if delta < e_tol && (d_tol <= 0 || rms_D < d_tol)
            iterations = n
            fock = make_fock_symmetry(D, h_so, eri_so, info)
            F_orth = symmetry_project(X * fock * X', info)
            orb_energies, c_orth = block_eigen(F_orth, info)
            c = X * c_orth
            E_elec = scf_energy(make_density_symmetry(c, info), fock, h_so)
            break
        end
    end

    order = sortperm(orb_energies)
    c_so = c[:, order]
    mo_coeffs = info.transform * c_so
    fock_ao = symmetry_backtransform_1e(fock, info)

    return (
        energy           = E_elec,
        total_energy     = E_elec + nuclear_repulsion,
        orbital_energies = orb_energies[order],
        mo_coeffs        = mo_coeffs,
        fock             = fock_ao,
        nbasis           = nbasis,
        iterations       = iterations,
        rms_density      = rms_D,
        symmetry_info    = info,
        fock_so          = fock,
        mo_coeffs_so     = c_so,
    )
end

# ── Public entry point ─────────────────────────────────────────────────────────

"""
    run_rhf(; atoms, basis, charge, spin, unit, n_elec, symmetry, outcore, direct, verbose) -> NamedTuple

Build the molecule and run RHF.  Returns everything downstream code needs:

- `total_energy`    : HF total energy (Eh)
- `orbital_energies`: MO eigenvalues (array)
- `mo_coeffs`       : MO coefficient matrix C
- `fock`            : converged AO Fock matrix
- `h1e`             : AO core Hamiltonian
- `eri`             : AO two-electron integrals
- `nbasis`          : number of basis functions
- `n_elec`          : number of electrons

With `outcore=true`, the SCF Fock build streams permutationally unique ERIs
from disk.  Full AO ERIs are omitted from the result unless `return_eri=true`.
With `direct=true`, the SCF Fock build uses PySCF direct J/K contractions from
the current density and also omits full AO ERIs unless `return_eri=true`.
"""
function run_rhf(;
    atoms  = "O 0.000000000000 -0.143225816552 0.000000000000;" *
             "H 1.638036840407  1.136548822547 -0.000000000000;" *
             "H -1.638036840407 1.136548822547 -0.000000000000",
    basis  = "sto-3g",
    charge = 0,
    spin   = 0,
    unit   = "Bohr",
    n_elec = 10,
    diis = false,
    diis_size = 6,
    diis_start = 2,
    symmetry = false,
    symmetry_info = nothing,
    outcore = false,
    direct = false,
    eri_file = nothing,
    eri_path = nothing,
    eri_cutoff = 1e-14,
    eri_batch_size = 4096,
    keep_eri_file = false,
    return_eri = nothing,
    verbose = true,
)
    use_symmetry = !(symmetry === false)
    use_outcore = outcore === true
    use_direct = direct === true
    use_symmetry && use_outcore &&
        throw(ArgumentError("outcore=true currently supports the non-symmetry RHF path"))
    use_symmetry && use_direct &&
        throw(ArgumentError("direct=true currently supports the non-symmetry RHF path"))
    use_outcore && use_direct &&
        throw(ArgumentError("outcore=true and direct=true are mutually exclusive RHF Fock builders"))

    mol = make_molecule(atoms, basis, charge, spin, unit; symmetry=use_symmetry ? symmetry : false)
    h1e = pyscf_1e(mol)
    S   = pyscf_overlap(mol)
    include_full_eri = isnothing(return_eri) ? !(use_outcore || use_direct) : return_eri
    eri = include_full_eri || use_symmetry ? pyscf_2e(mol) : nothing

    outcore_file = nothing
    generated_temp_eri_file = false
    if use_outcore
        outcore_file =
            if eri_file !== nothing
                eri_file isa ERIFile ? eri_file : read_eri_file_header(String(eri_file))
            elseif eri !== nothing
                path = isnothing(eri_path) ? tempname() : String(eri_path)
                generated_temp_eri_file = isnothing(eri_path)
                write_unique_eri_file(eri; path, cutoff=eri_cutoff)
            else
                path = isnothing(eri_path) ? tempname() : String(eri_path)
                generated_temp_eri_file = isnothing(eri_path)
                write_unique_eri_file(mol; path, cutoff=eri_cutoff)
            end
    end

    if use_symmetry
        info = isnothing(symmetry_info) ?
            pyscf_symmetry_info(mol; h1e, S, n_elec) :
            symmetry_info
        size(info.transform, 1) == size(h1e, 1) ||
            throw(ArgumentError("Symmetry transform size does not match the basis size"))
        result = scf_symmetry(mol, h1e, eri, S, n_elec, info; diis, diis_size, diis_start, verbose)
    elseif use_outcore
        fock_builder = D -> make_fock_outcore(D, h1e, outcore_file; batch_size=eri_batch_size)
        result = scf(mol, h1e, nothing, S, n_elec; diis, diis_size, diis_start, fock_builder, verbose)
    elseif use_direct
        fock_builder = D -> make_fock_direct(D, h1e, mol)
        result = scf(mol, h1e, nothing, S, n_elec; diis, diis_size, diis_start, fock_builder, verbose)
    else
        result = scf(mol, h1e, eri, S, n_elec; diis, diis_size, diis_start, verbose)
    end
    verbose && @printf("\nRHF total energy = %20.12f Eh\n", result.total_energy)

    returned_eri_file = use_outcore && (!generated_temp_eri_file || keep_eri_file) ? outcore_file : nothing
    if use_outcore && generated_temp_eri_file && !keep_eri_file
        rm(outcore_file.path; force=true)
    end

    return merge(result, (
        mol=mol,
        atoms=atoms,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        h1e=h1e,
        eri=eri,
        S=S,
        n_elec=n_elec,
        nuclear_repulsion=result.total_energy - result.energy,
        outcore=use_outcore,
        direct=use_direct,
        eri_file=returned_eri_file,
    ))
end
