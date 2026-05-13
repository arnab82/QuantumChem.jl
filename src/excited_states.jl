"""
Excited-state methods from Crawford Project #12: CIS and TDHF/RPA.
"""

using Base.Threads: @threads

"""
    HARTREE_TO_EV

Conversion factor from Hartree to electron-volts:

```text
1 Eh = 27.211386245988 eV.
```
"""
const HARTREE_TO_EV = 27.211386245988

"""
    spinorbital_spatial(p) -> Int

Map an interleaved spin-orbital index to its parent spatial orbital:
`P = (p + 1) div 2`.
"""
spinorbital_spatial(p::Integer) = (p + 1) ÷ 2

"""
    same_spin(p, q) -> Bool

Return whether two interleaved spin-orbital indices have the same spin label.
Odd indices are alpha spin orbitals and even indices are beta spin orbitals.
"""
same_spin(p::Integer, q::Integer) = isodd(p) == isodd(q)

"""
    spinorbital_eri(eri, p, q, r, s) -> Float64

Return the antisymmetrized spin-orbital integral `<pq||rs>` from spatial
MO-basis ERIs:

```text
<pq||rs> = (P R | Q S) delta(spin_p, spin_r) delta(spin_q, spin_s)
         - (P S | Q R) delta(spin_p, spin_s) delta(spin_q, spin_r)
```

where `P`, `Q`, `R`, and `S` are the spatial orbitals associated with the
interleaved spin orbitals `p`, `q`, `r`, and `s`.
"""
function spinorbital_eri(eri, p, q, r, s)
    P = spinorbital_spatial(p)
    Q = spinorbital_spatial(q)
    R = spinorbital_spatial(r)
    S = spinorbital_spatial(s)
    value = 0.0
    same_spin(p, r) && same_spin(q, s) && (value += eri[P,R,Q,S])
    same_spin(p, s) && same_spin(q, r) && (value -= eri[P,S,Q,R])
    return value
end

"""
    excitation_index(i, a, nvirt) -> Int

Linear index for an occupied-virtual excitation `(i -> a)` in row-major
occupied-major order: `idx = (i - 1) * nvirt + a`.
"""
excitation_index(i, a, nvirt) = (i - 1) * nvirt + a

"""
    excitation_pairs(nocc, nvirt) -> Vector{Tuple{Int,Int}}

List the occupied-virtual excitation labels used by CIS/RPA matrices.  The
returned pairs are virtual-block indices `(i, a0)`, so the absolute
spin-orbital virtual index is `nocc + a0`.
"""
function excitation_pairs(nocc, nvirt)
    pairs = Vector{NTuple{2,Int}}(undef, nocc * nvirt)
    idx = 0
    for i in 1:nocc, a in 1:nvirt
        idx += 1
        pairs[idx] = (i, a)
    end
    return pairs
end

"""
    build_cis_spinorbital_matrix(eri, orbital_energies, n_elec) -> Matrix

Build the spin-orbital CIS Hamiltonian,

```text
A_ia,jb = delta_ij delta_ab (eps_a - eps_i) + <a j || i b>
```

The occupied spin orbitals are `1:n_elec`; virtual labels are stored relative
to that occupied block.
"""
function build_cis_spinorbital_matrix(eri, orbital_energies, n_elec)
    nbasis = length(orbital_energies)
    nocc = n_elec
    nvirt = 2 * nbasis - nocc
    pairs = excitation_pairs(nocc, nvirt)
    A = zeros(Float64, length(pairs), length(pairs))

    for (row, (i, a0)) in enumerate(pairs)
        a = nocc + a0
        eps_a = orbital_energies[spinorbital_spatial(a)]
        eps_i = orbital_energies[spinorbital_spatial(i)]
        for (col, (j, b0)) in enumerate(pairs)
            b = nocc + b0
            value = (i == j && a0 == b0) ? eps_a - eps_i : 0.0
            value += spinorbital_eri(eri, a, j, i, b)
            A[row,col] = value
        end
    end
    return (A .+ A') ./ 2
end

"""
    build_cis_spin_adapted_matrix(eri, orbital_energies, n_elec; multiplicity=:singlet)

Build the spatial-orbital CIS Hamiltonian for singlet or triplet states:

```text
singlet: A_ia,jb = delta_ij delta_ab (eps_a - eps_i)
                 + 2(ai|jb) - (ab|ji)
triplet: A_ia,jb = delta_ij delta_ab (eps_a - eps_i)
                 - (ab|ji)
```
"""
function build_cis_spin_adapted_matrix(eri, orbital_energies, n_elec; multiplicity=:singlet)
    multiplicity in (:singlet, :triplet) ||
        throw(ArgumentError("multiplicity must be :singlet or :triplet"))

    nocc = n_elec ÷ 2
    nbasis = length(orbital_energies)
    nvirt = nbasis - nocc
    pairs = excitation_pairs(nocc, nvirt)
    H = zeros(Float64, length(pairs), length(pairs))

    for (row, (i, a0)) in enumerate(pairs)
        a = nocc + a0
        for (col, (j, b0)) in enumerate(pairs)
            b = nocc + b0
            value = (i == j && a == b) ? orbital_energies[a] - orbital_energies[i] : 0.0
            multiplicity == :singlet && (value += 2.0 * eri[a,i,j,b])
            value -= eri[a,b,j,i]
            H[row,col] = value
        end
    end
    return (H .+ H') ./ 2
end

"""
    cis_excitation_energies(H) -> Vector{Float64}

Diagonalize a Hermitian CIS/TDA matrix and return sorted excitation energies
`omega`, satisfying `H c = omega c`.
"""
cis_excitation_energies(H) = sort(eigvals(Symmetric((H .+ H') ./ 2)))

"""
    cis_singlet_diagonal(eri, orbital_energies, n_elec) -> Vector{Float64}

Return the diagonal approximation used by Davidson-Liu preconditioning for
spin-adapted singlet CIS:

```text
A_ia,ia = eps_a - eps_i + 2(ai|ia) - (aa|ii)
```
"""
function cis_singlet_diagonal(eri, orbital_energies, n_elec)
    nocc = n_elec ÷ 2
    nbasis = length(orbital_energies)
    nvirt = nbasis - nocc
    diagonal = zeros(Float64, nocc * nvirt)

    for i in 1:nocc, a0 in 1:nvirt
        a = nocc + a0
        idx = excitation_index(i, a0, nvirt)
        diagonal[idx] = orbital_energies[a] - orbital_energies[i] +
                        2.0 * eri[a,i,i,a] - eri[a,a,i,i]
    end
    return diagonal
end

"""
    cis_singlet_sigma(eri, orbital_energies, n_elec, c) -> Vector

Matrix-free spin-adapted singlet CIS sigma build:

```text
sigma_ia = (eps_a - eps_i) c_ia
         + sum_jb [2(ai|jb) - (ab|ji)] c_jb
```

This is used by the Project #13 Davidson-Liu solver.
"""
function cis_singlet_sigma(eri, orbital_energies, n_elec, c)
    nocc = n_elec ÷ 2
    nbasis = length(orbital_energies)
    nvirt = nbasis - nocc
    length(c) == nocc * nvirt ||
        throw(DimensionMismatch("CIS vector length does not match occupied-virtual space"))

    sigma = zeros(Float64, length(c))
    @threads for row in eachindex(c)
        i = (row - 1) ÷ nvirt + 1
        a0 = (row - 1) % nvirt + 1
        a = nocc + a0

        value = (orbital_energies[a] - orbital_energies[i]) * c[row]
        @inbounds for j in 1:nocc, b0 in 1:nvirt
            b = nocc + b0
            col = excitation_index(j, b0, nvirt)
            value += (2.0 * eri[a,i,j,b] - eri[a,b,j,i]) * c[col]
        end
        sigma[row] = value
    end
    return sigma
end

function _unit_vector(dim, idx)
    vector = zeros(Float64, dim)
    vector[idx] = 1.0
    return vector
end

function _orthonormalize(vector, basis; threshold=1e-10)
    candidate = copy(vector)
    for _ in 1:2
        for b in basis
            candidate .-= dot(b, candidate) .* b
        end
    end

    candidate_norm = norm(candidate)
    candidate_norm > threshold || return nothing
    return candidate ./ candidate_norm
end

function _davidson_correction(residual, value, diagonal)
    correction = similar(residual)
    for idx in eachindex(residual)
        denominator = value - diagonal[idx]
        correction[idx] = abs(denominator) > 1e-12 ? residual[idx] / denominator : 0.0
    end
    return correction
end

"""
    davidson_liu(sigma, diagonal; nroots, ...)

Compute a few lowest eigenpairs of a real symmetric matrix using the
Davidson-Liu simultaneous expansion algorithm.  `sigma(v)` must return `H*v`,
and `diagonal` supplies the diagonal preconditioner.

At each subspace solve the Ritz residual is

```text
r_k = H x_k - theta_k x_k
```

and the correction vector uses the diagonal approximation

```text
q_i = r_i / (theta_k - H_ii).
```
"""
function davidson_liu(sigma, diagonal;
                      nroots=3, nguess=nroots,
                      maxiter=80, max_subspace=max(20, 6*nroots),
                      eigen_tol=1e-10, residual_tol=1e-8,
                      correction_tol=1e-10)
    dim = length(diagonal)
    1 <= nroots <= dim || throw(ArgumentError("nroots must be between 1 and the problem dimension"))
    nguess = min(dim, max(nroots, nguess))
    max_subspace = min(dim, max(max_subspace, nroots))

    guess_indices = partialsortperm(diagonal, 1:nguess)
    basis = [_unit_vector(dim, idx) for idx in guess_indices]
    sigmas = [sigma(vector) for vector in basis]

    values = fill(Inf, nroots)
    vectors = zeros(Float64, dim, nroots)
    residual_norms = fill(Inf, nroots)
    previous_values = fill(Inf, nroots)
    converged = false
    iterations = 0

    for iteration in 1:maxiter
        iterations = iteration
        B = hcat(basis...)
        HB = hcat(sigmas...)
        subspace = B' * HB
        subspace = (subspace .+ subspace') ./ 2

        sub_values, sub_vectors = eigen(Symmetric(subspace))
        roots = 1:nroots
        values = sub_values[roots]
        alpha = sub_vectors[:, roots]
        vectors = B * alpha
        sigma_vectors = HB * alpha

        residuals = [sigma_vectors[:, root] .- values[root] .* vectors[:, root]
                     for root in 1:nroots]
        residual_norms = norm.(residuals)
        eigen_deltas = abs.(values .- previous_values)

        if all(residual_norms .< residual_tol) && all(eigen_deltas .< eigen_tol)
            converged = true
            break
        end
        previous_values = copy(values)

        corrections = Vector{Float64}[]
        for root in 1:nroots
            residual_norms[root] < residual_tol && continue
            push!(corrections, _davidson_correction(residuals[root], values[root], diagonal))
        end

        if length(basis) + length(corrections) > max_subspace
            basis = [copy(vectors[:, root]) for root in 1:nroots]
            sigmas = [copy(sigma_vectors[:, root]) for root in 1:nroots]
        end

        added = false
        for correction in corrections
            q = _orthonormalize(correction, basis; threshold=correction_tol)
            q === nothing && continue
            push!(basis, q)
            push!(sigmas, sigma(q))
            added = true
        end

        if !added && length(basis) == dim
            converged = all(residual_norms .< residual_tol)
            break
        elseif !added
            remaining = setdiff(partialsortperm(diagonal, 1:dim), guess_indices)
            for idx in remaining
                q = _orthonormalize(_unit_vector(dim, idx), basis; threshold=correction_tol)
                q === nothing && continue
                push!(basis, q)
                push!(sigmas, sigma(q))
                push!(guess_indices, idx)
                added = true
                break
            end
            added || break
        end
    end

    return (
        values=values,
        vectors=vectors,
        residual_norms=residual_norms,
        converged=converged,
        iterations=iterations,
        subspace_size=length(basis),
    )
end

"""
    run_davidson_cis(rhf, mp2_result; nroots=3, verbose=true)

Project #13 Davidson-Liu solver for the lowest spin-adapted singlet CIS roots.
The matrix is never built explicitly; the code repeatedly applies the
matrix-vector product `sigma = A c`.
"""
function run_davidson_cis(rhf, mp2_result;
                          nroots=3, nguess=nroots,
                          maxiter=80, max_subspace=max(20, 6*nroots),
                          eigen_tol=1e-10, residual_tol=1e-8,
                          correction_tol=1e-10, verbose=true)
    eri = mp2_result.new_eri
    E = rhf.orbital_energies
    n_elec = rhf.n_elec
    diagonal = cis_singlet_diagonal(eri, E, n_elec)
    sigma = c -> cis_singlet_sigma(eri, E, n_elec, c)

    result = davidson_liu(sigma, diagonal;
                          nroots, nguess, maxiter, max_subspace,
                          eigen_tol, residual_tol, correction_tol)

    if verbose
        @printf("Davidson-Liu CIS singlet roots:\n")
        for root in 1:length(result.values)
            @printf("  %3d  %16.10f Eh  %16.8f eV  |r|=%9.2e\n",
                    root, result.values[root], result.values[root] * HARTREE_TO_EV,
                    result.residual_norms[root])
        end
    end

    return merge(result, (
        energies=result.values,
        energies_ev=result.values .* HARTREE_TO_EV,
        diagonal=diagonal,
    ))
end

"""
    build_rpa_b_matrix(eri, orbital_energies, n_elec) -> Matrix{Float64}

Build the spin-orbital TDHF/RPA coupling matrix:

```text
B_ia,jb = <a b || i j>
```

The `orbital_energies` argument is accepted for a signature parallel to the
CIS `A` builder; the coupling itself depends only on the two-electron
integrals.
"""
function build_rpa_b_matrix(eri, orbital_energies, n_elec)
    nbasis = length(orbital_energies)
    nocc = n_elec
    nvirt = 2 * nbasis - nocc
    pairs = excitation_pairs(nocc, nvirt)
    B = zeros(Float64, length(pairs), length(pairs))

    for (row, (i, a0)) in enumerate(pairs)
        a = nocc + a0
        for (col, (j, b0)) in enumerate(pairs)
            b = nocc + b0
            B[row,col] = spinorbital_eri(eri, a, b, i, j)
        end
    end
    return B
end

"""
    build_rpa_matrix(A, B) -> Matrix{Float64}

Assemble the full non-Hermitian TDHF/RPA eigenvalue problem:

```text
[ A   B ][X] = omega [X]
[-B  -A ][Y]         [Y]
```
"""
build_rpa_matrix(A, B) = [A B; -B -A]

function _real_sorted(values; atol=1e-8)
    cleaned = Float64[]
    for value in values
        abs(imag(value)) <= atol ||
            throw(ArgumentError("Eigenvalue has non-negligible imaginary part: $value"))
        push!(cleaned, real(value))
    end
    return sort(cleaned)
end

function _positive_sorted(values; atol=1e-8)
    return [value for value in _real_sorted(values; atol) if value > atol]
end

"""
    rpa_full_energies(A, B; positive_only=false) -> Vector{Float64}

Diagonalize the full RPA matrix `[A B; -B -A]`.  The spectrum should occur in
`+-omega` pairs for a stable RHF reference; set `positive_only=true` to keep
only the physical positive frequencies.
"""
function rpa_full_energies(A, B; positive_only=false)
    values = eigvals(build_rpa_matrix(A, B))
    return positive_only ? _positive_sorted(values) : _real_sorted(values)
end

"""
    rpa_reduced_energies(A, B; atol=1e-8) -> Vector{Float64}

Solve the squared Hermitian-like RPA problem

```text
(A - B)(A + B) Z = omega^2 Z
```

and return the positive square roots `omega`.
"""
function rpa_reduced_energies(A, B; atol=1e-8)
    omega2 = eigvals((A - B) * (A + B))
    values = Float64[]
    for value in omega2
        abs(imag(value)) <= atol ||
            throw(ArgumentError("Reduced RPA eigenvalue has non-negligible imaginary part: $value"))
        real_value = real(value)
        real_value > atol && push!(values, sqrt(real_value))
    end
    return sort(values)
end

"""
    run_excited_states(rhf, mp2_result; verbose=true) -> NamedTuple

Compute Project #12 CIS and TDHF/RPA excitation energies from RHF orbitals and
MO-basis ERIs.  The returned named tuple contains spin-orbital CIS, spin-adapted
singlet/triplet CIS, full RPA, and reduced RPA spectra.
"""
function run_excited_states(rhf, mp2_result; verbose=true)
    eri = mp2_result.new_eri
    E = rhf.orbital_energies
    n_elec = rhf.n_elec

    cis_matrix = build_cis_spinorbital_matrix(eri, E, n_elec)
    cis_factorization = eigen(Symmetric(cis_matrix))
    cis_energies = cis_factorization.values

    singlet_matrix = build_cis_spin_adapted_matrix(eri, E, n_elec; multiplicity=:singlet)
    singlet_factorization = eigen(Symmetric(singlet_matrix))
    singlet_energies = singlet_factorization.values
    triplet_matrix = build_cis_spin_adapted_matrix(eri, E, n_elec; multiplicity=:triplet)
    triplet_factorization = eigen(Symmetric(triplet_matrix))
    triplet_energies = triplet_factorization.values

    A = cis_matrix
    B = build_rpa_b_matrix(eri, E, n_elec)
    full_rpa_energies = rpa_full_energies(A, B)
    rpa_energies = [value for value in full_rpa_energies if value > 1e-8]
    reduced_rpa_energies = rpa_reduced_energies(A, B)

    if verbose
        @printf("Lowest CIS excitation energy = %12.8f Eh  %12.8f eV\n",
                cis_energies[1], cis_energies[1] * HARTREE_TO_EV)
        @printf("Lowest RPA excitation energy = %12.8f Eh  %12.8f eV\n",
                rpa_energies[1], rpa_energies[1] * HARTREE_TO_EV)
    end

    return (
        cis_matrix=cis_matrix,
        cis_energies=cis_energies,
        cis_vectors=cis_factorization.vectors,
        cis_energies_ev=cis_energies .* HARTREE_TO_EV,
        singlet_matrix=singlet_matrix,
        singlet_energies=singlet_energies,
        singlet_vectors=singlet_factorization.vectors,
        triplet_matrix=triplet_matrix,
        triplet_energies=triplet_energies,
        triplet_vectors=triplet_factorization.vectors,
        rpa_A=A,
        rpa_B=B,
        rpa_matrix=build_rpa_matrix(A, B),
        rpa_full_energies=full_rpa_energies,
        rpa_energies=rpa_energies,
        rpa_energies_ev=rpa_energies .* HARTREE_TO_EV,
        rpa_reduced_energies=reduced_rpa_energies,
    )
end
