"""
Equation-of-motion CCSD utilities for Crawford Project #14.

The public driver builds the EOM-CCSD Jacobian as a finite-difference
linearization of the converged CCSD amplitude residual equations in the
spin-orbital singles plus unique antisymmetric doubles excitation space.
"""

using Base.Threads: @threads

"""
    eom_double_indices(o, v) -> Vector{NTuple{4,Int}}

Return the unique antisymmetric double-excitation labels `(i, j, a, b)` with
`i < j` and `a < b`.  These labels store one independent amplitude for each
antisymmetric block:

```text
R_ji^ab = -R_ij^ab,  R_ij^ba = -R_ij^ab.
```
"""
function eom_double_indices(o, v)
    indices = NTuple{4,Int}[]
    for i in 1:o, j in (i + 1):o, a in 1:v, b in (a + 1):v
        push!(indices, (i, j, a, b))
    end
    return indices
end

"""
    eom_dimension(o, v) -> Int

Return the dimension of the EOM excitation vector:

```text
dim = o v + C(o, 2) C(v, 2).
```
"""
eom_dimension(o, v) = o * v + length(eom_double_indices(o, v))

"""
    pack_eom_amplitudes(rs, rd, o, v) -> Vector{Float64}

Pack singles `R_i^a` and unique antisymmetric doubles `R_ij^ab` into the
linear EOM vector `[R1; R2]`.
"""
function pack_eom_amplitudes(rs, rd, o, v)
    doubles = eom_double_indices(o, v)
    vector = Vector{Float64}(undef, o * v + length(doubles))
    idx = 0
    @inbounds for i in 1:o, a in 1:v
        idx += 1
        vector[idx] = rs[i,a]
    end
    @inbounds for (i, j, a, b) in doubles
        idx += 1
        vector[idx] = rd[i,j,a,b]
    end
    return vector
end

"""
    unpack_eom_amplitudes(vector, o, v) -> (rs, rd)

Unpack a linear EOM vector into singles and a fully antisymmetric doubles
tensor.  If the packed component stores `R_ij^ab`, the reconstructed tensor
also fills `R_ji^ab`, `R_ij^ba`, and `R_ji^ba` with the required signs.
"""
function unpack_eom_amplitudes(vector, o, v)
    doubles = eom_double_indices(o, v)
    length(vector) == o * v + length(doubles) ||
        throw(DimensionMismatch("EOM vector length does not match singles+doubles space"))

    rs = zeros(Float64, o, v)
    rd = zeros(Float64, o, o, v, v)
    idx = 0
    @inbounds for i in 1:o, a in 1:v
        idx += 1
        rs[i,a] = vector[idx]
    end
    @inbounds for (i, j, a, b) in doubles
        idx += 1
        value = vector[idx]
        rd[i,j,a,b] = value
        rd[j,i,a,b] = -value
        rd[i,j,b,a] = -value
        rd[j,i,b,a] = value
    end
    return rs, rd
end

"""
    ccsd_residuals(aso, F_so, E, o, v, nofe, ts, td) -> (rs, rd)

Evaluate the spin-orbital CCSD amplitude residuals used to linearize the
EOM-CCSD problem.  The Jacobi-updated amplitudes are first formed as

```text
t_new = numerator / denominator
R = denominator * (t_new - t).
```

At convergence these residuals are zero; EOM-CCSD uses their first derivative
with respect to excitation amplitudes.
"""
function ccsd_residuals(aso, F_so, E, o, v, nofe, ts, td)
    F_oo = F_so[1:o, 1:o]
    F_vv = F_so[(nofe+1):(nofe+v), (nofe+1):(nofe+v)]
    F_ov = F_so[1:o, (nofe+1):(nofe+v)]

    oi = 1:o
    vi = (nofe+1):(nofe+v)
    aso_oooo = aso[oi,oi,oi,oi]
    aso_oovv = aso[oi,oi,vi,vi]
    aso_ooov = aso[oi,oi,oi,vi]
    aso_oovo = aso[oi,oi,vi,oi]
    aso_vvvv = aso[vi,vi,vi,vi]
    aso_ovoo = aso[oi,vi,oi,oi]
    aso_ovvv = aso[oi,vi,vi,vi]
    aso_ovov = aso[oi,vi,oi,vi]
    aso_ovvo = aso[oi,vi,vi,oi]
    aso_vovv = aso[vi,oi,vi,vi]
    aso_vvvo = aso[vi,vi,vi,oi]

    Dia = zeros(Float64, o, v)
    @threads for idx in eachindex(Dia)
        x = idx - 1
        i = (x % o) + 1
        a = (x ÷ o) + 1
        @inbounds Dia[i,a] = F_oo[i,i] - F_vv[a,a]
    end

    tao = zeros(Float64, o, o, v, v)
    taobar = zeros(Float64, o, o, v, v)
    @threads for idx in eachindex(td)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds begin
            outer = ts[i,a] * ts[j,b] - ts[i,b] * ts[j,a]
            tao[i,j,a,b] = td[i,j,a,b] + outer
            taobar[i,j,a,b] = td[i,j,a,b] + 0.5 * outer
        end
    end

    _fae1 = zeros(Float64,v,v); @einsum _fae1[a,e] = ts[m,a] * F_ov[m,e]
    _fae2 = zeros(Float64,v,v); @einsum _fae2[a,e] = ts[m,f] * aso_ovvv[m,a,f,e]
    _fae3 = zeros(Float64,v,v); @einsum _fae3[a,e] = taobar[m,n,a,f] * aso_oovv[m,n,e,f]
    Fae = -0.5 .* _fae1 .+ _fae2 .- 0.5 .* _fae3
    @threads for idx in eachindex(Fae)
        x = idx - 1
        a = (x % v) + 1
        e = (x ÷ v) + 1
        @inbounds Fae[a,e] += (1 - (a == e)) * F_vv[a,e]
    end

    _fmi1 = zeros(Float64,o,o); @einsum _fmi1[m,i] = ts[i,e] * F_ov[m,e]
    _fmi2 = zeros(Float64,o,o); @einsum _fmi2[m,i] = ts[n,e] * aso_ooov[m,n,i,e]
    _fmi3 = zeros(Float64,o,o); @einsum _fmi3[m,i] = taobar[i,n,e,f] * aso_oovv[m,n,e,f]
    Fmi = 0.5 .* _fmi1 .+ _fmi2 .+ 0.5 .* _fmi3
    @threads for idx in eachindex(Fmi)
        x = idx - 1
        m = (x % o) + 1
        i = (x ÷ o) + 1
        @inbounds Fmi[m,i] += (1 - (m == i)) * F_oo[m,i]
    end

    _fme1 = zeros(Float64,o,v); @einsum _fme1[m,e] = ts[n,f] * aso_oovv[m,n,e,f]
    Fme = F_ov .+ _fme1

    _w1 = zeros(Float64,o,o,o,o); @einsum _w1[m,n,i,j] = ts[j,e] * aso_ooov[m,n,i,e]
    _w2 = zeros(Float64,o,o,o,o); @einsum _w2[m,n,i,j] = tao[i,j,e,f] * aso_oovv[m,n,e,f]
    Wmnij = zeros(Float64, o, o, o, o)
    @threads for idx in eachindex(Wmnij)
        x = idx - 1
        m = (x % o) + 1; x ÷= o
        n = (x % o) + 1; x ÷= o
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1
        @inbounds Wmnij[m,n,i,j] = aso_oooo[m,n,i,j] + _w1[m,n,i,j] - _w1[m,n,j,i] + 0.25 * _w2[m,n,i,j]
    end

    _w3 = zeros(Float64,v,v,v,v); @einsum _w3[a,b,e,f] = ts[m,b] * aso_vovv[a,m,e,f]
    _w4 = zeros(Float64,v,v,v,v); @einsum _w4[a,b,e,f] = tao[m,n,a,b] * aso_oovv[m,n,e,f]
    Wabef = zeros(Float64, v, v, v, v)
    @threads for idx in eachindex(Wabef)
        x = idx - 1
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1; x ÷= v
        e = (x % v) + 1; x ÷= v
        f = (x % v) + 1
        @inbounds Wabef[a,b,e,f] = aso_vvvv[a,b,e,f] - _w3[a,b,e,f] + _w3[b,a,e,f] + 0.25 * _w4[a,b,e,f]
    end

    _w5 = zeros(Float64,o,v,v,o); @einsum _w5[m,b,e,j] = ts[j,f] * aso_ovvv[m,b,e,f]
    _w6 = zeros(Float64,o,v,v,o); @einsum _w6[m,b,e,j] = ts[n,b] * aso_oovo[m,n,e,j]
    _tw = zeros(Float64,o,o,v,v); @einsum _tw[j,n,f,b] = ts[j,f] * ts[n,b]
    _tw .= 0.5 .* td .+ _tw
    _w7 = zeros(Float64,o,v,v,o); @einsum _w7[m,b,e,j] = _tw[j,n,f,b] * aso_oovv[m,n,e,f]
    Wmbej = zeros(Float64, o, v, v, o)
    @threads for idx in eachindex(Wmbej)
        x = idx - 1
        m = (x % o) + 1; x ÷= o
        b = (x % v) + 1; x ÷= v
        e = (x % v) + 1; x ÷= v
        j = (x % o) + 1
        @inbounds Wmbej[m,b,e,j] = aso_ovvo[m,b,e,j] + _w5[m,b,e,j] - _w6[m,b,e,j] - _w7[m,b,e,j]
    end

    _t11 = zeros(Float64,o,v); @einsum _t11[i,a] = ts[i,e] * Fae[a,e]
    _t12 = zeros(Float64,o,v); @einsum _t12[i,a] = ts[m,a] * Fmi[m,i]
    _t13 = zeros(Float64,o,v); @einsum _t13[i,a] = td[i,m,a,e] * Fme[m,e]
    _t14 = zeros(Float64,o,v); @einsum _t14[i,a] = ts[n,f] * aso_ovov[n,a,i,f]
    _t15 = zeros(Float64,o,v); @einsum _t15[i,a] = td[i,m,e,f] * aso_ovvv[m,a,e,f]
    _t16 = zeros(Float64,o,v); @einsum _t16[i,a] = td[m,n,a,e] * aso_oovo[n,m,e,i]
    tsnew = (F_ov .+ _t11 .- _t12 .+ _t13 .- _t14 .- 0.5 .* _t15 .- 0.5 .* _t16) ./ Dia

    _tbe = zeros(Float64,v,v); @einsum _tbe[b,e] = ts[m,b] * Fme[m,e]
    Fae_mod = Fae .- 0.5 .* _tbe
    _tjm = zeros(Float64,o,o); @einsum _tjm[m,j] = ts[j,e] * Fme[m,e]
    Fmi_mod = Fmi .+ 0.5 .* _tjm

    _t2a = zeros(Float64,o,o,v,v); @einsum _t2a[i,j,a,b] = td[i,j,a,e] * Fae_mod[b,e]
    _t2b = zeros(Float64,o,o,v,v); @einsum _t2b[i,j,a,b] = td[i,m,a,b] * Fmi_mod[m,j]
    _t2c = zeros(Float64,o,o,v,v); @einsum _t2c[i,j,a,b] = tao[m,n,a,b] * Wmnij[m,n,i,j]
    _t2d = zeros(Float64,o,o,v,v); @einsum _t2d[i,j,a,b] = tao[i,j,e,f] * Wabef[a,b,e,f]
    _t2e = zeros(Float64,o,o,v,v); @einsum _t2e[i,j,a,b] = td[i,m,a,e] * Wmbej[m,b,e,j]
    _t2ets = zeros(Float64,o,o,v,v)
    @einsum _t2ets[i,j,a,b] = ts[i,e] * ts[m,a] * aso_ovvo[m,b,e,j]
    _t2f = zeros(Float64,o,o,v,v); @einsum _t2f[i,j,a,b] = ts[i,e] * aso_vvvo[a,b,e,j]
    _t2g = zeros(Float64,o,o,v,v); @einsum _t2g[i,j,a,b] = ts[m,a] * aso_ovoo[m,b,i,j]

    tdnew = zeros(Float64, o, o, v, v)
    @threads for idx in eachindex(tdnew)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds begin
            num = (aso_oovv[i,j,a,b]
                   + _t2a[i,j,a,b] - _t2a[i,j,b,a]
                   - _t2b[i,j,a,b] + _t2b[j,i,a,b]
                   + 0.5 * _t2c[i,j,a,b] + 0.5 * _t2d[i,j,a,b]
                   + (_t2e[i,j,a,b] - _t2ets[i,j,a,b])
                   - (_t2e[j,i,a,b] - _t2ets[j,i,a,b])
                   - (_t2e[i,j,b,a] - _t2ets[i,j,b,a])
                   + (_t2e[j,i,b,a] - _t2ets[j,i,b,a])
                   + _t2f[i,j,a,b] - _t2f[j,i,a,b]
                   - _t2g[i,j,a,b] + _t2g[i,j,b,a])
            tdnew[i,j,a,b] = num / (Dia[i,a] + Dia[j,b])
        end
    end

    rs = Dia .* (tsnew .- ts)
    rd = similar(td)
    @threads for idx in eachindex(td)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds rd[i,j,a,b] = (Dia[i,a] + Dia[j,b]) * (tdnew[i,j,a,b] - td[i,j,a,b])
    end
    return rs, rd
end

"""
    eom_ccsd_diagonal(F_so, o, v, nofe) -> (singles, doubles)

Build orbital-energy/Fock diagonal estimates for the EOM-CCSD Jacobian:

```text
Delta_i^a     = f_aa - f_ii
Delta_ij^ab   = f_aa + f_bb - f_ii - f_jj.
```
"""
function eom_ccsd_diagonal(F_so, o, v, nofe)
    singles = zeros(Float64, o, v)
    @inbounds for i in 1:o, a in 1:v
        aa = nofe + a
        singles[i,a] = F_so[aa,aa] - F_so[i,i]
    end

    doubles = zeros(Float64, o, o, v, v)
    @inbounds for i in 1:o, j in 1:o, a in 1:v, b in 1:v
        aa = nofe + a
        bb = nofe + b
        doubles[i,j,a,b] = F_so[aa,aa] + F_so[bb,bb] - F_so[i,i] - F_so[j,j]
    end
    return singles, doubles
end

"""
    build_eom_ccsd_jacobian(aso, F_so, E, o, v, nofe, ts, td; finite_difference_step=1e-6)

Build a finite-difference EOM-CCSD Jacobian from the converged CCSD residual
function:

```text
J_{mu,nu} = d R_mu / d T_nu
          approx [R_mu(T + h e_nu) - R_mu(T)] / h.
```

The returned `base` vector is the residual at the supplied amplitudes, useful
for confirming that the CCSD reference is converged.
"""
function build_eom_ccsd_jacobian(aso, F_so, E, o, v, nofe, ts, td;
                                 finite_difference_step=1e-6)
    dim = eom_dimension(o, v)
    base = pack_eom_amplitudes(ccsd_residuals(aso, F_so, E, o, v, nofe, ts, td)..., o, v)
    J = zeros(Float64, dim, dim)

    for col in 1:dim
        direction = zeros(Float64, dim)
        direction[col] = 1.0
        dts, dtd = unpack_eom_amplitudes(direction, o, v)
        perturbed = pack_eom_amplitudes(
            ccsd_residuals(aso, F_so, E, o, v, nofe,
                           ts .+ finite_difference_step .* dts,
                           td .+ finite_difference_step .* dtd)...,
            o, v,
        )
        J[:, col] .= (perturbed .- base) ./ finite_difference_step
    end
    return J, base
end

function _positive_real_eom_roots(values; atol=1e-7)
    roots = Float64[]
    for value in values
        abs(imag(value)) <= atol || continue
        real_value = real(value)
        real_value > atol && push!(roots, real_value)
    end
    return sort(roots)
end

function _positive_real_eom_root_indices(values; atol=1e-7)
    indices = Int[]
    for (idx, value) in pairs(values)
        abs(imag(value)) <= atol || continue
        real(value) > atol && push!(indices, idx)
    end
    return sort(indices; by=idx -> real(values[idx]))
end

function _real_eom_vectors(vectors, indices; atol=1e-7)
    selected = zeros(Float64, size(vectors, 1), length(indices))
    for (col, idx) in enumerate(indices)
        vector = vectors[:, idx]
        maximum(abs, imag.(vector)) <= atol ||
            throw(ArgumentError("EOM-CCSD eigenvector has non-negligible imaginary part"))
        real_vector = real.(vector)
        vector_norm = norm(real_vector)
        vector_norm > 0.0 || throw(ArgumentError("EOM-CCSD eigenvector has zero norm"))
        selected[:, col] .= real_vector ./ vector_norm
    end
    return selected
end

"""
    run_eom_ccsd(rhf, mp2_result, ccsd_result; nroots=5, verbose=true)

Build and diagonalize the EOM-CCSD Jacobian in the spin-orbital singles plus
unique doubles excitation space.  Project #14 in the Crawford repository only
contains a title, so this implementation uses the converged CCSD residual
linearization as a compact, inspectable educational EOM-CCSD route:

```text
J R_k = omega_k R_k.
```
"""
function run_eom_ccsd(rhf, mp2_result, ccsd_result;
                      nroots=5, finite_difference_step=1e-6, verbose=true)
    ccsd_result.converged ||
        throw(ArgumentError("EOM-CCSD requires converged CCSD amplitudes"))

    n_elec = rhf.n_elec
    o = n_elec
    v = 2 * rhf.nbasis - n_elec
    aso = mo_to_aso(mp2_result.new_eri)
    F_so = Mat_motoMat_so(Mat_aotoMat_mo(rhf.mo_coeffs, rhf.fock))

    jacobian, residual = build_eom_ccsd_jacobian(
        aso, F_so, rhf.orbital_energies, o, v, n_elec,
        ccsd_result.ts, ccsd_result.td;
        finite_difference_step,
    )
    factorization = eigen(jacobian)
    values = factorization.values
    root_indices = _positive_real_eom_root_indices(values)
    count = min(nroots, length(root_indices))
    selected_indices = root_indices[1:count]
    energies = real.(values[selected_indices])
    vectors = _real_eom_vectors(factorization.vectors, selected_indices)

    singles_diag, doubles_diag = eom_ccsd_diagonal(F_so, o, v, n_elec)
    diagonal = pack_eom_amplitudes(singles_diag, doubles_diag, o, v)

    if verbose
        @printf("EOM-CCSD excitation roots:\n")
        for root in 1:length(energies)
            @printf("  %3d  %16.10f Eh  %16.8f eV\n",
                    root, energies[root], energies[root] * HARTREE_TO_EV)
        end
    end

    return (
        energies=energies,
        energies_ev=energies .* HARTREE_TO_EV,
        all_eigenvalues=values,
        all_eigenvectors=factorization.vectors,
        root_indices=selected_indices,
        vectors=vectors,
        jacobian=jacobian,
        residual=residual,
        diagonal=diagonal,
        dimension=size(jacobian, 1),
        singles=o * v,
        doubles=length(eom_double_indices(o, v)),
        finite_difference_step=finite_difference_step,
    )
end
