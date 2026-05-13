"""
Coupled-Cluster Singles and Doubles (CCSD) in the antisymmetrized spin-orbital (ASO) basis.

Follows Stanton et al., J. Chem. Phys. 94, 4334 (1991).

Spin-orbital index convention
  occupied : 1…o  (= n_elec spin-orbitals, paired α/β)
  virtual  : 1…v  (shifted by o when indexing into full ASO tensors)
  spatial p ↔ spin-orbitals 2p-1 (α) and 2p (β)

Public entry point: `run_ccsd(rhf, mp2_result; ...)`
"""

using Einsum, Printf, TensorOperations
using Base.Threads: @threads, maxthreadid, threadid

# ─────────────────────────────────────────────────────────────────────────────
# AO/MO → spin-orbital basis transformations
# ─────────────────────────────────────────────────────────────────────────────

"""
    mo_to_aso(new_eri) -> Array{Float64,4}

Transform MO ERIs to the antisymmetrized spin-orbital (ASO) basis:
    <pq||rs> = <pq|rs> δ_{σp,σr} δ_{σq,σs} − <pq|sr> δ_{σp,σs} δ_{σq,σr}

Spatial orbital p → spin-orbitals 2p-1 (α) and 2p (β).
"""
function mo_to_aso(new_eri)
    _, nbasis = size(new_eri)
    nso = 2 * nbasis
    spatial = [(i + 1) ÷ 2 for i in 1:nso]
    spin = [isodd(i) for i in 1:nso]
    A = zeros(Float64, nso, nso, nso, nso)
    @threads for idx in eachindex(A)
        x = idx - 1
        i = (x % nso) + 1; x ÷= nso
        j = (x % nso) + 1; x ÷= nso
        k = (x % nso) + 1; x ÷= nso
        l = (x % nso) + 1

        @inbounds begin
            p = spatial[i]; q = spatial[j]
            r = spatial[k]; s = spatial[l]
            A[i,j,k,l] = (new_eri[p,r,q,s] * (spin[i] == spin[k]) * (spin[j] == spin[l])
                          - new_eri[p,s,q,r] * (spin[i] == spin[l]) * (spin[j] == spin[k]))
        end
    end
    return A
end

"""
    Mat_aotoMat_mo(c, h_ao) -> Matrix{Float64}

Transform one-electron AO matrix to the MO basis:
    h_MO[p,q] = ∑_{ij} C_{ip} h_{ij} C_{jq}
"""
function Mat_aotoMat_mo(c, h_ao)
    @tensoropt h_mo[p,q] := c[i,p] * h_ao[i,j] * c[j,q]
    return h_mo
end

"""
    Mat_motoMat_so(Mat_mo) -> Matrix{Float64}

Lift a diagonal MO matrix to the spin-orbital (SO) basis.
Each spatial orbital energy ε_p fills both α (2p-1) and β (2p) diagonal entries.
"""
function Mat_motoMat_so(Mat_mo)
    _, nbasis = size(Mat_mo)
    F = zeros(Float64, 2*nbasis, 2*nbasis)
    @threads for i in 1:(2*nbasis)
        p = (i + 1) ÷ 2
        @inbounds F[i,i] = Mat_mo[p,p]
    end
    return F
end

# ─────────────────────────────────────────────────────────────────────────────
# MP2 initial amplitudes and energy in the SO basis
# ─────────────────────────────────────────────────────────────────────────────

"""
    make_td(aso_oovv, E, o, nofe) -> Array{Float64,4}

Compute MP2 (zeroth-iteration CCSD) doubles amplitudes:
    t_{ij}^{ab} = <ij||ab> / (ε_i + ε_j − ε_a − ε_b)
"""
function make_td(aso_oovv, E, o, nofe)
    v  = size(aso_oovv, 3)
    td = zeros(Float64, o, o, v, v)
    eps_occ = [E[(i + 1) ÷ 2] for i in 1:o]
    eps_vir = [E[(a + nofe + 1) ÷ 2] for a in 1:v]
    @threads for idx in eachindex(td)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds td[i,j,a,b] = aso_oovv[i,j,a,b] / (eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b])
    end
    return td
end

"""
    make_td_spinorbital(aso_oovv, eps_occ, eps_vir) -> Array{Float64,4}

Build MP2 doubles amplitudes from explicit occupied and virtual spin-orbital
energies.  This is used by UCCSD, where alpha and beta orbital energies are not
paired spatially.
"""
function make_td_spinorbital(aso_oovv, eps_occ, eps_vir)
    o = length(eps_occ)
    v = length(eps_vir)
    size(aso_oovv) == (o, o, v, v) ||
        throw(DimensionMismatch("ASO OOVV block does not match spin-orbital energy lengths"))

    td = zeros(Float64, o, o, v, v)
    @threads for idx in eachindex(td)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds td[i,j,a,b] = aso_oovv[i,j,a,b] / (eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b])
    end
    return td
end

"""
    mp2_so(oovv, td) -> Float64

MP2 correlation energy in the SO basis:
    E_MP2 = (1/4) ∑_{ijab} <ij||ab> t_{ij}^{ab}
"""
function mp2_so(oovv, td)
    partial = zeros(Float64, maxthreadid())
    @threads for idx in eachindex(td)
        @inbounds partial[threadid()] += 0.25 * oovv[idx] * td[idx]
    end
    return sum(partial)
end

# ─────────────────────────────────────────────────────────────────────────────
# CC amplitude DIIS helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    cc_amplitude_error(ts_new, td_new, ts_old, td_old) -> Vector{Float64}

Flattened Project #10 CC DIIS error vector:
    e_i = T_i - T_{i-1}

The caller should pass the freshly computed, non-extrapolated amplitudes as the
new amplitudes.  In the CCSD loop this is used as the Jacobi residual
`T_new(raw) - T_current`.
"""
function cc_amplitude_error(ts_new, td_new, ts_old, td_old)
    length(ts_new) == length(ts_old) ||
        throw(DimensionMismatch("T1 amplitude sizes differ"))
    length(td_new) == length(td_old) ||
        throw(DimensionMismatch("T2 amplitude sizes differ"))

    error = Vector{Float64}(undef, length(ts_new) + length(td_new))
    @inbounds for idx in eachindex(ts_new)
        error[idx] = ts_new[idx] - ts_old[idx]
    end
    offset = length(ts_new)
    @inbounds for idx in eachindex(td_new)
        error[offset + idx] = td_new[idx] - td_old[idx]
    end
    return error
end

function _regularized_diis_coefficients(errors; regularization=1e-12)
    n = length(errors)
    n >= 2 || throw(ArgumentError("DIIS needs at least two error vectors"))

    B = zeros(Float64, n + 1, n + 1)
    for i in 1:n, j in 1:i
        value = dot(errors[i], errors[j])
        B[i,j] = value
        B[j,i] = value
    end
    for i in 1:n
        B[i,i] += regularization
    end
    B[1:n,n+1] .= -1.0
    B[n+1,1:n] .= -1.0

    rhs = zeros(Float64, n + 1)
    rhs[n+1] = -1.0
    return (B \ rhs)[1:n]
end

"""
    cc_diis_coefficients(errors) -> Vector{Float64}

Pulay coefficients for flattened CC amplitude DIIS residuals.  Uses the same
linear equations as SCF DIIS, with a tiny diagonal regularization only if the
DIIS B matrix is singular.
"""
function cc_diis_coefficients(errors)
    try
        return diis_coefficients(errors)
    catch err
        err isa SingularException || rethrow()
        return _regularized_diis_coefficients(errors)
    end
end

"""
    extrapolate_amplitudes(amplitude_history, coefficients) -> (ts, td)

Linear-combine stored `(T1, T2)` amplitude tuples with Pulay coefficients.
"""
function extrapolate_amplitudes(amplitude_history, coefficients)
    length(amplitude_history) == length(coefficients) ||
        throw(ArgumentError("Amplitude and DIIS coefficient counts differ"))

    ts_template, td_template = amplitude_history[end]
    ts = zeros(Float64, size(ts_template))
    td = zeros(Float64, size(td_template))
    for (coefficient, (ts_old, td_old)) in zip(coefficients, amplitude_history)
        ts .+= coefficient .* ts_old
        td .+= coefficient .* td_old
    end
    return ts, td
end

"""
    ccsd_energy(aso_oovv, F_ov, ts, td) -> Float64

Evaluate the CCSD correlation energy in the antisymmetrized spin-orbital basis:

```text
E_CCSD = sum_ia f_i^a t_i^a
       + 1/4 sum_ijab <ij||ab> t_ij^ab
       + 1/2 sum_ijab <ij||ab> t_i^a t_j^b.
```
"""
function ccsd_energy(aso_oovv, F_ov, ts, td)
    o, _, v, _ = size(td)

    singles_energy = zeros(Float64, maxthreadid())
    @threads for idx in eachindex(ts)
        @inbounds singles_energy[threadid()] += F_ov[idx] * ts[idx]
    end

    doubles_energy = zeros(Float64, maxthreadid())
    @threads for idx in eachindex(td)
        x = idx - 1
        i = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        a = (x % v) + 1; x ÷= v
        b = (x % v) + 1
        @inbounds doubles_energy[threadid()] += (
            0.25 * aso_oovv[i,j,a,b] * td[i,j,a,b]
            + 0.5 * aso_oovv[i,j,a,b] * ts[i,a] * ts[j,b]
        )
    end
    return sum(singles_energy) + sum(doubles_energy)
end

# ─────────────────────────────────────────────────────────────────────────────
# CCSD amplitude iterations
# ─────────────────────────────────────────────────────────────────────────────

"""
    ccsd_scf(aso, F_so, E, o, v, nofe; maxiter, e_tol, amp_tol,
             diis, diis_size, diis_start, verbose)

Core CCSD iteration loop.  All inputs are pre-built arrays; no global state.

- `aso`   : full ASO 4-index tensor (2*nbasis)^4
- `F_so`  : spin-orbital Fock matrix (2*nbasis × 2*nbasis)
- `E`     : spatial orbital energies from RHF
- `o`/`v` : number of occupied/virtual spin-orbitals
- `nofe`  : total electron count (= o)

Returns `(E_ccsd, ts, td, converged, iterations)`.
"""
function ccsd_scf(aso, F_so, E, o, v, nofe;
                  maxiter=500, e_tol=1e-10, amp_tol=1e-10,
                  diis=false, diis_size=8, diis_start=3, verbose=true)

    if diis
        diis_size >= 2 || throw(ArgumentError("CC amplitude DIIS needs diis_size >= 2"))
        diis_start >= 3 || throw(ArgumentError("CC amplitude DIIS should start after at least 3 iterations"))
    end

    # ── Fock blocks ──────────────────────────────────────────────────────────
    F_oo = F_so[1:o, 1:o]
    F_vv = F_so[(nofe+1):(nofe+v), (nofe+1):(nofe+v)]
    F_ov = F_so[1:o, (nofe+1):(nofe+v)]

    # ── ASO integral slices (copies so inner loops hit dense arrays) ──────────
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

    # ── Møller-Plesset denominators (2D: Dia[i,a] = f_ii − f_aa) ─────────────
    Dia = zeros(Float64, o, v)
    @threads for idx in eachindex(Dia)
        x = idx - 1
        i = (x % o) + 1
        a = (x ÷ o) + 1
        @inbounds Dia[i,a] = F_oo[i,i] - F_vv[a,a]
    end

    # ── Initial amplitudes: T1=0, T2=MP2 ─────────────────────────────────────
    ts     = zeros(Float64, o, v)
    td     = length(E) == o + v ?
        make_td_spinorbital(aso_oovv, E[1:o], E[(nofe+1):(nofe+v)]) :
        make_td(aso_oovv, E, o, nofe)
    E_ccsd = 0.0

    if verbose
        @printf("  MP2 correlation (SO basis) = %.12f\n", mp2_so(aso_oovv, td))
        println("\n  CCSD SCF Iterations:")
        println("  " * "─"^64)
        @printf("  %6s  %20s  %12s  %12s\n", "Iter", "E_corr", "ΔE", "‖ΔT‖")
        println("  " * "─"^64)
    end

    amplitude_history = Tuple{Matrix{Float64}, Array{Float64,4}}[]
    error_history = Vector{Float64}[]
    converged = false
    iterations = maxiter
    for iteration in 1:maxiter

        # τ_{ij}^{ab} = t_{ij}^{ab} + t_i^a t_j^b − t_i^b t_j^a
        # τ̃_{ij}^{ab} = t_{ij}^{ab} + ½(t_i^a t_j^b − t_i^b t_j^a)
        tao    = zeros(Float64, o, o, v, v)
        taobar = zeros(Float64, o, o, v, v)
        @threads for idx in eachindex(td)
            x = idx - 1
            i = (x % o) + 1; x ÷= o
            j = (x % o) + 1; x ÷= o
            a = (x % v) + 1; x ÷= v
            b = (x % v) + 1
            @inbounds begin
                outer = ts[i,a]*ts[j,b] - ts[i,b]*ts[j,a]
                tao[i,j,a,b]    = td[i,j,a,b] + outer
                taobar[i,j,a,b] = td[i,j,a,b] + 0.5*outer
            end
        end

        # ── F̃_ae: (1−δ)f_ae − ½∑_m t_m^a f_{me} + ∑_{mf} t_m^f <ma||fe>
        #           − ½∑_{mnf} τ̃_{mn}^{af} <mn||ef>
        # Each contraction goes into its own array to avoid the @einsum
        # "zero-before-accumulate" pitfall when the destination appears on both sides.
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

        # ── F̃_mi: (1−δ)f_mi + ½∑_e t_i^e f_{me} + ∑_{ne} t_n^e <mn||ie>
        #           + ½∑_{nef} τ̃_{in}^{ef} <mn||ef>
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

        # ── F̃_me = f_{me} + ∑_{nf} t_n^f <mn||ef>
        _fme1 = zeros(Float64,o,v); @einsum _fme1[m,e] = ts[n,f] * aso_oovv[m,n,e,f]
        Fme = F_ov .+ _fme1

        # ── W_{mnij} = <mn||ij> + P̂(ij)∑_e t_j^e <mn||ie> + ¼∑_{ef} τ_{ij}^{ef} <mn||ef>
        _w1 = zeros(Float64,o,o,o,o); @einsum _w1[m,n,i,j] = ts[j,e] * aso_ooov[m,n,i,e]
        _w2 = zeros(Float64,o,o,o,o); @einsum _w2[m,n,i,j] = tao[i,j,e,f] * aso_oovv[m,n,e,f]
        Wmnij = zeros(Float64, o, o, o, o)
        @threads for idx in eachindex(Wmnij)
            x = idx - 1
            m = (x % o) + 1; x ÷= o
            n = (x % o) + 1; x ÷= o
            i = (x % o) + 1; x ÷= o
            j = (x % o) + 1
            @inbounds Wmnij[m,n,i,j] = aso_oooo[m,n,i,j] + _w1[m,n,i,j] - _w1[m,n,j,i] + 0.25*_w2[m,n,i,j]
        end

        # ── W_{abef} = <ab||ef> − P̂(ab)∑_m t_m^b <am||ef> + ¼∑_{mn} τ_{mn}^{ab} <mn||ef>
        _w3 = zeros(Float64,v,v,v,v); @einsum _w3[a,b,e,f] = ts[m,b] * aso_vovv[a,m,e,f]
        _w4 = zeros(Float64,v,v,v,v); @einsum _w4[a,b,e,f] = tao[m,n,a,b] * aso_oovv[m,n,e,f]
        Wabef = zeros(Float64, v, v, v, v)
        @threads for idx in eachindex(Wabef)
            x = idx - 1
            a = (x % v) + 1; x ÷= v
            b = (x % v) + 1; x ÷= v
            e = (x % v) + 1; x ÷= v
            f = (x % v) + 1
            @inbounds Wabef[a,b,e,f] = aso_vvvv[a,b,e,f] - _w3[a,b,e,f] + _w3[b,a,e,f] + 0.25*_w4[a,b,e,f]
        end

        # ── W_{mbej} = <mb||ej> + ∑_f t_j^f <mb||ef> − ∑_n t_n^b <mn||ej>
        #              − ∑_{nf}(½ t_{jn}^{fb} + t_j^f t_n^b) <mn||ef>
        _w5  = zeros(Float64,o,v,v,o); @einsum _w5[m,b,e,j]  = ts[j,f] * aso_ovvv[m,b,e,f]
        _w6  = zeros(Float64,o,v,v,o); @einsum _w6[m,b,e,j]  = ts[n,b] * aso_oovo[m,n,e,j]
        _tw  = zeros(Float64,o,o,v,v); @einsum _tw[j,n,f,b]  = ts[j,f] * ts[n,b]
        _tw .= 0.5 .* td .+ _tw
        _w7  = zeros(Float64,o,v,v,o); @einsum _w7[m,b,e,j]  = _tw[j,n,f,b] * aso_oovv[m,n,e,f]
        Wmbej = zeros(Float64, o, v, v, o)
        @threads for idx in eachindex(Wmbej)
            x = idx - 1
            m = (x % o) + 1; x ÷= o
            b = (x % v) + 1; x ÷= v
            e = (x % v) + 1; x ÷= v
            j = (x % o) + 1
            @inbounds Wmbej[m,b,e,j] = aso_ovvo[m,b,e,j] + _w5[m,b,e,j] - _w6[m,b,e,j] - _w7[m,b,e,j]
        end

        # ── T1 update ─────────────────────────────────────────────────────────
        # tsnew[i,a] = [f_{ia} + ∑_e t_i^e F̃_{ae} − ∑_m t_m^a F̃_{mi}
        #               + ∑_{me} t_{im}^{ae} F̃_{me} − ∑_{nf} t_n^f <na||if>
        #               − ½∑_{mef} t_{im}^{ef} <ma||ef>
        #               − ½∑_{mne} t_{mn}^{ae} <nm||ei>] / D_{ia}
        _t11 = zeros(Float64,o,v); @einsum _t11[i,a] = ts[i,e] * Fae[a,e]
        _t12 = zeros(Float64,o,v); @einsum _t12[i,a] = ts[m,a] * Fmi[m,i]
        _t13 = zeros(Float64,o,v); @einsum _t13[i,a] = td[i,m,a,e] * Fme[m,e]
        _t14 = zeros(Float64,o,v); @einsum _t14[i,a] = ts[n,f] * aso_ovov[n,a,i,f]
        _t15 = zeros(Float64,o,v); @einsum _t15[i,a] = td[i,m,e,f] * aso_ovvv[m,a,e,f]
        _t16 = zeros(Float64,o,v); @einsum _t16[i,a] = td[m,n,a,e] * aso_oovo[n,m,e,i]
        tsnew = (F_ov .+ _t11 .- _t12 .+ _t13 .- _t14 .- 0.5.*_t15 .- 0.5.*_t16) ./ Dia

        # ── T2 update ─────────────────────────────────────────────────────────
        # Modified F intermediates
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
        # P̂(ij) ∑_e t_i^e <ab||ej>
        _t2f = zeros(Float64,o,o,v,v); @einsum _t2f[i,j,a,b] = ts[i,e] * aso_vvvo[a,b,e,j]
        # −P̂(ab) ∑_m t_m^a <mb||ij>
        _t2g = zeros(Float64,o,o,v,v); @einsum _t2g[i,j,a,b] = ts[m,a] * aso_ovoo[m,b,i,j]

        tdnew = zeros(Float64, o, o, v, v)
        @threads for idx in eachindex(tdnew)
            x = idx - 1
            i = (x % o) + 1; x ÷= o
            j = (x % o) + 1; x ÷= o
            a = (x % v) + 1; x ÷= v
            b = (x % v) + 1
            @inbounds begin
                num = ( aso_oovv[i,j,a,b]
                      + _t2a[i,j,a,b] - _t2a[i,j,b,a]          # P̂(ab)
                      - _t2b[i,j,a,b] + _t2b[j,i,a,b]          # −P̂(ij)
                      + 0.5*_t2c[i,j,a,b] + 0.5*_t2d[i,j,a,b]
                      + (_t2e[i,j,a,b]-_t2ets[i,j,a,b])         # P̂(ij)P̂(ab)
                      - (_t2e[j,i,a,b]-_t2ets[j,i,a,b])
                      - (_t2e[i,j,b,a]-_t2ets[i,j,b,a])
                      + (_t2e[j,i,b,a]-_t2ets[j,i,b,a])
                      + _t2f[i,j,a,b] - _t2f[j,i,a,b]           # P̂(ij)
                      - _t2g[i,j,a,b] + _t2g[i,j,b,a] )         # −P̂(ab)
                tdnew[i,j,a,b] = num / (Dia[i,a] + Dia[j,b])
            end
        end

        # ── Project #10 DIIS extrapolation in CC amplitude space ─────────────
        tsnext = tsnew
        tdnext = tdnew
        if diis
            push!(error_history, cc_amplitude_error(tsnew, tdnew, ts, td))
            push!(amplitude_history, (copy(tsnew), copy(tdnew)))

            while length(error_history) > diis_size
                popfirst!(error_history)
                popfirst!(amplitude_history)
            end

            if iteration >= diis_start && length(error_history) >= 2
                coefficients = cc_diis_coefficients(error_history)
                tsnext, tdnext = extrapolate_amplitudes(amplitude_history, coefficients)
            end
        end

        # ── CCSD correlation energy ───────────────────────────────────────────
        E_new = ccsd_energy(aso_oovv, F_ov, tsnext, tdnext)

        delta_E = abs(E_new - E_ccsd)
        delta_T = max(sqrt(sum((tsnext .- ts).^2)),
                      sqrt(sum((tdnext .- td).^2)))
        verbose && @printf("  %6d  %20.12f  %12.4e  %12.4e\n",
                           iteration, E_new, delta_E, delta_T)

        ts = tsnext;  td = tdnext;  E_ccsd = E_new

        if delta_E < e_tol && delta_T < amp_tol
            converged = true
            iterations = iteration
            if verbose
                println("  " * "─"^64)
                @printf("  Converged in %d iterations.\n", iteration)
            end
            break
        end
        iteration == maxiter && @warn "CCSD did not converge in $maxiter iterations."
    end

    return (E_ccsd=E_ccsd, ts=ts, td=td, converged=converged, iterations=iterations)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_ccsd(rhf, mp2_result; maxiter, e_tol, amp_tol, diis, diis_size,
             diis_start, verbose) -> NamedTuple

Build spin-orbital integrals from `rhf` and `mp2_result`, then run CCSD.

Arguments:
- `rhf`        : NamedTuple from `run_rhf`
- `mp2_result` : NamedTuple from `run_mp2`

Returns:
- `E_ccsd`        : CCSD correlation energy (Eh)
- `total_energy`  : E_HF + E_CCSD (Eh)
- `ts`, `td`      : converged T1 and T2 amplitudes
- `converged`     : Bool
- `iterations`    : number of CCSD iterations used
"""
function run_ccsd(rhf, mp2_result;
                  maxiter=500, e_tol=1e-10, amp_tol=1e-10,
                  diis=false, diis_size=8, diis_start=3, verbose=true)

    n_elec = rhf.n_elec
    o      = n_elec                   # occupied spin-orbitals
    v      = 2*rhf.nbasis - n_elec    # virtual  spin-orbitals

    aso  = mo_to_aso(mp2_result.new_eri)
    F_so = Mat_motoMat_so(Mat_aotoMat_mo(rhf.mo_coeffs, rhf.fock))

    result = ccsd_scf(aso, F_so, rhf.orbital_energies, o, v, n_elec;
                      maxiter, e_tol, amp_tol, diis, diis_size, diis_start, verbose)

    total = result.E_ccsd + rhf.total_energy
    if verbose
        @printf("\nCCSD correlation energy = %20.12f Eh\n", result.E_ccsd)
        @printf("CCSD total energy       = %20.12f Eh\n", total)
    end

    return merge(result, (total_energy=total,))
end
