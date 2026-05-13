"""
Perturbative triples correction to CCSD: CCSD(T).

Implements Crawford Programming Project #6 in the same antisymmetrized
spin-orbital basis used by `ccsd.jl`.
"""

using Printf
using Base.Threads: @threads, maxthreadid, threadid

"""
    spin_orbital_denominator(F_so, i, j, k, a, b, c, o) -> Float64

Return D_ijk^abc = f_ii + f_jj + f_kk - f_aa - f_bb - f_cc.
Virtual indices `a`, `b`, and `c` are relative to the virtual block.
"""
function spin_orbital_denominator(F_so, i, j, k, a, b, c, o)
    return (F_so[i,i] + F_so[j,j] + F_so[k,k]
            - F_so[o+a,o+a] - F_so[o+b,o+b] - F_so[o+c,o+c])
end

@inline function connected_triples_raw(aso, td, ip, jp, kp, ap, bp, cp, o, v)
    value = 0.0
    @inbounds for e in 1:v
        value += td[jp,kp,ap,e] * aso[o+e,ip,o+bp,o+cp]
    end
    @inbounds for m in 1:o
        value -= td[ip,m,bp,cp] * aso[m,o+ap,jp,kp]
    end
    return value
end

@inline function disconnected_triples_raw(aso, ts, ip, jp, kp, ap, bp, cp, o)
    @inbounds return ts[ip,ap] * aso[jp,kp,o+bp,o+cp]
end

"""
    connected_triples_residual(aso, td, i, j, k, a, b, c, o, v) -> Float64

Return D_ijk^abc * t_ijk^abc(c), the connected triples numerator:

P(i/jk)P(a/bc) [ sum_e t_jk^ae <ei||bc> - sum_m t_im^bc <ma||jk> ].
"""
function connected_triples_residual(aso, td, i, j, k, a, b, c, o, v)
    occ_1 = (connected_triples_raw(aso, td, i, j, k, a, b, c, o, v)
             - connected_triples_raw(aso, td, j, i, k, a, b, c, o, v)
             - connected_triples_raw(aso, td, k, j, i, a, b, c, o, v))
    occ_2 = (connected_triples_raw(aso, td, i, j, k, b, a, c, o, v)
             - connected_triples_raw(aso, td, j, i, k, b, a, c, o, v)
             - connected_triples_raw(aso, td, k, j, i, b, a, c, o, v))
    occ_3 = (connected_triples_raw(aso, td, i, j, k, c, b, a, o, v)
             - connected_triples_raw(aso, td, j, i, k, c, b, a, o, v)
             - connected_triples_raw(aso, td, k, j, i, c, b, a, o, v))
    return occ_1 - occ_2 - occ_3
end

"""
    disconnected_triples_residual(aso, ts, i, j, k, a, b, c, o) -> Float64

Return D_ijk^abc * t_ijk^abc(d), the disconnected triples numerator:

P(i/jk)P(a/bc) t_i^a <jk||bc>.
"""
function disconnected_triples_residual(aso, ts, i, j, k, a, b, c, o)
    occ_1 = (disconnected_triples_raw(aso, ts, i, j, k, a, b, c, o)
             - disconnected_triples_raw(aso, ts, j, i, k, a, b, c, o)
             - disconnected_triples_raw(aso, ts, k, j, i, a, b, c, o))
    occ_2 = (disconnected_triples_raw(aso, ts, i, j, k, b, a, c, o)
             - disconnected_triples_raw(aso, ts, j, i, k, b, a, c, o)
             - disconnected_triples_raw(aso, ts, k, j, i, b, a, c, o))
    occ_3 = (disconnected_triples_raw(aso, ts, i, j, k, c, b, a, o)
             - disconnected_triples_raw(aso, ts, j, i, k, c, b, a, o)
             - disconnected_triples_raw(aso, ts, k, j, i, c, b, a, o))
    return occ_1 - occ_2 - occ_3
end

"""
    triples_correction(aso, F_so, ts, td, o, v) -> Float64

Compute the non-iterative perturbative triples correction:

E(T) = (1/36) sum_ijkabc t_ijk^abc(c) D_ijk^abc
       [t_ijk^abc(c) + t_ijk^abc(d)].
"""
function triples_correction(aso, F_so, ts, td, o, v)
    eps_occ = [F_so[i,i] for i in 1:o]
    eps_vir = [F_so[o+a,o+a] for a in 1:v]
    partial = zeros(Float64, maxthreadid())

    @threads for idx in 1:(o * o * o * v * v * v)
        x = idx - 1
        c = (x % v) + 1; x ÷= v
        b = (x % v) + 1; x ÷= v
        a = (x % v) + 1; x ÷= v
        k = (x % o) + 1; x ÷= o
        j = (x % o) + 1; x ÷= o
        i = (x % o) + 1

        @inbounds denom = eps_occ[i] + eps_occ[j] + eps_occ[k] - eps_vir[a] - eps_vir[b] - eps_vir[c]
        connected = connected_triples_residual(aso, td, i, j, k, a, b, c, o, v)
        disconnected = disconnected_triples_residual(aso, ts, i, j, k, a, b, c, o)
        partial[threadid()] += connected * (connected + disconnected) / denom
    end
    return sum(partial) / 36.0
end

"""
    run_ccsd_t(rhf, mp2_result, ccsd_result; verbose=true) -> NamedTuple

Build the spin-orbital tensors and evaluate the CCSD(T) energy correction.

Returns:
- `E_triples`    : perturbative triples correction E(T) in Eh
- `total_energy` : E_HF + E_CCSD + E(T) in Eh
"""
function run_ccsd_t(rhf, mp2_result, ccsd_result; verbose=true)
    o = rhf.n_elec
    v = 2 * rhf.nbasis - o

    aso = mo_to_aso(mp2_result.new_eri)
    F_so = Mat_motoMat_so(Mat_aotoMat_mo(rhf.mo_coeffs, rhf.fock))

    e_triples = triples_correction(aso, F_so, ccsd_result.ts, ccsd_result.td, o, v)
    total = ccsd_result.total_energy + e_triples

    if verbose
        @printf("\nCCSD(T) triples correction = %20.12f Eh\n", e_triples)
        @printf("CCSD(T) total energy       = %20.12f Eh\n", total)
    end

    return (E_triples=e_triples, total_energy=total)
end
