"""
AO→MO integral transformation and MP2 correlation energy.

Exported entry point: `run_mp2(rhf_result)`
"""

using Printf, TensorOperations
using Base.Threads: @threads, maxthreadid, threadid

# ── AO → MO transformation ─────────────────────────────────────────────────────

"""
    transform_eri(c, eri) -> Array{Float64,4}

Transform AO electron-repulsion integrals to the MO basis via four successive
half-transformations, each contracting one AO index with the MO coefficients:

    (pq|rs) = ∑_{μνλσ} C_{μp} C_{νq} (μν|λσ) C_{λr} C_{σs}
"""
function transform_eri(c, eri)
    @tensoropt mo[p,q,r,s] := c[i,p] * c[j,q] * eri[i,j,k,l] * c[k,r] * c[l,s]
    return mo
end

# ── MP2 correlation energy ─────────────────────────────────────────────────────

"""
    compute_mp2(new_eri, E, ndocc, nbasis) -> Float64

Evaluate the MP2 correlation energy in the spatial MO basis:

    E_MP2 = ∑_{i≤ndocc, a>ndocc} (ia|jb)(2(ia|jb) − (ib|ja)) / (εi+εj−εa−εb)
"""
function compute_mp2(new_eri, E, ndocc, nbasis)
    nvirt = nbasis - ndocc
    partial = zeros(Float64, maxthreadid())
    @threads for idx in 1:(ndocc * ndocc * nvirt * nvirt)
        x = idx - 1
        b = ndocc + (x % nvirt) + 1; x ÷= nvirt
        a = ndocc + (x % nvirt) + 1; x ÷= nvirt
        j = (x % ndocc) + 1; x ÷= ndocc
        i = (x % ndocc) + 1
        @inbounds begin
            iajb = new_eri[i,a,j,b]
            denom = E[i] + E[j] - E[a] - E[b]
            partial[threadid()] += iajb * (2*iajb - new_eri[i,b,j,a]) / denom
        end
    end
    return sum(partial)
end

# ── Public entry point ─────────────────────────────────────────────────────────

"""
    run_mp2(rhf; verbose) -> NamedTuple

Transform ERIs to the MO basis and compute the MP2 correlation energy.

`rhf` should be the NamedTuple returned by `run_rhf`.

Returns:
- `emp2`    : MP2 correlation energy (Eh)
- `new_eri` : MO-basis two-electron integrals (nbasis^4 array)
"""
function run_mp2(rhf; verbose=true)
    new_eri = transform_eri(rhf.mo_coeffs, rhf.eri)
    ndocc   = rhf.n_elec ÷ 2
    emp2    = compute_mp2(new_eri, rhf.orbital_energies, ndocc, rhf.nbasis)

    if verbose
        @printf("MP2 correlation energy = %20.12f Eh\n", emp2)
        @printf("MP2 total energy       = %20.12f Eh\n", emp2 + rhf.total_energy)
    end
    return (emp2=emp2, new_eri=new_eri)
end
