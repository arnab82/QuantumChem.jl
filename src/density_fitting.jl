"""
Density fitting / resolution-of-the-identity helpers.

Project #27 builds Coulomb-metric orthonormal three-index AO factors from PySCF
auxiliary-basis integrals and uses them for RI-MP2.  The MP2 driver avoids
forming a full four-index MO ERI tensor.
"""

using LinearAlgebra, Printf, PyCall, TensorOperations
using Base.Threads: @threads, maxthreadid, threadid

"""
    DensityFitInfo

Container for AO-basis density-fitting data:

- `auxbasis`: auxiliary basis name passed to PySCF.
- `naux`: number of auxiliary functions.
- `three_center`: raw AO three-center integrals `(mu nu | P)`.
- `metric`: auxiliary Coulomb metric `(P|Q)`.
- `metric_inv_sqrt`: inverse square root `(P|Q)^(-1/2)`.
- `ao_factors`: orthonormalized factors `B_munu^P`.
- `metric_eigenvalues`: eigenvalues of the auxiliary Coulomb metric.

The factors satisfy the RI approximation

```text
(mu nu | lambda sigma) approx sum_P B_munu^P B_lambdasigma^P.
```
"""
struct DensityFitInfo
    auxbasis::String
    naux::Int
    three_center::Array{Float64,3}
    metric::Matrix{Float64}
    metric_inv_sqrt::Matrix{Float64}
    ao_factors::Array{Float64,3}
    metric_eigenvalues::Vector{Float64}
end

function _pyscf_auxmol(mol::PyObject, auxbasis::AbstractString)
    df = pyimport("pyscf.df")
    make_auxmol = df.addons.make_auxmol
    auxmol = pycall(make_auxmol, PyObject, mol, String(auxbasis))
    _keep_pyscf!(df, make_auxmol, auxmol)
    return df, auxmol
end

"""
    df_metric_inv_sqrt(metric; cutoff=1e-10) -> (matrix, eigenvalues)

Return the symmetric inverse square root of the positive definite auxiliary
Coulomb metric:

```text
J = U diag(j) U'
J^(-1/2) = U diag(j^(-1/2)) U'.
```

Eigenvalues below `cutoff` are treated as linear dependencies.
"""
function df_metric_inv_sqrt(metric; cutoff=1e-10)
    size(metric, 1) == size(metric, 2) ||
        throw(DimensionMismatch("Density-fitting metric must be square"))
    cutoff > 0.0 || throw(ArgumentError("metric cutoff must be positive"))

    fact = eigen(Symmetric((metric .+ metric') ./ 2))
    min_value = minimum(fact.values)
    min_value > cutoff ||
        throw(ArgumentError("Auxiliary Coulomb metric has eigenvalue $min_value below cutoff $cutoff"))
    inv_sqrt = fact.vectors * Diagonal(fact.values .^ (-0.5)) * fact.vectors'
    return (matrix=inv_sqrt, eigenvalues=copy(fact.values))
end

"""
    df_ao_factors(rhf; auxbasis="weigend", metric_cutoff=1e-10) -> DensityFitInfo

Build orthonormal AO density-fitting factors from PySCF three-center integrals.
The raw factors are

```text
B_munu^P = sum_Q (mu nu | Q) (Q|P)^(-1/2).
```

`rhf` must be the result returned by `run_rhf`, so the PySCF molecule object is
available.
"""
function df_ao_factors(rhf; auxbasis="weigend", metric_cutoff=1e-10)
    mol = _require_rhf_molecule(rhf)
    df, auxmol = _pyscf_auxmol(mol, String(auxbasis))

    aux_e2 = df.incore.aux_e2
    three_obj = pycall(aux_e2, PyObject, mol, auxmol; intor="int3c2e", aosym="s1")
    three_py = PyArray(three_obj)
    three_center = Array{Float64,3}(Array(three_py))

    intor = auxmol.intor
    metric_obj = pycall(intor, PyObject, "int2c2e")
    metric_py = PyArray(metric_obj)
    metric = Array{Float64,2}(Array(metric_py))

    inv = df_metric_inv_sqrt(metric; cutoff=metric_cutoff)
    @tensoropt ao_factors[mu,nu,P] := three_center[mu,nu,Q] * inv.matrix[Q,P]

    _keep_pyscf!(df, auxmol, aux_e2, three_obj, three_py, intor, metric_obj, metric_py)
    return DensityFitInfo(
        String(auxbasis),
        size(metric, 1),
        three_center,
        metric,
        inv.matrix,
        ao_factors,
        inv.eigenvalues,
    )
end

"""
    df_mo_factors(c, ao_factors) -> Array{Float64,3}

Transform AO density-fitting factors to the MO basis:

```text
B_pq^P = sum_munu C_mup C_nuq B_munu^P.
```
"""
function df_mo_factors(c, ao_factors)
    nbasis = size(c, 1)
    size(ao_factors, 1) == nbasis && size(ao_factors, 2) == nbasis ||
        throw(DimensionMismatch("AO density-fitting factors do not match MO coefficients"))
    @tensoropt mo_factors[p,q,P] := c[mu,p] * c[nu,q] * ao_factors[mu,nu,P]
    return mo_factors
end

"""
    df_eri_from_factors(mo_factors) -> Array{Float64,4}

Reconstruct approximate MO ERIs from density-fitting factors:

```text
(pq|rs)_DF = sum_P B_pq^P B_rs^P.
```

This helper is useful for validation and small systems; `compute_df_mp2` uses
the factors directly.
"""
function df_eri_from_factors(mo_factors)
    @tensoropt eri[p,q,r,s] := mo_factors[p,q,P] * mo_factors[r,s,P]
    return eri
end

"""
    compute_df_mp2(mo_factors, orbital_energies, ndocc, nbasis) -> Float64

Evaluate the RI-MP2 correlation energy without building four-index MO ERIs:

```text
(ia|jb)_DF = sum_P B_ia^P B_jb^P
E_RI-MP2 = sum_ijab (ia|jb)_DF [2(ia|jb)_DF - (ib|ja)_DF]
           / (eps_i + eps_j - eps_a - eps_b).
```
"""
function compute_df_mp2(mo_factors, orbital_energies, ndocc, nbasis)
    size(mo_factors, 1) == nbasis && size(mo_factors, 2) == nbasis ||
        throw(DimensionMismatch("MO density-fitting factors do not match basis size"))
    nvirt = nbasis - ndocc
    partial = zeros(Float64, maxthreadid())

    @threads for idx in 1:(ndocc * ndocc * nvirt * nvirt)
        x = idx - 1
        b = ndocc + (x % nvirt) + 1; x ÷= nvirt
        a = ndocc + (x % nvirt) + 1; x ÷= nvirt
        j = (x % ndocc) + 1; x ÷= ndocc
        i = (x % ndocc) + 1

        @inbounds begin
            iajb = dot(view(mo_factors, i, a, :), view(mo_factors, j, b, :))
            ibja = dot(view(mo_factors, i, b, :), view(mo_factors, j, a, :))
            denom = orbital_energies[i] + orbital_energies[j] -
                    orbital_energies[a] - orbital_energies[b]
            partial[threadid()] += iajb * (2.0 * iajb - ibja) / denom
        end
    end
    return sum(partial)
end

"""
    run_df_mp2(rhf; auxbasis="weigend", metric_cutoff=1e-10,
               return_eri=false, verbose=true) -> NamedTuple

Run density-fitted MP2 from an RHF reference.  The returned `emp2` is the
RI-MP2 correlation energy and `total_energy` is `E_RHF + E_RI-MP2`.

Set `return_eri=true` to materialize the approximate four-index MO ERI tensor
for validation or for small educational examples.
"""
function run_df_mp2(rhf; auxbasis="weigend", metric_cutoff=1e-10,
                    return_eri=false, verbose=true)
    info = df_ao_factors(rhf; auxbasis=auxbasis, metric_cutoff=metric_cutoff)
    mo_factors = df_mo_factors(rhf.mo_coeffs, info.ao_factors)
    ndocc = rhf.n_elec ÷ 2
    emp2 = compute_df_mp2(mo_factors, rhf.orbital_energies, ndocc, rhf.nbasis)
    total = rhf.total_energy + emp2
    new_eri = return_eri ? df_eri_from_factors(mo_factors) : nothing

    if verbose
        @printf("RI-MP2 aux basis          = %s (%d functions)\n", info.auxbasis, info.naux)
        @printf("RI-MP2 correlation energy = %20.12f Eh\n", emp2)
        @printf("RI-MP2 total energy       = %20.12f Eh\n", total)
    end

    return (
        emp2 = emp2,
        total_energy = total,
        mo_factors = mo_factors,
        new_eri = new_eri,
        df_info = info,
        auxbasis = info.auxbasis,
        naux = info.naux,
        metric_condition = maximum(info.metric_eigenvalues) / minimum(info.metric_eigenvalues),
    )
end
