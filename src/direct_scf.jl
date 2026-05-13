"""
Integral-direct RHF Fock builds.

Project #26 avoids materializing the four-index AO ERI tensor during SCF.  The
current AO density is sent to PySCF's direct J/K builder, and the closed-shell
Fock matrix is assembled in Julia as `F = h + 2J - K`.
"""

using LinearAlgebra, PyCall

function _pyscf_direct_jk(mol::PyObject, D)
    hf = pyimport("pyscf.scf.hf")
    operator = pyimport("operator")
    get_jk = hf.get_jk
    jk = pycall(get_jk, PyObject, mol, D)
    vj_obj = pycall(operator.getitem, PyObject, jk, 0)
    vk_obj = pycall(operator.getitem, PyObject, jk, 1)
    vj_array = PyArray(vj_obj)
    vk_array = PyArray(vk_obj)
    _keep_pyscf!(hf, operator, get_jk, jk, vj_obj, vk_obj, vj_array, vk_array)
    return (
        J = Array{Float64,2}(Array(vj_array)),
        K = Array{Float64,2}(Array(vk_array)),
    )
end

"""
    direct_jk(mol, D)

Return Coulomb and exchange matrices for AO density `D` using PySCF's direct
J/K builder.  `D` follows this package's closed-shell convention
`D = C_occ C_occ'`, without the factor of two.

```text
J_mn[D] = sum_ls D_ls (mn|ls)
K_mn[D] = sum_ls D_ls (ml|ns).
```
"""
direct_jk(mol::PyObject, D) = _pyscf_direct_jk(mol, D)

"""
    make_fock_direct(D, h1e, mol) -> Matrix{Float64}

Build the closed-shell AO Fock matrix without storing AO ERIs:

```text
F = h + 2J[D] - K[D]
```
"""
function make_fock_direct(D, h1e, mol::PyObject)
    nbasis = size(D, 1)
    size(D) == (nbasis, nbasis) || throw(DimensionMismatch("Density matrix must be square"))
    size(h1e) == (nbasis, nbasis) || throw(DimensionMismatch("Core Hamiltonian size does not match density"))
    jk = direct_jk(mol, D)
    fock = h1e .+ 2.0 .* jk.J .- jk.K
    return (fock .+ fock') ./ 2
end

"""
    run_direct_rhf(; kwargs...)

Convenience wrapper for `run_rhf(direct=true; kwargs...)`.
"""
run_direct_rhf(; kwargs...) = run_rhf(; direct=true, kwargs...)
