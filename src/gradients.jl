"""
RHF analytic nuclear gradients.

Project #24 evaluates closed-shell Hartree-Fock forces in the AO basis.  PySCF
provides derivative one- and two-electron AO integrals, and this file assembles
the density, energy-weighted density, electronic gradient, and nuclear
repulsion gradient in Julia.
"""

using LinearAlgebra, Printf, PyCall, Statistics

function _pyarray_call(method, ::Type{T}, ::Val{N}, args...; kwargs...) where {T,N}
    values = pycall(method, PyObject, args...; kwargs...)
    pyarray = PyArray(values)
    _keep_pyscf!(method, values, pyarray)
    return Array{T,N}(Array(pyarray))
end

function _require_rhf_molecule(rhf)
    :mol in propertynames(rhf) ||
        throw(ArgumentError("RHF gradient needs an RHF result from run_rhf so the PySCF molecule is available"))
    return rhf.mol
end

function _gradient_atom_list(mol; atmlst=nothing)
    natoms = pyscf_getattr(mol, "natm", Int)
    if atmlst === nothing
        return collect(1:natoms)
    end
    atoms = [Int(atom) for atom in atmlst]
    all(atom -> 1 <= atom <= natoms, atoms) ||
        throw(ArgumentError("atmlst uses Julia atom indices and must be within 1:$natoms"))
    return atoms
end

"""
    rhf_spin_density(rhf) -> Matrix{Float64}

Return the closed-shell spin-summed AO density matrix `P = 2 C_occ C_occ'`.
"""
function rhf_spin_density(rhf)
    nocc = rhf.n_elec ÷ 2
    c_occ = @view rhf.mo_coeffs[:, 1:nocc]
    return 2.0 .* (c_occ * c_occ')
end

"""
    rhf_energy_weighted_density(rhf) -> Matrix{Float64}

Return the closed-shell energy-weighted AO density matrix
`W = C_occ diag(2 epsilon_i) C_occ'` used in the Pulay overlap term.
"""
function rhf_energy_weighted_density(rhf)
    nocc = rhf.n_elec ÷ 2
    c_occ = @view rhf.mo_coeffs[:, 1:nocc]
    weighted = c_occ .* reshape(2.0 .* rhf.orbital_energies[1:nocc], 1, :)
    return weighted * c_occ'
end

function _pyscf_rhf_object(rhf)
    mol = _require_rhf_molecule(rhf)
    pyscf = pyimport("pyscf")
    mf = pyscf.scf.RHF(mol)
    nocc = rhf.n_elec ÷ 2
    mo_occ = zeros(Float64, rhf.nbasis)
    mo_occ[1:nocc] .= 2.0
    mf.mo_coeff = rhf.mo_coeffs
    mf.mo_energy = rhf.orbital_energies
    mf.mo_occ = mo_occ
    mf.converged = true
    mf.verbose = 0
    _keep_pyscf!(pyscf, mf)
    return mf
end

"""
    nuclear_repulsion_gradient(rhf; atmlst=nothing)
    nuclear_repulsion_gradient(mol::PyObject; atmlst=nothing)

Evaluate the derivative of the nuclear repulsion energy in Eh/Bohr.
`atmlst` uses Julia's 1-based atom indices.

```text
dE_nuc/dR_A = - sum_{B != A} Z_A Z_B (R_A - R_B) / |R_A - R_B|^3.
```
"""
function nuclear_repulsion_gradient(mol::PyObject; atmlst=nothing)
    atoms = _gradient_atom_list(mol; atmlst)
    coords = _pyarray_call(mol.atom_coords, Float64, Val(2))
    charges = _pyarray_call(mol.atom_charges, Float64, Val(1))
    gradient = zeros(Float64, length(atoms), 3)

    @inbounds for (row, atom) in pairs(atoms)
        za = charges[atom]
        for other in eachindex(charges)
            atom == other && continue
            displacement = coords[atom, :] .- coords[other, :]
            distance = norm(displacement)
            gradient[row, :] .-= za * charges[other] .* displacement ./ distance^3
        end
    end
    return gradient
end

nuclear_repulsion_gradient(rhf; atmlst=nothing) =
    nuclear_repulsion_gradient(_require_rhf_molecule(rhf); atmlst)

"""
    rhf_electronic_gradient(rhf; atmlst=nothing)

Assemble the closed-shell RHF electronic gradient in Eh/Bohr from AO derivative
integrals, the spin-summed density, and the energy-weighted density:

```text
dE_elec/dR_A = sum_mn P_mn h_mn^A
             + sum_mn P_mn V_mn^A[P]
             - 2 sum_mn W_mn S_mn^A.
```

The two-electron derivative contribution `V^A[P]` is obtained from PySCF's
gradient J/K builder and the AO slices select the basis functions centered on
atom `A`.
"""
function rhf_electronic_gradient(rhf; atmlst=nothing)
    mol = _require_rhf_molecule(rhf)
    atoms = _gradient_atom_list(mol; atmlst)
    mf = _pyscf_rhf_object(rhf)
    gradient_driver = mf.nuc_grad_method()
    gradient_driver.verbose = 0

    hcore_generator = gradient_driver.hcore_generator
    hcore_deriv = pycall(hcore_generator, PyObject, mol)
    s1 = _pyarray_call(gradient_driver.get_ovlp, Float64, Val(3), mol)
    density = rhf_spin_density(rhf)
    vhf = _pyarray_call(gradient_driver.get_veff, Float64, Val(3), mol, density)
    weighted_density = rhf_energy_weighted_density(rhf)
    aoslices = _pyarray_call(mol.aoslice_by_atom, Int, Val(2))
    gradient = zeros(Float64, length(atoms), 3)

    @inbounds for (row, atom) in pairs(atoms)
        h1ao = _pyarray_call(hcore_deriv, Float64, Val(3), atom - 1)
        p0 = aoslices[atom, 3] + 1
        p1 = aoslices[atom, 4]

        for xyz in 1:3
            gradient[row, xyz] += sum(h1ao[xyz, :, :] .* density)
            gradient[row, xyz] += 2.0 * sum(vhf[xyz, p0:p1, :] .* density[p0:p1, :])
            gradient[row, xyz] -= 2.0 * sum(s1[xyz, p0:p1, :] .* weighted_density[p0:p1, :])
        end
    end

    _keep_pyscf!(mf, gradient_driver, hcore_generator, hcore_deriv)
    return gradient
end

"""
    rhf_gradient(rhf; atmlst=nothing)

Return the total closed-shell RHF analytic nuclear gradient in Eh/Bohr:

```text
grad E_RHF = grad E_elec + grad E_nuc.
```
"""
function rhf_gradient(rhf; atmlst=nothing)
    return rhf_electronic_gradient(rhf; atmlst) .+
           nuclear_repulsion_gradient(rhf; atmlst)
end

"""
    run_rhf_gradient([rhf]; atmlst=nothing, verbose=true, rhf_verbose=false, kwargs...)

Run or reuse an RHF calculation and return analytic nuclear gradients.  If `rhf`
is omitted, remaining keyword arguments are passed to `run_rhf`.
"""
function run_rhf_gradient(rhf=nothing; atmlst=nothing, verbose=true,
                          rhf_verbose=false, kwargs...)
    if rhf === nothing
        rhf = run_rhf(; verbose=rhf_verbose, kwargs...)
    elseif !isempty(kwargs)
        throw(ArgumentError("When an RHF result is supplied, run_rhf_gradient does not accept run_rhf keywords"))
    end

    electronic = rhf_electronic_gradient(rhf; atmlst)
    nuclear = nuclear_repulsion_gradient(rhf; atmlst)
    total = electronic .+ nuclear
    max_force = maximum(abs.(total))
    rms_force = sqrt(mean(abs2, total))

    verbose && begin
        @printf("RHF analytic gradient (Eh/Bohr)\n")
        for atom in 1:size(total, 1)
            @printf("  atom %3d  %16.10f %16.10f %16.10f\n",
                    atom, total[atom, 1], total[atom, 2], total[atom, 3])
        end
        @printf("RHF max |gradient| = %.6e Eh/Bohr\n", max_force)
        @printf("RHF RMS gradient   = %.6e Eh/Bohr\n", rms_force)
    end

    return (
        gradient = total,
        electronic_gradient = electronic,
        nuclear_gradient = nuclear,
        max_force = max_force,
        rms_force = rms_force,
        rhf = rhf,
        atmlst = atmlst === nothing ? nothing : collect(atmlst),
    )
end
