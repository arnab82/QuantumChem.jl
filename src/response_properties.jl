"""
Excited-state response/property helpers.

Project #28 computes one-photon properties from CIS and the local EOM-CCSD
right eigenvectors.  The CIS transition moments are spin-adapted singlet
transition moments.  The EOM-CCSD transition moments use the one-body dipole
operator and the singles block of the selected right eigenvectors, which is an
educational approximation to full coupled-cluster response properties.
"""

using LinearAlgebra, Printf, PyCall

"""
    ao_dipole_integrals(rhf) -> Array{Float64,3}

Return AO position integrals from PySCF:

```text
r_x[mu,nu] = <chi_mu | x | chi_nu>
r_y[mu,nu] = <chi_mu | y | chi_nu>
r_z[mu,nu] = <chi_mu | z | chi_nu>.
```

The electronic dipole operator is `-r`; transition-property routines apply
that sign when forming dipole moments.
"""
function ao_dipole_integrals(rhf)
    mol = _require_rhf_molecule(rhf)
    return pyscf_intor(mol, "int1e_r", Val(3))
end

"""
    nuclear_dipole(rhf) -> Vector{Float64}

Return the nuclear contribution to the electric dipole moment in atomic units:

```text
mu_nuc = sum_A Z_A R_A.
```
"""
function nuclear_dipole(rhf)
    mol = _require_rhf_molecule(rhf)
    charges_obj = pycall(mol.atom_charges, PyObject)
    coords_obj = pycall(mol.atom_coords, PyObject)
    charges_py = PyArray(charges_obj)
    coords_py = PyArray(coords_obj)
    charges = Array{Float64,1}(Array(charges_py))
    coords = Array{Float64,2}(Array(coords_py))
    _keep_pyscf!(charges_obj, coords_obj, charges_py, coords_py)
    return vec(sum(coords .* reshape(charges, :, 1); dims=1))
end

"""
    mo_dipole_integrals(rhf; ao_dipoles=ao_dipole_integrals(rhf)) -> Array{Float64,3}

Transform AO position integrals to the spatial MO basis:

```text
r_alpha[p,q] = sum_munu C_mup r_alpha[mu,nu] C_nuq.
```
"""
function mo_dipole_integrals(rhf; ao_dipoles=ao_dipole_integrals(rhf))
    nbasis = rhf.nbasis
    size(ao_dipoles) == (3, nbasis, nbasis) ||
        throw(DimensionMismatch("AO dipole integrals must have shape (3, nbasis, nbasis)"))
    mo_dipoles = zeros(Float64, 3, nbasis, nbasis)
    for axis in 1:3
        mo_dipoles[axis, :, :] .= Mat_aotoMat_mo(rhf.mo_coeffs, view(ao_dipoles, axis, :, :))
    end
    return mo_dipoles
end

"""
    rhf_dipole_moment(rhf; ao_dipoles=ao_dipole_integrals(rhf)) -> Vector{Float64}

Compute the RHF permanent dipole moment:

```text
mu = sum_A Z_A R_A - 2 sum_munu D_munu <chi_mu|r|chi_nu>,
D_munu = sum_i^occ C_mui C_nui.
```
"""
function rhf_dipole_moment(rhf; ao_dipoles=ao_dipole_integrals(rhf))
    D = make_density(rhf.mo_coeffs, rhf.n_elec ÷ 2)
    electronic = zeros(Float64, 3)
    for axis in 1:3
        electronic[axis] = -2.0 * dot(D, view(ao_dipoles, axis, :, :))
    end
    return nuclear_dipole(rhf) .+ electronic
end

"""
    oscillator_strengths(energies, transition_dipoles) -> Vector{Float64}

Compute length-gauge electric-dipole oscillator strengths:

```text
f_k = (2/3) omega_k |mu_0k|^2.
```

`energies[k]` is the excitation energy in Hartree and
`transition_dipoles[k, :]` is the transition dipole vector in atomic units.
"""
function oscillator_strengths(energies, transition_dipoles)
    length(energies) == size(transition_dipoles, 1) ||
        throw(DimensionMismatch("Energy count must match transition-dipole rows"))
    strengths = zeros(Float64, length(energies))
    for root in eachindex(energies)
        strengths[root] = (2.0 / 3.0) * energies[root] *
                          sum(abs2, view(transition_dipoles, root, :))
    end
    return strengths
end

function _as_column_matrix(vectors)
    ndims(vectors) == 1 && return reshape(vectors, :, 1)
    ndims(vectors) == 2 && return vectors
    throw(DimensionMismatch("Excited-state vectors must be a vector or matrix"))
end

"""
    cis_transition_dipoles(rhf, singlet_vectors; mo_dipoles=mo_dipole_integrals(rhf))

Compute spin-adapted CIS transition dipoles from the RHF ground state:

```text
mu_0k = -sqrt(2) sum_ia C_ia^k <i|r|a>.
```

The CIS vector layout must match `build_cis_spin_adapted_matrix`: occupied
orbital index first, then virtual orbital index.
"""
function cis_transition_dipoles(rhf, singlet_vectors; mo_dipoles=mo_dipole_integrals(rhf))
    vectors = _as_column_matrix(singlet_vectors)
    nocc = rhf.n_elec ÷ 2
    nbasis = rhf.nbasis
    nvirt = nbasis - nocc
    size(vectors, 1) == nocc * nvirt ||
        throw(DimensionMismatch("CIS vector dimension does not match occupied-virtual space"))
    size(mo_dipoles) == (3, nbasis, nbasis) ||
        throw(DimensionMismatch("MO dipole integrals must have shape (3, nbasis, nbasis)"))

    transition = zeros(Float64, size(vectors, 2), 3)
    factor = -sqrt(2.0)
    for root in axes(vectors, 2), axis in 1:3, i in 1:nocc, a0 in 1:nvirt
        a = nocc + a0
        idx = excitation_index(i, a0, nvirt)
        @inbounds transition[root, axis] += factor * vectors[idx, root] * mo_dipoles[axis, i, a]
    end
    return transition
end

"""
    run_cis_properties(rhf, excited; nroots=length(excited.singlet_energies), verbose=true)

Compute CIS singlet transition dipoles and oscillator strengths from an
`run_excited_states` result.
"""
function run_cis_properties(rhf, excited; nroots=length(excited.singlet_energies), verbose=true)
    1 <= nroots <= length(excited.singlet_energies) ||
        throw(ArgumentError("nroots must be between 1 and the number of CIS singlet roots"))

    vectors =
        if hasproperty(excited, :singlet_vectors)
            excited.singlet_vectors[:, 1:nroots]
        else
            eigen(Symmetric(excited.singlet_matrix)).vectors[:, 1:nroots]
        end
    energies = excited.singlet_energies[1:nroots]
    ao_dipoles = ao_dipole_integrals(rhf)
    mo_dipoles = mo_dipole_integrals(rhf; ao_dipoles)
    transition = cis_transition_dipoles(rhf, vectors; mo_dipoles)
    strengths = oscillator_strengths(energies, transition)
    ground_dipole = rhf_dipole_moment(rhf; ao_dipoles)

    if verbose
        @printf("CIS transition properties:\n")
        for root in eachindex(energies)
            @printf("  %3d  omega=%12.8f Eh  f=%12.8f  mu=(% .6f,% .6f,% .6f)\n",
                    root, energies[root], strengths[root],
                    transition[root, 1], transition[root, 2], transition[root, 3])
        end
    end

    return (
        method=:CIS,
        energies=energies,
        energies_ev=energies .* HARTREE_TO_EV,
        transition_dipoles=transition,
        oscillator_strengths=strengths,
        ground_dipole=ground_dipole,
        mo_dipoles=mo_dipoles,
        singlet_vectors=vectors,
    )
end

"""
    eom_ccsd_transition_dipoles(rhf, eom; mo_dipoles=mo_dipole_integrals(rhf))

Compute one-body EOM-CCSD right-vector transition dipoles:

```text
mu_0k approx -sum_ia R_i^a(k) <i|r|a>.
```

Only the singles block of each right eigenvector contributes.  Full
coupled-cluster response properties also require left eigenvectors and
similarity-transformed property operators; this helper is intentionally the
compact educational analogue used by this project.
"""
function eom_ccsd_transition_dipoles(rhf, eom; mo_dipoles=mo_dipole_integrals(rhf))
    hasproperty(eom, :vectors) ||
        throw(ArgumentError("EOM-CCSD result must contain right eigenvectors; rerun run_eom_ccsd"))
    vectors = _as_column_matrix(eom.vectors)
    o = rhf.n_elec
    v = 2 * rhf.nbasis - o
    size(vectors, 1) == eom_dimension(o, v) ||
        throw(DimensionMismatch("EOM vector dimension does not match RHF occupied/virtual space"))

    nbasis = rhf.nbasis
    size(mo_dipoles) == (3, nbasis, nbasis) ||
        throw(DimensionMismatch("MO dipole integrals must have shape (3, nbasis, nbasis)"))
    so_dipoles = zeros(Float64, 3, 2 * nbasis, 2 * nbasis)
    for axis in 1:3
        so_dipoles[axis, :, :] .= spatial_to_spinorbital_1e(view(mo_dipoles, axis, :, :))
    end

    transition = zeros(Float64, size(vectors, 2), 3)
    for root in axes(vectors, 2)
        rs, _ = unpack_eom_amplitudes(view(vectors, :, root), o, v)
        for axis in 1:3, i in 1:o, a0 in 1:v
            a = o + a0
            @inbounds transition[root, axis] -= rs[i, a0] * so_dipoles[axis, i, a]
        end
    end
    return transition
end

"""
    run_eom_ccsd_properties(rhf, eom; nroots=length(eom.energies), verbose=true)

Compute EOM-CCSD right-vector transition dipoles and oscillator strengths from
an `run_eom_ccsd` result.
"""
function run_eom_ccsd_properties(rhf, eom; nroots=length(eom.energies), verbose=true)
    1 <= nroots <= length(eom.energies) ||
        throw(ArgumentError("nroots must be between 1 and the number of EOM roots"))
    hasproperty(eom, :vectors) ||
        throw(ArgumentError("EOM-CCSD result must contain right eigenvectors; rerun run_eom_ccsd"))

    energies = eom.energies[1:nroots]
    ao_dipoles = ao_dipole_integrals(rhf)
    mo_dipoles = mo_dipole_integrals(rhf; ao_dipoles)
    sliced = merge(eom, (vectors=eom.vectors[:, 1:nroots],))
    transition = eom_ccsd_transition_dipoles(rhf, sliced; mo_dipoles)
    strengths = oscillator_strengths(energies, transition)
    ground_dipole = rhf_dipole_moment(rhf; ao_dipoles)

    if verbose
        @printf("EOM-CCSD transition properties:\n")
        for root in eachindex(energies)
            @printf("  %3d  omega=%12.8f Eh  f=%12.8f  mu=(% .6f,% .6f,% .6f)\n",
                    root, energies[root], strengths[root],
                    transition[root, 1], transition[root, 2], transition[root, 3])
        end
    end

    return (
        method=:EOM_CCSD,
        energies=energies,
        energies_ev=energies .* HARTREE_TO_EV,
        transition_dipoles=transition,
        oscillator_strengths=strengths,
        ground_dipole=ground_dipole,
        mo_dipoles=mo_dipoles,
        vectors=eom.vectors[:, 1:nroots],
    )
end
