"""
    QuantumChem

Julia package implementing quantum chemistry programming
projects: molecular geometry, vibrations, RHF/ROHF/UHF/DIIS/symmetry/out-of-core
SCF/direct SCF, RHF gradients/geometry optimization, MP2/UMP2, CCSD/UCCSD/CCSDT, CCSD(T), CIS/CISD/HCI/FCI/DMRG, TDHF/RPA, Davidson-Liu CIS, and EOM-CCSD.
Uses PySCF for one- and two-electron integrals.

# Quick start

```julia
using QuantumChem

rhf    = run_rhf()                       # default: H₂O STO-3G
direct = run_direct_rhf()                # integral-direct RHF Fock builds
grad   = run_rhf_gradient(rhf)           # RHF analytic nuclear gradients
opt    = run_rhf_geometry_optimization(rhf; maxiter=5)
rohf   = run_rohf()                      # restricted open-shell SCF
uhf    = run_uhf()                       # unrestricted SCF
mp2    = run_mp2(rhf)
ump2   = run_ump2(uhf)
ccsd   = run_ccsd(rhf, mp2)
uccsd  = run_uccsd(uhf)
cisd   = run_cisd(rhf, mp2)
ccsdt_full = run_ccsdt(rhf, mp2)
hci    = run_hci(rhf, mp2)
fci    = run_fci(rhf, mp2)
dmrg   = run_dmrg(rhf, mp2)              # custom direct HF-MO-basis DMRG
ccsdt  = run_ccsd_t(rhf, mp2, ccsd)
```
"""
module QuantumChem

using Einsum
using LinearAlgebra
using Printf
using PyCall
using Statistics
using TensorOperations

include("geometry.jl")
include("vibrations.jl")
include("symmetry.jl")
include("rhf.jl")
include("gradients.jl")
include("optimization.jl")
include("outcore.jl")
include("direct_scf.jl")
include("uhf.jl")
include("rohf.jl")
include("mp2.jl")
include("ccsd.jl")
include("uccsd.jl")
include("ump2.jl")
include("ccsd_t.jl")
include("excited_states.jl")
include("cisd.jl")
include("selected_ci.jl")
include("fci.jl")
include("fci_dmrg.jl")
include("custom_dmrg.jl")
include("ccsdt.jl")
include("eom_ccsd.jl")

export Molecule, read_molecule, atomic_mass, atomic_masses
export distance, distance_matrix, bond_angle, out_of_plane_angle, torsion_angle
export center_of_mass, translate_to_center_of_mass
export inertia_tensor, principal_moments, rotor_type, rotational_constants
export moments_amu_angstrom2, moments_g_cm2, run_geometry_analysis
export read_hessian, mass_weight_hessian, harmonic_frequencies, run_vibrational_analysis
export diis_error_matrix, diis_coefficients, extrapolate_fock
export SymmetryInfo, water_c2v_sto3g_symmetry, pyscf_symmetry_info, pyscf_symmetry_orbitals
export infer_symmetry_occupations, normalize_symmetry_occupations, block_ranges, sopi, irrep_block_names
export symmetry_transform_1e, symmetry_backtransform_1e, symmetry_transform_eri
export symmetry_blocks, assemble_symmetry_blocks, symmetry_project, symmetry_offblock_norm
export block_multiply, block_s_half, block_eigen, make_density_symmetry, make_fock_symmetry
export ERIFile, compound_index, read_eri_file_header, write_unique_eri_file, make_fock_outcore
export direct_jk, make_fock_direct, run_direct_rhf
export run_rhf, run_rohf, run_uhf, run_mp2, run_ump2, run_ccsd, run_uccsd, run_ccsd_t, run_ccsdt, run_excited_states, run_davidson_cis, run_cisd, run_hci, run_fci, run_dmrg, run_fci_dmrg, run_eom_ccsd
export run_rhf_gradient, rhf_gradient, rhf_electronic_gradient
export rhf_spin_density, rhf_energy_weighted_density, nuclear_repulsion_gradient
export run_rhf_geometry_optimization, rhf_geometry_coordinates, rhf_atom_symbols
export format_atoms_bohr
export uhf_electron_counts, make_uhf_density, make_uhf_fock, uhf_energy, spin_square
export rohf_shell_counts, rohf_occupations, make_rohf_density, roothaan_fock
export compute_mp2, compute_ump2, transform_eri
export mo_to_aso, Mat_aotoMat_mo, Mat_motoMat_so, mp2_so, make_td_spinorbital
export uhf_spinorbital_order, uhf_spinorbital_coefficients
export antisymmetrize_spinorbital_eri, build_uccsd_spinorbital_inputs
export cc_amplitude_error, cc_diis_coefficients, extrapolate_amplitudes
export HARTREE_TO_EV, spinorbital_eri, excitation_pairs
export build_cis_spinorbital_matrix, build_cis_spin_adapted_matrix, cis_excitation_energies
export cis_singlet_diagonal, cis_singlet_sigma, davidson_liu
export build_rpa_b_matrix, build_rpa_matrix, rpa_full_energies, rpa_reduced_energies
export determinant_from_orbitals, occupied_orbitals, cisd_determinants
export spatial_to_spinorbital_1e, build_cisd_hamiltonian
export connected_determinants, hamiltonian_element, build_selected_ci_hamiltonian
export hci_pt2_correction
export fci_dimension, fci_determinants
export determinants_to_fock_vector, fock_vector_to_determinants, mps_decompose, mps_reconstruct
export OneBodyMPOTerm, TwoBodyMPOTerm, FermionMPO, ProductMPOTerm, ProductMPO
export build_dmrg_mpo, build_product_dmrg_mpo
export mpo_apply, mpo_expectation, product_mpo_expectation
export hartree_fock_mps, mps_particle_number, dmrg_sweep!
export two_site_tensor, split_two_site_tensor, replace_two_site_tensor
export two_site_projection_matrix, two_site_tensor_contract, two_site_energy
export cisdt_determinants, ci_to_ccsdt_amplitudes, ccsdt_wavefunction_coefficients
export ccsdt_energy, ccsdt_residuals, ccsdt_scf, run_cisdt_model
export eom_double_indices, eom_dimension, pack_eom_amplitudes, unpack_eom_amplitudes
export ccsd_residuals, eom_ccsd_diagonal, build_eom_ccsd_jacobian
export triples_correction

end
