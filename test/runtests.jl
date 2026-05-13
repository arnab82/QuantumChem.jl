"""
Test suite for QuantumChem.jl against Crawford Programming Project #5 reference values.
Also includes the Project #6 perturbative triples correction.

Reference system : H₂O in STO-3G (geometry in Bohr, from Crawford group)
Reference source : https://github.com/CrawfordGroup/ProgrammingProjects/tree/master/Project%2305

Expected energies (Eh):
  E_HF       = -74.942 079 928 192
  E_MP2 corr =  -0.049 149 636 120
  E_CCSD corr=  -0.070 680 088 376
  E_tot      = -75.012 760 016 568
  E(T)       =  -0.000 099 877 272
  E_CCSD(T)  = -75.012 859 893 840
  E_CCSDT    = -75.012 892 735 890
  E_HCI+PT2  = -75.012 980 218 058
  E_FCI      = -75.012 980 198 443
  E_DMRG     = -75.012 980 198 443
"""

using Test
using LinearAlgebra
using QuantumChem

# PyCall with libpython 3.13 can run Python-object finalizers without the GIL.
# The test process is short-lived, so disabling GC avoids nondeterministic
# libpython finalizer crashes without changing package behavior.
GC.enable(false)

# Tolerances — tight enough to catch regressions, loose enough for float variance
const RHF_TOL  = 1e-8
const MP2_TOL  = 1e-6
const DF_MP2_TOL = 1e-9
const CCSD_TOL = 1e-6
const CCSDT_TOL = 1e-6
const FULL_CCSDT_TOL = 1e-8
const UHF_TOL = 1e-9
const ROHF_TOL = 1e-9
const UCCSD_TOL = 1e-8
const UMP2_TOL = 1e-9
const CISD_TOL = 1e-8
const HCI_TOL = 1e-8
const FCI_TOL = 1e-8
const DMRG_TOL = 1e-8
const GRADIENT_TOL = 1e-8
const PROPERTY_TOL = 1e-8

# Reference values from Crawford Project #5
const E_HF_REF    = -74.942079928192
const E_MP2_REF   =  -0.049149636120
const E_DF_MP2_REF = -0.04909043720344019
const E_CCSD_REF  =  -0.070680088376
const E_TOT_REF   = -75.012760016568
const E_TRIPLES_REF = -0.000099877272
const E_CCSDT_REF   = -75.012859893840
const E_FULL_CCSDT_REF = -0.07081280769760294
const E_FULL_CCSDT_TOTAL_REF = -75.01289273589046
const E_UHF_OH_REF = -74.36040547332438
const S2_UHF_OH_REF = 0.7529381425680679
const E_ROHF_OH_REF = -74.35932208774248
const S2_ROHF_OH_REF = 0.75
const E_UCCSD_OH_REF = -0.02343230248645602
const E_UCCSD_OH_TOTAL_REF = -74.38383777581083
const E_UMP2_OH_REF = -0.015217583817377001
const E_UMP2_OH_TOTAL_REF = -74.37562305714175
const E_CISD_REF = -0.0691430716170365
const E_CISD_TOTAL_REF = -75.01122299980942
const E_HCI_REF = -0.07089840260761093
const E_HCI_PT2_REF = -1.8872576615707358e-6
const E_HCI_PT2_TOTAL_REF = -75.01298021805813
const E_FCI_REF = -0.07090027025027723
const E_FCI_TOTAL_REF = -75.01298019844313
const RHF_GRADIENT_REF = [
     0.0                 -0.0974413804       0.0;
     0.0863000587         0.0487206902       0.0;
    -0.0863000587         0.0487206902       0.0;
]
const CIS_FIRST_REF = [
    0.2872554996, 0.2872554996, 0.2872554996,
    0.3444249963, 0.3444249963, 0.3444249963,
    0.3564617587,
    0.3659889948, 0.3659889948, 0.3659889948,
    0.3945137992, 0.3945137992, 0.3945137992,
    0.4160717386,
]
const RPA_FIRST_REF = [
    0.2851637170, 0.2851637170, 0.2851637170,
    0.2997434467, 0.2997434467, 0.2997434467,
    0.3526266606, 0.3526266606, 0.3526266606,
    0.3547782530,
    0.3651313107, 0.3651313107, 0.3651313107,
    0.4153174946,
]
const RHF_DIPOLE_REF = [-5.124270729348505e-15, 0.6035212965259636, 5.193233064992663e-16]
const CIS_OSCILLATOR_REF = [
    0.0023412740124832583,
    9.677920247152556e-30,
    0.06492627323862235,
    0.01546735543794491,
    1.2519369106319123,
]
const EOM_OSCILLATOR_FOURTH_REF = 0.0019506820621747502
const EOM_TRANSITION_Z_FOURTH_REF = 0.09514228128608646

const ACETALDEHYDE_GEOM = """
7
6  0.000000000000     0.000000000000     0.000000000000
6  0.000000000000     0.000000000000     2.845112131228
8  1.899115961744     0.000000000000     4.139062527233
1 -1.894048308506     0.000000000000     3.747688672216
1  1.942500819960     0.000000000000    -0.701145981971
1 -1.007295466862    -1.669971842687    -0.705916966833
1 -1.007295466862     1.669971842687    -0.705916966833
"""

const H2O_PROJECT2_GEOM = """
3
8.00000000000     0.000000000000     0.000000000000    -0.134503695264
1.00000000000     0.000000000000    -1.684916670000     1.067335684736
1.00000000000     0.000000000000     1.684916670000     1.067335684736
"""

const H2O_PROJECT2_HESSIAN = """
3
0.0927643390 0.0000000000 0.0000000000
-0.0463821695 0.0000000000 0.0000000000
-0.0463821695 0.0000000000 0.0000000000
0.0000000000 0.3171327134 0.0000000000
0.0000000000 -0.1585663567 0.0800202030
0.0000000000 -0.1585663567 -0.0800202030
0.0000000000 0.0000000000 0.2800907293
0.0000000000 0.0347765865 -0.1400453646
0.0000000000 -0.0347765865 -0.1400453646
-0.0463821695 0.0000000000 0.0000000000
0.0514668232 0.0000000000 0.0000000000
-0.0050846537 0.0000000000 0.0000000000
0.0000000000 -0.1585663567 0.0347765865
0.0000000000 0.1730075524 -0.0573983947
0.0000000000 -0.0144411957 0.0226218083
0.0000000000 0.0800202030 -0.1400453646
0.0000000000 -0.0573983947 0.1268373488
0.0000000000 -0.0226218083 0.0132080159
-0.0463821695 0.0000000000 0.0000000000
-0.0050846537 0.0000000000 0.0000000000
0.0514668232 0.0000000000 0.0000000000
0.0000000000 -0.1585663567 -0.0347765865
0.0000000000 -0.0144411957 -0.0226218083
0.0000000000 0.1730075524 0.0573983947
0.0000000000 -0.0800202030 -0.1400453646
0.0000000000 0.0226218083 0.0132080159
0.0000000000 0.0573983947 0.1268373488
"""

const OH_RADICAL_GEOM = "O 0.0 0.0 0.0; H 0.0 0.0 1.8"

# Run the full calculation once and reuse in all tests
const rhf_result  = run_rhf(verbose=false)
const rhf_gradient_result = run_rhf_gradient(rhf_result; verbose=false)
const rhf_optimization_step = run_rhf_geometry_optimization(rhf_result;
                                                            maxiter=1,
                                                            initial_step=0.5,
                                                            max_step=0.05,
                                                            verbose=false)
const diis_rhf_result = run_rhf(diis=true; verbose=false)
const symmetry_rhf_result = run_rhf(symmetry=true; verbose=false)
const cs_symmetry_rhf_result = run_rhf(symmetry="Cs"; verbose=false)
const outcore_rhf_result = run_rhf(outcore=true; verbose=false)
const direct_rhf_result = run_direct_rhf(verbose=false)
const uhf_result = run_uhf(verbose=false)
const uhf_oh_result = run_uhf(atoms=OH_RADICAL_GEOM, basis="sto-3g",
                              charge=0, spin=1, unit="Bohr",
                              diis=true, verbose=false)
const rohf_result = run_rohf(verbose=false)
const rohf_oh_result = run_rohf(atoms=OH_RADICAL_GEOM, basis="sto-3g",
                                charge=0, spin=1, unit="Bohr",
                                diis=true, verbose=false)
const ump2_closed_result = run_ump2(uhf_result; verbose=false)
const ump2_oh_result = run_ump2(uhf_oh_result; verbose=false)
const uccsd_closed_result = run_uccsd(uhf_result; diis=true, verbose=false)
const uccsd_oh_result = run_uccsd(uhf_oh_result; diis=true, verbose=false)
const mp2_result  = run_mp2(rhf_result; verbose=false)
const df_mp2_result = run_df_mp2(rhf_result; return_eri=true, verbose=false)
const cisd_result = run_cisd(rhf_result, mp2_result; verbose=false)
const hci_result = run_hci(rhf_result, mp2_result; verbose=false)
const fci_result = run_fci(rhf_result, mp2_result; verbose=false)
const dmrg_result = run_fci_dmrg(rhf_result, mp2_result; fci_result=fci_result, verbose=false)
const ccsd_result = run_ccsd(rhf_result, mp2_result; verbose=false)
const ccsd_diis_result = run_ccsd(rhf_result, mp2_result; diis=true, verbose=false)
const ccsdt_result = run_ccsd_t(rhf_result, mp2_result, ccsd_result; verbose=false)
const ccsdt_full_result = run_ccsdt(rhf_result, mp2_result; verbose=false)
const excited_result = run_excited_states(rhf_result, mp2_result; verbose=false)
const davidson_result = run_davidson_cis(rhf_result, mp2_result; nroots=5, verbose=false)
const eomccsd_result = run_eom_ccsd(rhf_result, mp2_result, ccsd_diis_result; nroots=5, verbose=false)
const cis_properties_result = run_cis_properties(rhf_result, excited_result; nroots=5, verbose=false)
const eomccsd_properties_result = run_eom_ccsd_properties(rhf_result, eomccsd_result; nroots=5, verbose=false)

@testset "QuantumChem" begin

    @testset "Project #1 molecular geometry" begin
        mol = read_molecule(ACETALDEHYDE_GEOM)
        geometry = run_geometry_analysis(mol)

        @test length(mol) == 7
        @test isapprox(distance(mol, 1, 2), 2.845112131228; atol=1e-12)
        @test isapprox(distance(mol, 2, 3), 2.29803; atol=1e-5)
        @test isapprox(bond_angle(mol, 1, 2, 3), 124.268308; atol=1e-6)
        @test all(isapprox.(center_of_mass(mol), [0.64494926, 0.0, 2.31663792]; atol=2e-8))
        @test all(isapprox.(geometry.principal_moments, [31.964078, 178.649562, 199.371127]; atol=2e-6))
        @test geometry.rotor_type == :asymmetric
        @test all(isapprox.(geometry.rotational_constants.MHz, [56461.542, 10102.130, 9052.169]; atol=1e-3))
        @test all(isapprox.(geometry.rotational_constants.cm, [1.8834, 0.3370, 0.3019]; atol=1e-4))
    end

    @testset "Project #2 harmonic vibrations" begin
        hessian_data = read_hessian(H2O_PROJECT2_HESSIAN)
        vib = run_vibrational_analysis(read_molecule(H2O_PROJECT2_GEOM), hessian_data.hessian)

        @test hessian_data.natoms == 3
        @test all(isapprox.(vib.eigenvalues[1:6],
                            [0.2351542439, 0.2107113210, 0.1317512832,
                             0.0561123974, 0.0547551476, 0.0518216614];
                            atol=1e-8))
        @test all(isapprox.(vib.frequencies[1:6],
                            [2492.7600, 2359.6522, 1865.8704,
                             1217.6808, 1202.8640, 1170.1990];
                            atol=2e-3))
    end

    @testset "RHF" begin
        @test isapprox(rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test rhf_result.nbasis == 7          # STO-3G H₂O has 7 basis functions
        @test rhf_result.n_elec == 10
        @test length(rhf_result.orbital_energies) == 7
    end

    @testset "Project #24 RHF analytic gradients" begin
        gradient = rhf_gradient_result.gradient

        @test size(gradient) == (3, 3)
        @test isapprox(gradient, RHF_GRADIENT_REF; atol=GRADIENT_TOL)
        @test isapprox(gradient,
                       rhf_gradient_result.electronic_gradient .+
                       rhf_gradient_result.nuclear_gradient;
                       atol=1e-12)
        @test all(isapprox.(vec(sum(gradient; dims=1)), zeros(3); atol=1e-12))
        @test isapprox(rhf_gradient_result.max_force, maximum(abs.(gradient)); atol=1e-14)
        @test isapprox(tr(rhf_spin_density(rhf_result) * rhf_result.S),
                       rhf_result.n_elec; atol=1e-10)

        atom2_gradient = rhf_gradient(rhf_result; atmlst=[2])
        @test size(atom2_gradient) == (1, 3)
        @test isapprox(atom2_gradient[1, :], gradient[2, :]; atol=1e-12)
    end

    @testset "Project #25 RHF geometry optimization" begin
        opt = rhf_optimization_step

        @test opt.iterations == 1
        @test opt.method == :bfgs
        @test opt.symbols == ["O", "H", "H"]
        @test opt.unit == "Bohr"
        @test size(opt.coordinates) == (3, 3)
        @test length(opt.energies) == 2
        @test opt.energies[end] < opt.energies[1]
        @test opt.energy_changes[1] < 0.0
        @test opt.max_forces[end] < opt.max_forces[1]
        @test opt.atoms == format_atoms_bohr(opt.symbols, opt.coordinates)
        @test maximum(norm(@view opt.displacements[1][atom, :])
                      for atom in axes(opt.displacements[1], 1)) <= opt.max_step + 1e-12
        @test isapprox(rhf_gradient(opt.rhf), opt.gradient; atol=1e-12)
    end

    @testset "Project #8 RHF DIIS" begin
        @test isapprox(diis_rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test diis_rhf_result.iterations < rhf_result.iterations
        @test diis_rhf_result.iterations <= 12
        @test isapprox(sum(diis_coefficients([Matrix(I, 2, 2), 2.0 .* Matrix(I, 2, 2)])), 1.0; atol=1e-12)
    end

    @testset "Project #9 symmetry in SCF" begin
        info = water_c2v_sto3g_symmetry()
        S_so = symmetry_transform_1e(rhf_result.S, info)
        S_blocks = symmetry_blocks(S_so, info)

        @test sopi(info) == [4, 1, 2]
        @test symmetry_offblock_norm(S_so, info) < 1e-12
        @test isapprox(S_so[4,4], 1.1817599; atol=1e-7)
        @test isapprox(S_so[6,7], 0.3796290; atol=1e-7)
        @test assemble_symmetry_blocks(S_blocks) == symmetry_project(S_so, info)
        @test block_multiply(S_blocks, S_blocks) == [block * block for block in S_blocks]

        @test isapprox(symmetry_rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test sopi(symmetry_rhf_result.symmetry_info) == [4, 1, 2]
        @test symmetry_rhf_result.symmetry_info.occupations == [3, 1, 1]
        @test symmetry_offblock_norm(symmetry_rhf_result.fock_so, symmetry_rhf_result.symmetry_info) < 1e-10

        @test isapprox(cs_symmetry_rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test sopi(cs_symmetry_rhf_result.symmetry_info) == [6, 1]
        @test cs_symmetry_rhf_result.symmetry_info.occupations == [4, 1]
        @test symmetry_offblock_norm(cs_symmetry_rhf_result.fock_so, cs_symmetry_rhf_result.symmetry_info) < 1e-10
    end

    @testset "MP2" begin
        @test isapprox(mp2_result.emp2, E_MP2_REF; atol=MP2_TOL)
        @test size(mp2_result.new_eri) == (7, 7, 7, 7)
    end

    @testset "Project #27 density-fitted MP2" begin
        @test df_mp2_result.auxbasis == "weigend"
        @test df_mp2_result.naux == 71
        @test size(df_mp2_result.mo_factors) == (7, 7, 71)
        @test size(df_mp2_result.new_eri) == (7, 7, 7, 7)
        @test isapprox(df_mp2_result.emp2, E_DF_MP2_REF; atol=DF_MP2_TOL)
        @test isapprox(df_mp2_result.total_energy,
                       rhf_result.total_energy + E_DF_MP2_REF; atol=DF_MP2_TOL)
        @test abs(df_mp2_result.emp2 - mp2_result.emp2) < 1e-4
        @test df_mp2_result.metric_condition > 1.0

        @test isapprox(
            df_mp2_result.new_eri[1, 2, 3, 4],
            dot(view(df_mp2_result.mo_factors, 1, 2, :),
                view(df_mp2_result.mo_factors, 3, 4, :));
            atol=1e-12,
        )
        @test isapprox(df_mp2_result.new_eri[1,2,3,4],
                       df_mp2_result.new_eri[3,4,1,2]; atol=1e-12)
        @test isapprox(
            compute_df_mp2(df_mp2_result.mo_factors,
                           rhf_result.orbital_energies,
                           rhf_result.n_elec ÷ 2,
                           rhf_result.nbasis),
            df_mp2_result.emp2;
            atol=1e-12,
        )
        @test isapprox(
            compute_mp2(df_mp2_result.new_eri,
                        rhf_result.orbital_energies,
                        rhf_result.n_elec ÷ 2,
                        rhf_result.nbasis),
            df_mp2_result.emp2;
            atol=1e-12,
        )
    end

    @testset "ASO basis" begin
        aso = mo_to_aso(mp2_result.new_eri)
        # <pq||rs> must be antisymmetric: <pq||rs> = -<qp||rs> = -<pq||sr>
        @test isapprox(aso[1,2,3,4], -aso[2,1,3,4]; atol=1e-14)
        @test isapprox(aso[1,2,3,4], -aso[1,2,4,3]; atol=1e-14)
        # Hermitian symmetry: <pq||rs> = <rs||pq>
        @test isapprox(aso[1,2,3,4],  aso[3,4,1,2]; atol=1e-14)
    end

    @testset "Spin-orbital Fock matrix" begin
        F_so = Mat_motoMat_so(Mat_aotoMat_mo(rhf_result.mo_coeffs, rhf_result.fock))
        # Must be diagonal (canonical MO basis)
        nso = size(F_so, 1)
        for i in 1:nso, j in 1:nso
            i != j && @test isapprox(F_so[i,j], 0.0; atol=1e-10)
        end
        # α and β spin-orbitals of the same spatial orbital share the same energy
        for p in 1:7
            @test isapprox(F_so[2p-1, 2p-1], F_so[2p, 2p]; atol=1e-14)
        end
    end

    @testset "MP2 in SO basis" begin
        aso = mo_to_aso(mp2_result.new_eri)
        o   = rhf_result.n_elec
        v   = 2*rhf_result.nbasis - o
        oi  = 1:o;  vi = (o+1):(o+v)
        aso_oovv = aso[oi, oi, vi, vi]
        td = QuantumChem.make_td(aso_oovv, rhf_result.orbital_energies, o, o)
        @test isapprox(mp2_so(aso_oovv, td), E_MP2_REF; atol=MP2_TOL)
    end

    @testset "CCSD correlation energy" begin
        @test ccsd_result.converged
        @test isapprox(ccsd_result.E_ccsd, E_CCSD_REF; atol=CCSD_TOL)
    end

    @testset "Project #10 CCSD amplitude DIIS" begin
        @test ccsd_diis_result.converged
        @test isapprox(ccsd_diis_result.E_ccsd, E_CCSD_REF; atol=CCSD_TOL)
        @test ccsd_diis_result.iterations < ccsd_result.iterations
        @test ccsd_diis_result.iterations <= 20
        error = cc_amplitude_error(ccsd_diis_result.ts, ccsd_diis_result.td,
                                   ccsd_result.ts, ccsd_result.td)
        @test length(error) == length(ccsd_result.ts) + length(ccsd_result.td)
        @test isapprox(sum(cc_diis_coefficients([error, 2.0 .* error])), 1.0; atol=1e-12)
    end

    @testset "Project #11 out-of-core SCF" begin
        @test outcore_rhf_result.outcore
        @test !outcore_rhf_result.direct
        @test outcore_rhf_result.eri === nothing
        @test isapprox(outcore_rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)

        path = tempname()
        eri_file = write_unique_eri_file(rhf_result.eri; path, cutoff=1e-14)
        try
            npair = rhf_result.nbasis * (rhf_result.nbasis + 1) ÷ 2
            @test eri_file.nrecords <= npair * (npair + 1) ÷ 2
            @test read_eri_file_header(path).nrecords == eri_file.nrecords

            D = QuantumChem.make_density(rhf_result.mo_coeffs, rhf_result.n_elec ÷ 2)
            fock_core = QuantumChem.make_fock(D, rhf_result.h1e, rhf_result.eri)
            fock_outcore = make_fock_outcore(D, rhf_result.h1e, eri_file; batch_size=5)
            @test isapprox(fock_outcore, fock_core; atol=1e-10)
        finally
            rm(path; force=true)
        end
    end

    @testset "Project #26 integral-direct RHF" begin
        @test direct_rhf_result.direct
        @test !direct_rhf_result.outcore
        @test direct_rhf_result.eri === nothing
        @test isapprox(direct_rhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test isapprox(direct_rhf_result.energy, rhf_result.energy; atol=1e-10)
        @test isapprox(direct_rhf_result.fock, rhf_result.fock; atol=1e-9)

        D = QuantumChem.make_density(rhf_result.mo_coeffs, rhf_result.n_elec ÷ 2)
        direct_fock = make_fock_direct(D, rhf_result.h1e, rhf_result.mol)
        incore_fock = QuantumChem.make_fock(D, rhf_result.h1e, rhf_result.eri)
        @test isapprox(direct_fock, incore_fock; atol=1e-10)

        jk = direct_jk(rhf_result.mol, D)
        @test size(jk.J) == size(rhf_result.h1e)
        @test size(jk.K) == size(rhf_result.h1e)
        @test isapprox(rhf_result.h1e .+ 2.0 .* jk.J .- jk.K, direct_fock; atol=1e-12)

        @test_throws ArgumentError run_rhf(direct=true, outcore=true, verbose=false)
        @test_throws ArgumentError run_rhf(direct=true, symmetry=true, verbose=false)
    end

    @testset "Project #15 unrestricted Hartree-Fock" begin
        @test uhf_electron_counts(10, 0) == (5, 5)
        @test uhf_electron_counts(9, 1) == (5, 4)
        @test_throws ArgumentError uhf_electron_counts(9, 0)

        @test isapprox(uhf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test uhf_result.n_alpha == 5
        @test uhf_result.n_beta == 5
        @test isapprox(uhf_result.spin_square, 0.0; atol=1e-10)
        @test isapprox(uhf_result.spin_multiplicity, 1.0; atol=1e-10)
        @test isapprox(uhf_result.density_alpha, uhf_result.density_beta; atol=1e-12)
        @test isapprox(uhf_result.fock_alpha, uhf_result.fock_beta; atol=1e-12)
        @test isapprox(uhf_result.fock_alpha, rhf_result.fock; atol=1e-10)

        fock_alpha, fock_beta = make_uhf_fock(uhf_result.density_alpha,
                                              uhf_result.density_beta,
                                              uhf_result.h1e,
                                              uhf_result.eri)
        @test isapprox(fock_alpha, uhf_result.fock_alpha; atol=1e-12)
        @test isapprox(fock_beta, uhf_result.fock_beta; atol=1e-12)
        @test isapprox(uhf_energy(uhf_result.density_alpha,
                                  uhf_result.density_beta,
                                  fock_alpha, fock_beta,
                                  uhf_result.h1e),
                        uhf_result.energy; atol=1e-10)

        @test isapprox(uhf_oh_result.total_energy, E_UHF_OH_REF; atol=UHF_TOL)
        @test isapprox(uhf_oh_result.spin_square, S2_UHF_OH_REF; atol=1e-8)
        @test uhf_oh_result.n_alpha == 5
        @test uhf_oh_result.n_beta == 4
        @test uhf_oh_result.iterations <= 20
    end

    @testset "Project #16 restricted open-shell Hartree-Fock" begin
        @test rohf_shell_counts(10, 0) == (n_closed = 5, n_open = 0, n_alpha = 5, n_beta = 5)
        @test rohf_shell_counts(9, 1) == (n_closed = 4, n_open = 1, n_alpha = 5, n_beta = 4)
        @test_throws ArgumentError rohf_shell_counts(9, -1)
        @test rohf_occupations(6, 4, 1) == [2.0, 2.0, 2.0, 2.0, 1.0, 0.0]

        @test isapprox(rohf_result.total_energy, E_HF_REF; atol=RHF_TOL)
        @test rohf_result.n_closed == 5
        @test rohf_result.n_open == 0
        @test isapprox(rohf_result.spin_square, 0.0; atol=1e-12)
        @test isapprox(rohf_result.fock, rhf_result.fock; atol=1e-10)

        @test isapprox(rohf_oh_result.total_energy, E_ROHF_OH_REF; atol=ROHF_TOL)
        @test isapprox(rohf_oh_result.spin_square, S2_ROHF_OH_REF; atol=1e-12)
        @test rohf_oh_result.n_closed == 4
        @test rohf_oh_result.n_open == 1
        @test rohf_oh_result.n_alpha == 5
        @test rohf_oh_result.n_beta == 4
        @test rohf_oh_result.occupations == [2.0, 2.0, 2.0, 2.0, 1.0, 0.0]
        @test rohf_oh_result.iterations <= 12
        @test rohf_oh_result.total_energy > uhf_oh_result.total_energy

        density = make_rohf_density(rohf_oh_result.mo_coeffs,
                                    rohf_oh_result.n_closed,
                                    rohf_oh_result.n_open)
        @test isapprox(density.alpha, rohf_oh_result.density_alpha; atol=1e-12)
        @test isapprox(density.beta, rohf_oh_result.density_beta; atol=1e-12)
        @test isapprox(density.alpha - density.beta, rohf_oh_result.density_open; atol=1e-12)

        f_roothaan = roothaan_fock(rohf_oh_result.fock_alpha,
                                   rohf_oh_result.fock_beta,
                                   rohf_oh_result.density_alpha,
                                   rohf_oh_result.density_beta,
                                   rohf_oh_result.S)
        @test isapprox(f_roothaan, f_roothaan'; atol=1e-12)
        @test isapprox(f_roothaan, rohf_oh_result.fock; atol=1e-12)
    end

    @testset "Project #17 unrestricted CCSD" begin
        order = uhf_spinorbital_order(uhf_result)
        @test length(order) == 2 * uhf_result.nbasis
        @test all(item.spin == :alpha && item.occupied for item in order[1:uhf_result.n_alpha])
        @test all(item.spin == :beta && item.occupied
                  for item in order[(uhf_result.n_alpha + 1):uhf_result.n_elec])
        @test count(==(:alpha), uccsd_oh_result.spins) == uhf_oh_result.nbasis
        @test count(==(:beta), uccsd_oh_result.spins) == uhf_oh_result.nbasis

        @test uccsd_closed_result.converged
        @test isapprox(uccsd_closed_result.E_ccsd, E_CCSD_REF; atol=CCSD_TOL)
        @test isapprox(uccsd_closed_result.total_energy, E_TOT_REF; atol=CCSD_TOL)
        @test size(uccsd_closed_result.aso) == (14, 14, 14, 14)
        @test isapprox(uccsd_closed_result.aso[1,2,3,4],
                        -uccsd_closed_result.aso[2,1,3,4]; atol=1e-14)
        @test isapprox(uccsd_closed_result.aso[1,2,3,4],
                        -uccsd_closed_result.aso[1,2,4,3]; atol=1e-14)

        @test uccsd_oh_result.converged
        @test isapprox(uccsd_oh_result.E_ccsd, E_UCCSD_OH_REF; atol=UCCSD_TOL)
        @test isapprox(uccsd_oh_result.total_energy, E_UCCSD_OH_TOTAL_REF; atol=UCCSD_TOL)
        @test uccsd_oh_result.total_energy < uhf_oh_result.total_energy
        @test uccsd_oh_result.iterations <= 25
        @test size(uccsd_oh_result.ts) == (9, 3)
        @test size(uccsd_oh_result.td) == (9, 9, 3, 3)
        @test uccsd_oh_result.orbital_order[1] == (spin=:alpha, orbital=1, occupied=true)
        @test uccsd_oh_result.orbital_order[end] == (spin=:beta, orbital=6, occupied=false)

        inputs = build_uccsd_spinorbital_inputs(uhf_oh_result)
        td0 = make_td_spinorbital(inputs.aso[1:uhf_oh_result.n_elec,
                                             1:uhf_oh_result.n_elec,
                                             (uhf_oh_result.n_elec + 1):end,
                                             (uhf_oh_result.n_elec + 1):end],
                                  inputs.orbital_energies[1:uhf_oh_result.n_elec],
                                  inputs.orbital_energies[(uhf_oh_result.n_elec + 1):end])
        @test size(td0) == (9, 9, 3, 3)
    end

    @testset "Project #18 unrestricted MP2" begin
        @test isapprox(ump2_closed_result.emp2, E_MP2_REF; atol=MP2_TOL)
        @test isapprox(ump2_closed_result.total_energy, E_HF_REF + E_MP2_REF; atol=MP2_TOL)
        @test size(ump2_closed_result.oovv) == (10, 10, 4, 4)
        @test size(ump2_closed_result.td) == (10, 10, 4, 4)
        @test count(==(:alpha), ump2_closed_result.spins) == uhf_result.nbasis
        @test count(==(:beta), ump2_closed_result.spins) == uhf_result.nbasis

        @test isapprox(ump2_oh_result.emp2, E_UMP2_OH_REF; atol=UMP2_TOL)
        @test isapprox(ump2_oh_result.total_energy, E_UMP2_OH_TOTAL_REF; atol=UMP2_TOL)
        @test ump2_oh_result.total_energy < uhf_oh_result.total_energy
        @test uccsd_oh_result.total_energy < ump2_oh_result.total_energy
        @test size(ump2_oh_result.oovv) == (9, 9, 3, 3)
        @test size(ump2_oh_result.td) == (9, 9, 3, 3)
        @test ump2_oh_result.orbital_order[1] == (spin=:alpha, orbital=1, occupied=true)
        @test ump2_oh_result.orbital_order[end] == (spin=:beta, orbital=6, occupied=false)

        recomputed = compute_ump2(ump2_oh_result.oovv,
                                  ump2_oh_result.orbital_energies[1:uhf_oh_result.n_elec],
                                  ump2_oh_result.orbital_energies[(uhf_oh_result.n_elec + 1):end])
        @test isapprox(recomputed.emp2, ump2_oh_result.emp2; atol=1e-12)
        @test isapprox(recomputed.td, ump2_oh_result.td; atol=1e-12)
        @test isapprox(mp2_so(ump2_oh_result.oovv, ump2_oh_result.td),
                        ump2_oh_result.emp2; atol=1e-12)
    end

    @testset "Project #19 CISD" begin
        dets = cisd_determinants(2 * rhf_result.nbasis, rhf_result.n_elec)
        @test length(dets.determinants) == 311
        @test dets.labels[1] == (rank=0, holes=(), particles=())
        @test occupied_orbitals(dets.determinants[1], 2 * rhf_result.nbasis) == collect(1:rhf_result.n_elec)

        @test cisd_result.dimension == 311
        @test cisd_result.singles == 40
        @test cisd_result.doubles == 270
        @test size(cisd_result.hamiltonian) == (311, 311)
        @test isapprox(cisd_result.hamiltonian, cisd_result.hamiltonian'; atol=1e-12)
        @test isapprox(cisd_result.hamiltonian[1,1] + rhf_result.total_energy - rhf_result.energy,
                        E_HF_REF; atol=RHF_TOL)

        @test isapprox(cisd_result.E_cisd, E_CISD_REF; atol=CISD_TOL)
        @test isapprox(cisd_result.total_energy, E_CISD_TOTAL_REF; atol=CISD_TOL)
        @test cisd_result.total_energy < rhf_result.total_energy
        @test cisd_result.total_energy < rhf_result.total_energy + mp2_result.emp2
        @test cisd_result.total_energy > ccsd_result.total_energy
        @test 0.9 < cisd_result.reference_weight < 1.0

        h_mo = Mat_aotoMat_mo(rhf_result.mo_coeffs, rhf_result.h1e)
        h_so = spatial_to_spinorbital_1e(h_mo)
        @test isapprox(h_so[1,3], h_mo[1,2]; atol=1e-12)
        @test isapprox(h_so[2,4], h_mo[1,2]; atol=1e-12)
        @test isapprox(h_so[1,4], 0.0; atol=1e-14)
    end

    @testset "Project #12 CIS and TDHF/RPA" begin
        @test size(excited_result.cis_matrix) == (40, 40)
        @test size(excited_result.singlet_matrix) == (10, 10)
        @test size(excited_result.triplet_matrix) == (10, 10)
        @test size(excited_result.rpa_matrix) == (80, 80)
        @test size(excited_result.singlet_vectors) == (10, 10)

        @test all(isapprox.(excited_result.cis_energies[1:length(CIS_FIRST_REF)],
                            CIS_FIRST_REF; atol=1e-9))
        @test all(isapprox.(excited_result.rpa_energies[1:length(RPA_FIRST_REF)],
                            RPA_FIRST_REF; atol=1e-9))
        @test all(isapprox.(excited_result.rpa_reduced_energies, excited_result.rpa_energies; atol=1e-9))

        @test isapprox(excited_result.singlet_energies[1], 0.3564617587; atol=1e-9)
        @test isapprox(excited_result.triplet_energies[1], 0.2872554996; atol=1e-9)
        @test isapprox(excited_result.cis_energies_ev[1],
                        excited_result.cis_energies[1] * HARTREE_TO_EV; atol=1e-12)
    end

    @testset "Project #13 Davidson-Liu CIS" begin
        @test davidson_result.converged
        @test davidson_result.iterations <= 12
        @test all(davidson_result.residual_norms .< 1e-7)
        @test all(isapprox.(davidson_result.energies, excited_result.singlet_energies[1:5]; atol=1e-9))

        trial = collect(range(0.1, 1.0; length=size(excited_result.singlet_matrix, 1)))
        sigma = cis_singlet_sigma(mp2_result.new_eri, rhf_result.orbital_energies, rhf_result.n_elec, trial)
        @test isapprox(sigma, excited_result.singlet_matrix * trial; atol=1e-12)
        @test isapprox(davidson_result.diagonal, diag(excited_result.singlet_matrix); atol=1e-12)
    end

    @testset "Project #14 EOM-CCSD" begin
        @test eomccsd_result.dimension == 310
        @test eomccsd_result.singles == 40
        @test eomccsd_result.doubles == 270
        @test maximum(abs.(eomccsd_result.residual)) < 1e-8
        @test length(eomccsd_result.all_eigenvalues) == eomccsd_result.dimension
        @test size(eomccsd_result.vectors) == (eomccsd_result.dimension, 5)
        @test all(eomccsd_result.energies .> 0.0)
        @test all(isapprox.(eomccsd_result.energies,
                            [0.2752578782, 0.2752578782, 0.2752578782,
                             0.3232441161, 0.3613244250];
                            atol=1e-7))

        rs = ones(Float64, rhf_result.n_elec, 2 * rhf_result.nbasis - rhf_result.n_elec)
        rd = zeros(Float64, rhf_result.n_elec, rhf_result.n_elec,
                   2 * rhf_result.nbasis - rhf_result.n_elec,
                   2 * rhf_result.nbasis - rhf_result.n_elec)
        rd[1,2,1,2] = 0.25
        rd[2,1,1,2] = -0.25
        rd[1,2,2,1] = -0.25
        rd[2,1,2,1] = 0.25
        packed = pack_eom_amplitudes(rs, rd, rhf_result.n_elec, 2 * rhf_result.nbasis - rhf_result.n_elec)
        rs2, rd2 = unpack_eom_amplitudes(packed, rhf_result.n_elec, 2 * rhf_result.nbasis - rhf_result.n_elec)
        @test rs2 == rs
        @test isapprox(rd2[1,2,1,2], 0.25; atol=1e-14)
        @test isapprox(rd2[2,1,1,2], -0.25; atol=1e-14)
        @test isapprox(rd2[1,2,2,1], -0.25; atol=1e-14)
        @test isapprox(rd2[2,1,2,1], 0.25; atol=1e-14)
    end

    @testset "Project #28 CIS and EOM-CCSD properties" begin
        ao_dipoles = ao_dipole_integrals(rhf_result)
        @test size(ao_dipoles) == (3, rhf_result.nbasis, rhf_result.nbasis)
        @test isapprox(rhf_dipole_moment(rhf_result; ao_dipoles), RHF_DIPOLE_REF; atol=PROPERTY_TOL)

        @test cis_properties_result.method == :CIS
        @test size(cis_properties_result.transition_dipoles) == (5, 3)
        @test all(isapprox.(cis_properties_result.energies,
                            excited_result.singlet_energies[1:5]; atol=1e-12))
        @test all(isapprox.(cis_properties_result.oscillator_strengths,
                            CIS_OSCILLATOR_REF; atol=PROPERTY_TOL))
        @test isapprox(
            oscillator_strengths(cis_properties_result.energies,
                                 cis_properties_result.transition_dipoles),
            cis_properties_result.oscillator_strengths;
            atol=1e-12,
        )
        @test isapprox(
            cis_transition_dipoles(rhf_result,
                                   excited_result.singlet_vectors[:, 1:5];
                                   mo_dipoles=cis_properties_result.mo_dipoles),
            cis_properties_result.transition_dipoles;
            atol=1e-12,
        )

        @test eomccsd_properties_result.method == :EOM_CCSD
        @test size(eomccsd_properties_result.transition_dipoles) == (5, 3)
        @test all(isapprox.(eomccsd_properties_result.energies,
                            eomccsd_result.energies; atol=1e-12))
        @test isapprox(
            oscillator_strengths(eomccsd_properties_result.energies,
                                 eomccsd_properties_result.transition_dipoles),
            eomccsd_properties_result.oscillator_strengths;
            atol=1e-12,
        )
        @test maximum(eomccsd_properties_result.oscillator_strengths[[1, 2, 3, 5]]) < 1e-20
        @test isapprox(eomccsd_properties_result.oscillator_strengths[4],
                       EOM_OSCILLATOR_FOURTH_REF; atol=PROPERTY_TOL)
        @test isapprox(abs(eomccsd_properties_result.transition_dipoles[4, 3]),
                       EOM_TRANSITION_Z_FOURTH_REF; atol=PROPERTY_TOL)
    end

    @testset "CCSD total energy" begin
        @test isapprox(ccsd_result.total_energy, E_TOT_REF; atol=CCSD_TOL)
    end

    @testset "CCSD(T) triples correction" begin
        @test isapprox(ccsdt_result.E_triples, E_TRIPLES_REF; atol=CCSDT_TOL)
    end

    @testset "CCSD(T) total energy" begin
        @test isapprox(ccsdt_result.total_energy, E_CCSDT_REF; atol=CCSDT_TOL)
    end

    @testset "Project #20 CCSDT" begin
        o = rhf_result.n_elec
        v = 2 * rhf_result.nbasis - o

        @test ccsdt_full_result.converged
        @test ccsdt_full_result.iterations <= 25
        @test ccsdt_full_result.residual_norm < 1e-8
        @test ccsdt_full_result.wicked_terms == (energy=3, singles=15, doubles=37, triples=47)
        @test isapprox(ccsdt_full_result.E_ccsdt, E_FULL_CCSDT_REF; atol=FULL_CCSDT_TOL)
        @test isapprox(ccsdt_full_result.total_energy, E_FULL_CCSDT_TOTAL_REF; atol=FULL_CCSDT_TOL)
        @test ccsdt_full_result.total_energy < ccsd_result.total_energy
        @test ccsdt_full_result.total_energy < ccsdt_result.total_energy

        @test size(ccsdt_full_result.t1) == (o, v)
        @test size(ccsdt_full_result.t2) == (o, o, v, v)
        @test size(ccsdt_full_result.t3) == (o, o, o, v, v, v)
        @test isapprox(ccsdt_full_result.t3[1,2,3,1,2,3],
                       -ccsdt_full_result.t3[2,1,3,1,2,3]; atol=1e-10)
        @test isapprox(ccsdt_full_result.t3[1,2,3,1,2,3],
                       -ccsdt_full_result.t3[1,2,3,2,1,3]; atol=1e-10)
    end

    @testset "Project #21 heat-bath selected CI" begin
        @test hci_result.converged
        @test hci_result.dimension == 101
        @test hci_result.iterations == 4
        @test hci_result.added_per_iteration == [40, 54, 6, 0]
        @test 0 < hci_result.n_external <= fci_result.dimension - hci_result.dimension
        @test hci_result.max_external_score <= hci_result.epsilon1
        @test isapprox(hci_result.E_hci, E_HCI_REF; atol=HCI_TOL)
        @test isapprox(hci_result.E_pt2, E_HCI_PT2_REF; atol=HCI_TOL)
        @test isapprox(hci_result.total_energy, E_HCI_PT2_TOTAL_REF; atol=HCI_TOL)
        @test hci_result.variational_total_energy < ccsdt_full_result.total_energy
        @test hci_result.total_energy < hci_result.variational_total_energy
        @test 0.9 < hci_result.reference_weight < 1.0

        dets = cisd_determinants(2 * rhf_result.nbasis, rhf_result.n_elec)
        @test length(connected_determinants(dets.determinants[1], 2 * rhf_result.nbasis)) ==
              rhf_result.n_elec * (2 * rhf_result.nbasis - rhf_result.n_elec) +
              binomial(rhf_result.n_elec, 2) * binomial(2 * rhf_result.nbasis - rhf_result.n_elec, 2)
        @test isapprox(
            hamiltonian_element(cisd_result.h_so, cisd_result.aso,
                                dets.determinants[1], dets.determinants[1]),
            cisd_result.hamiltonian[1,1];
            atol=1e-10,
        )
        @test isapprox(
            hamiltonian_element(cisd_result.h_so, cisd_result.aso,
                                dets.determinants[2], dets.determinants[1]),
            cisd_result.hamiltonian[2,1];
            atol=1e-10,
        )
    end

    @testset "Project #22 full CI" begin
        nspin = 2 * rhf_result.nbasis
        nelec = rhf_result.n_elec

        @test fci_dimension(nspin, nelec) == 1001
        @test fci_result.dimension == 1001
        @test fci_result.labels[1] == (rank=0, holes=(), particles=())
        @test occupied_orbitals(fci_result.determinants[1], nspin) == collect(1:nelec)
        @test fci_result.rank_counts[0] == 1
        @test fci_result.rank_counts[1] == 40
        @test fci_result.rank_counts[2] == 270
        @test fci_result.rank_counts[3] == 480
        @test fci_result.rank_counts[4] == 210
        @test isapprox(fci_result.hamiltonian, fci_result.hamiltonian'; atol=1e-12)
        @test isapprox(fci_result.E_fci, E_FCI_REF; atol=FCI_TOL)
        @test isapprox(fci_result.total_energy, E_FCI_TOTAL_REF; atol=FCI_TOL)
        @test fci_result.total_energy < hci_result.variational_total_energy
        @test fci_result.total_energy < ccsdt_full_result.total_energy
        @test 0.9 < fci_result.reference_weight < 1.0
    end

    @testset "Project #23 DMRG/MPS" begin
        nspin = 2 * rhf_result.nbasis

        @test dmrg_result.dimension == fci_result.dimension
        @test length(dmrg_result.tensors) == nspin
        @test dmrg_result.bond_dimensions ==
              [2, 4, 8, 16, 24, 29, 25, 19, 22, 16, 8, 4, 2]
        @test dmrg_result.max_bond_dimension == 29
        @test dmrg_result.discarded_weight < 1e-20
        @test dmrg_result.reconstruction_error < 1e-12
        @test isapprox(dmrg_result.sector_weight, 1.0; atol=1e-12)
        @test isapprox(dmrg_result.E_dmrg, E_FCI_REF; atol=DMRG_TOL)
        @test isapprox(dmrg_result.total_energy, E_FCI_TOTAL_REF; atol=DMRG_TOL)
        @test isapprox(mps_reconstruct(dmrg_result.tensors), dmrg_result.fock_state; atol=1e-12)
        @test isapprox(fock_vector_to_determinants(dmrg_result.reconstructed_state,
                                                   fci_result.determinants),
                        fci_result.ci_vector; atol=1e-12)
        @test dmrg_result.mpo.nspin == nspin
        @test dmrg_result.mpo.n_elec == rhf_result.n_elec
        @test 0 < length(dmrg_result.mpo.one_body) <= nspin^2
        @test 0 < length(dmrg_result.mpo.two_body) <= nspin^4
        @test all(abs(term.value) > 1e-14 for term in dmrg_result.mpo.one_body)
        @test all(abs(term.value) > 1e-14 for term in dmrg_result.mpo.two_body)
        @test isapprox(mpo_apply(dmrg_result.mpo, fci_result.ci_vector; use_cache=false),
                        fci_result.hamiltonian * fci_result.ci_vector; atol=1e-10)
        @test isapprox(mpo_expectation(dmrg_result.mpo, dmrg_result.tensors),
                        fci_result.electronic_energy; atol=1e-10)

        product_mpo = build_dmrg_mpo(rhf_result, mp2_result; integral_cutoff=1e-12)
        hf_mps = hartree_fock_mps(product_mpo.nspin, product_mpo.n_elec)
        @test product_mpo isa ProductMPO
        @test product_mpo.nspin == nspin
        @test product_mpo.n_elec == rhf_result.n_elec
        @test length(product_mpo.terms) > 0
        @test isapprox(mps_particle_number(hf_mps), rhf_result.n_elec; atol=1e-12)
        @test isapprox(product_mpo_expectation(product_mpo, hf_mps),
                        fci_result.hamiltonian[1,1]; atol=1e-10)

        theta = two_site_tensor(dmrg_result.tensors, dmrg_result.two_site_site)
        sigma = two_site_tensor_contract(dmrg_result.mpo, dmrg_result.tensors,
                                         dmrg_result.two_site_site, theta)
        projection = two_site_projection_matrix(dmrg_result.tensors,
                                                fci_result.determinants,
                                                dmrg_result.two_site_site)
        @test size(theta) == size(sigma)
        @test isapprox(projection * vec(theta), fci_result.ci_vector; atol=1e-12)
        @test isapprox(two_site_energy(dmrg_result.mpo, dmrg_result.tensors,
                                       dmrg_result.two_site_site, theta),
                        fci_result.electronic_energy; atol=1e-10)

        split = split_two_site_tensor(theta; cutoff=1e-12)
        replaced = replace_two_site_tensor(dmrg_result.tensors, dmrg_result.two_site_site,
                                           theta; cutoff=1e-12)
        @test split.discarded_weight < 1e-20
        @test replaced.discarded_weight < 1e-20
        @test isapprox(mps_reconstruct(replaced.tensors), dmrg_result.fock_state; atol=1e-12)

        compressed = run_fci_dmrg(rhf_result, mp2_result; fci_result=fci_result,
                                  max_bond=8, verbose=false)
        @test compressed.max_bond_dimension == 8
        @test all(compressed.bond_dimensions .<= 8)
        @test compressed.discarded_weight > 0.0
        @test compressed.total_energy > fci_result.total_energy
        @test compressed.total_energy < rhf_result.total_energy
    end

    @testset "T1 and T2 amplitude shapes" begin
        o = rhf_result.n_elec
        v = 2*rhf_result.nbasis - o
        @test size(ccsd_result.ts) == (o, v)
        @test size(ccsd_result.td) == (o, o, v, v)
    end

    @testset "T2 antisymmetry" begin
        td = ccsd_result.td
        o, _, v, _ = size(td)
        # t_{ij}^{ab} = -t_{ji}^{ab} = -t_{ij}^{ba}
        for i in 1:min(o,3), j in 1:min(o,3), a in 1:min(v,3), b in 1:min(v,3)
            @test isapprox(td[i,j,a,b], -td[j,i,a,b]; atol=1e-10)
            @test isapprox(td[i,j,a,b], -td[i,j,b,a]; atol=1e-10)
        end
    end

end
