# QuantumChem.jl

Quantum chemistry methods (geometry analysis → vibrations → RHF/ROHF/UHF/direct SCF → RHF gradients/geometry optimization → MP2/UMP2 → CCSD/UCCSD → CCSD(T)/CCSDT → CIS/CISD/HCI/FCI/DMRG/TDHF → Davidson CIS → EOM-CCSD)
implemented in Julia,
using PySCF for integrals plus Einsum.jl and TensorOperations.jl for tensor contractions.

This repo is tracking the Crawford Group quantum chemistry programming projects
plus local follow-on inclusions:

| Crawford project | Topic | Status |
| ---------------- | ----- | ------ |
| #1 | Molecular geometry/rotational constants | Implemented via `src/geometry.jl` |
| #2 | Harmonic vibrational analysis | Implemented via `src/vibrations.jl` |
| #3 | RHF self-consistent field | Implemented via `src/rhf.jl` |
| #4 | MP2 energy | Implemented via `src/mp2.jl` |
| #5 | CCSD energy | Implemented via `src/ccsd.jl` |
| #6 | CCSD(T) triples correction | Implemented via `src/ccsd_t.jl` |
| #7 | Connecting code to PSI4 | Skipped |
| #8 | DIIS for SCF | Implemented via `src/rhf.jl` |
| #9 | Symmetry in SCF | Implemented via PySCF-backed `src/symmetry.jl` and `run_rhf(symmetry=true)` |
| #10 | DIIS for CC amplitudes | Implemented via `run_ccsd(...; diis=true)` |
| #11 | Out-of-core SCF | Implemented via disk-backed ERI batches and `run_rhf(outcore=true)` |
| #12 | CIS and TDHF/RPA | Implemented via `src/excited_states.jl` |
| #13 | Davidson-Liu CIS | Implemented via `run_davidson_cis(...)` |
| #14 | EOM-CCSD | Implemented via residual-Jacobian `run_eom_ccsd(...)` |
| #15 | Unrestricted Hartree-Fock | Implemented via `src/uhf.jl` and `run_uhf(...)` |
| #16 | Restricted open-shell Hartree-Fock | Implemented via `src/rohf.jl` and `run_rohf(...)` |
| #17 | Unrestricted CCSD | Implemented via `src/uccsd.jl` and `run_uccsd(...)` |
| #18 | Unrestricted MP2 | Implemented via `src/ump2.jl` and `run_ump2(...)` |
| #19 | Configuration interaction singles and doubles | Implemented via `src/cisd.jl` and `run_cisd(...)` |
| #20 | CCSDT | Implemented via Wicked-generated residual equations in `src/ccsdt.jl` |
| #21 | SCI algorithms: HCI | Implemented via `src/selected_ci.jl` and `run_hci(...)` |
| #22 | Full configuration interaction | Implemented via `src/fci.jl` and `run_fci(...)` |
| #23 | Density matrix renormalization group | Implemented via custom HF-basis MPS/MPO DMRG in `src/custom_dmrg.jl` plus exact small-system MPS references in `src/fci_dmrg.jl` |
| #24 | RHF analytic nuclear gradients | Implemented via `src/gradients.jl` and `run_rhf_gradient(...)` |
| #25 | RHF geometry optimization | Implemented via `src/optimization.jl` and `run_rhf_geometry_optimization(...)` |
| #26 | Integral-direct RHF | Implemented via `src/direct_scf.jl` and `run_rhf(direct=true)` |

## Inclusions

All local inclusions through #26 are implemented.

## Files

| File | Description |
| ---- | ----------- |
| `geometry.jl` | Molecular geometry, rotational constants, and inertia analysis |
| `vibrations.jl` | Harmonic vibrational analysis from Cartesian Hessians |
| `symmetry.jl` | Symmetry-orbital transforms and block-matrix utilities |
| `rhf.jl` | Restricted Hartree-Fock SCF |
| `gradients.jl` | RHF analytic nuclear gradients and force helpers |
| `optimization.jl` | RHF Cartesian geometry optimization with BFGS/backtracking |
| `outcore.jl` | Out-of-core unique AO ERI storage and Fock builds |
| `direct_scf.jl` | Integral-direct RHF J/K and Fock builds |
| `rohf.jl` | Restricted open-shell Hartree-Fock SCF |
| `uhf.jl` | Unrestricted Hartree-Fock SCF |
| `mp2.jl` | AO→MO integral transformation + MP2 energy |
| `ump2.jl` | UMP2 from unrestricted alpha/beta spin orbitals |
| `ccsd.jl` | CCSD in the antisymmetrized spin-orbital basis |
| `uccsd.jl` | UCCSD from unrestricted alpha/beta spin orbitals |
| `ccsd_t.jl` | Perturbative triples correction for CCSD(T) |
| `ccsdt.jl` | Full CCSDT amplitude equations generated from Wick&d |
| `excited_states.jl` | CIS, spin-adapted CIS, and TDHF/RPA excitation energies |
| `cisd.jl` | Determinant-space CISD Hamiltonian and diagonalization |
| `selected_ci.jl` | Heat-bath selected CI and deterministic PT2 correction |
| `fci.jl` | Full fixed-electron determinant-space CI |
| `custom_dmrg.jl` | Hartree-Fock-basis product-MPO DMRG and two-site sweep optimizer |
| `fci_dmrg.jl` | FCI-reference MPS helpers, determinant-space MPO contractions, and exact small-system DMRG checks |
| `eom_ccsd.jl` | EOM-CCSD residual Jacobian and excitation roots |

The package entry point is `src/QuantumChem.jl`; use `Pkg.test()` to run the
Crawford Project #1, #2, #5, #6, #8, #9, #10, #11, #12, #13, #14, #15, #16, #17, #18, #19, #20, #21, #22, #23, #24, #25, and #26 reference checks.
Per-project study notes live in [`notes/README.md`](notes/README.md).

## System

H₂O in STO-3G, geometry from Crawford Programming Project #5:

```text
O   0.000000000000  -0.143225816552   0.000000000000  (Bohr)
H   1.638036840407   1.136548822547   0.000000000000
H  -1.638036840407   1.136548822547   0.000000000000
```

## Reference energies (Crawford Projects #5 and #6)

| Quantity | Energy (Eh) |
| -------- | ----------- |
| E_HF | −74.942 079 928 192 |
| E_MP2 corr | −0.049 149 636 120 |
| E_CCSD corr | −0.070 680 088 376 |
| E_total (CCSD) | −75.012 760 016 568 |
| E(T) | −0.000 099 877 272 |
| E_total (CCSD(T)) | −75.012 859 893 840 |
| E_CCSDT corr | −0.070 812 807 698 |
| E_total (CCSDT) | −75.012 892 735 890 |
| E_HCI variational corr | −0.070 898 402 608 |
| E_HCI+PT2 total | −75.012 980 218 058 |
| E_FCI corr | −0.070 900 270 250 |
| E_total (FCI) | −75.012 980 198 443 |
| E_DMRG/MPS corr | −0.070 900 270 250 |
| E_total (DMRG/MPS) | −75.012 980 198 443 |

## Dependencies

```julia
using Pkg
Pkg.add(["Einsum", "PyCall", "TensorOperations"])
```

PySCF must be available in the Python environment used by PyCall.

For the threaded kernels, launch Julia with multiple threads:

```shell
julia --threads=auto --project=. -e 'using Pkg; Pkg.test()'
```

Symmetry-orbital SCF can use PySCF's detected point group or a requested
subgroup:

```julia
run_rhf(symmetry=true)   # PySCF-detected symmetry, e.g. C2v for default water
run_rhf(symmetry="Cs")   # force a PySCF-supported subgroup
run_rhf(symmetry="C1")   # no symmetry blocks beyond the full space
```

CCSD amplitude DIIS follows Crawford Project #10.  The default no-DIIS path is
unchanged, and DIIS can be enabled with an eight-vector subspace after three
iterations:

```julia
ccsd = run_ccsd(rhf, mp2; diis=true)
```

Project #11 can run RHF with a disk-backed file of permutationally unique AO
electron-repulsion integrals.  During each Fock build the file is streamed in
batches and the ERI symmetry-related Fock contributions are accumulated:

```julia
rhf = run_rhf(outcore=true)
```

Project #26 can run RHF without storing the AO ERI tensor at all.  Each SCF
iteration sends the current density to PySCF's direct J/K builder and assembles
the closed-shell Fock matrix as `F = h + 2J - K`:

```julia
rhf = run_rhf(direct=true)
direct = run_direct_rhf()
D = make_density(rhf.mo_coeffs, rhf.n_elec ÷ 2)
F = make_fock_direct(D, rhf.h1e, rhf.mol)
```

Project #12 excited states use the RHF canonical orbitals and MO ERIs:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
excited = run_excited_states(rhf, mp2)
excited.cis_energies       # spin-orbital CIS energies in Eh
excited.singlet_energies   # spin-adapted singlet CIS energies
excited.triplet_energies   # spin-adapted triplet CIS energies
excited.rpa_energies       # positive TDHF/RPA energies
```

Project #13 computes the lowest spin-adapted singlet CIS roots with the
Davidson-Liu simultaneous expansion algorithm:

```julia
davidson = run_davidson_cis(rhf, mp2; nroots=5)
davidson.energies          # lowest singlet CIS roots in Eh
davidson.residual_norms    # convergence diagnostics for each root
```

Project #14 is represented by a finite-difference EOM-CCSD Jacobian built from
the converged CCSD amplitude residual equations in the unique singles+doubles
spin-orbital excitation space:

```julia
ccsd = run_ccsd(rhf, mp2; diis=true)
eom = run_eom_ccsd(rhf, mp2, ccsd; nroots=5)
eom.energies               # EOM-CCSD excitation energies in Eh
```

Project #15 uses separate alpha and beta spin spaces for unrestricted
Hartree-Fock:

```julia
uhf = run_uhf()  # closed-shell default, agrees with RHF
oh = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
oh.spin_square
```

Project #16 keeps one shared spatial orbital set for restricted open-shell
Hartree-Fock:

```julia
rohf = run_rohf()  # closed-shell default, agrees with RHF
oh_rohf = run_rohf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
oh_rohf.occupations
```

Project #17 uses UHF alpha/beta orbitals as the reference for unrestricted
CCSD:

```julia
uhf = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
uccsd = run_uccsd(uhf)
uccsd.total_energy
```

Project #18 evaluates unrestricted MP2 from the same UHF spin-orbital blocks:

```julia
uhf = run_uhf(atoms="O 0 0 0; H 0 0 1.8", spin=1, unit="Bohr", diis=true)
ump2 = run_ump2(uhf)
ump2.emp2
```

Project #19 diagonalizes the determinant-space CISD Hamiltonian:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
cisd = run_cisd(rhf, mp2)
cisd.total_energy
```

Project #20 solves the full spin-orbital CCSDT equations.  The residual
contractions were generated with Wick&d and translated into Julia; the helper
script can regenerate the source equations from a built Wicked checkout:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
ccsdt = run_ccsdt(rhf, mp2)
ccsdt.total_energy
```

```shell
WICKED_PATH=/private/tmp/wicked /usr/bin/python3 tools/generate_ccsdt_wicked.py --summary
```

Project #21 builds a heat-bath selected CI variational space and applies a
deterministic Epstein-Nesbet PT2 correction over the connected external
determinants:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
hci = run_hci(rhf, mp2; epsilon1=1e-3, epsilon2=0.0)
hci.dimension
hci.total_energy
```

Project #22 diagonalizes the full fixed-electron determinant Hamiltonian in the
chosen spin-orbital basis:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
fci = run_fci(rhf, mp2)
fci.dimension
fci.total_energy
```

Project #23 runs a custom DMRG from the Hartree-Fock molecular-orbital basis.
The spin-orbital Hamiltonian is built from RHF one-electron integrals and MO
ERIs, converted to an in-repo Jordan-Wigner product-term MPO, and optimized as
an MPS without building the FCI determinant space:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
dmrg = run_dmrg(rhf, mp2; maxdim=[50, 100, 200, 400], cutoff=1e-8)
dmrg.max_bond_dimension
dmrg.total_energy
```

For small systems, `run_fci_dmrg` keeps the exact reference path that compresses
the FCI wavefunction as an occupation-number MPS and tests the explicit
two-site tensor contraction helpers:

```julia
fci = run_fci(rhf, mp2)
reference = run_fci_dmrg(rhf, mp2; fci_result=fci)
theta = two_site_tensor(reference.tensors, reference.two_site_site)
sigma = two_site_tensor_contract(reference.mpo, reference.tensors,
                                 reference.two_site_site, theta)

compressed = run_fci_dmrg(rhf, mp2; fci_result=fci, max_bond=8)
compressed.discarded_weight
```

Project #24 evaluates analytic RHF nuclear gradients in Eh/Bohr.  The code
uses PySCF derivative AO integrals and assembles the density, energy-weighted
density, Pulay overlap term, and nuclear repulsion derivative in Julia:

```julia
rhf = run_rhf()
grad = run_rhf_gradient(rhf)
grad.gradient
grad.max_force
```

Project #25 optimizes Cartesian coordinates using the RHF analytic gradient.
Each trial step rebuilds the RHF wavefunction, caps the largest Cartesian atom
step, and uses backtracking to keep the energy moving downhill:

```julia
rhf = run_rhf()
opt = run_rhf_geometry_optimization(rhf; maxiter=20)
opt.energies
opt.coordinates
opt.atoms
```

## H2O cc-pVDZ benchmark driver

The `examples/h2o_ccpvdz_benchmark.jl` script runs the water comparison you can
use for RHF, MP2, CCSD, CCSD(T), HCI, and custom DMRG.  Full-basis CCSDT and
dense CISD are guarded because they are intentionally expensive in this
educational implementation:

```shell
julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl
julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl --all
julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl --all --force-heavy
```

The DMRG defaults in that script are a quick exploratory run.  Increase
`--dmrg-maxdim`, `--dmrg-nsweeps`, and lower `--dmrg-integral-cutoff` for a
more serious calculation.

## CCSD implementation notes

Follows **Stanton et al., J. Chem. Phys. 94, 4334 (1991)**.
