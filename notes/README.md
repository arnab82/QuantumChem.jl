# Project Notes

These notes summarize the Crawford programming projects as implemented in this
Julia package.  They are meant as study notes plus a map from each project to
the local code and tests.

| Project | Topic | Note |
| ------- | ----- | ---- |
| #1 | Molecular geometry | [project-01-geometry.md](project-01-geometry.md) |
| #2 | Harmonic vibrations | [project-02-vibrations.md](project-02-vibrations.md) |
| #3 | RHF SCF | [project-03-rhf.md](project-03-rhf.md) |
| #4 | MP2 | [project-04-mp2.md](project-04-mp2.md) |
| #5 | CCSD | [project-05-ccsd.md](project-05-ccsd.md) |
| #6 | CCSD(T) | [project-06-ccsd-t.md](project-06-ccsd-t.md) |
| #7 | PSI4 interface | [project-07-psi4.md](project-07-psi4.md) |
| #8 | SCF DIIS | [project-08-scf-diis.md](project-08-scf-diis.md) |
| #9 | SCF symmetry | [project-09-symmetry.md](project-09-symmetry.md) |
| #10 | CC amplitude DIIS | [project-10-cc-diis.md](project-10-cc-diis.md) |
| #11 | Out-of-core SCF | [project-11-outcore-scf.md](project-11-outcore-scf.md) |
| #12 | CIS and TDHF/RPA | [project-12-cis-tdhf.md](project-12-cis-tdhf.md) |
| #13 | Davidson-Liu CIS | [project-13-davidson-cis.md](project-13-davidson-cis.md) |
| #14 | EOM-CCSD | [project-14-eom-ccsd.md](project-14-eom-ccsd.md) |
| #15 | UHF | [project-15-uhf.md](project-15-uhf.md) |
| #16 | ROHF | [project-16-rohf.md](project-16-rohf.md) |
| #17 | UCCSD | [project-17-uccsd.md](project-17-uccsd.md) |
| #18 | UMP2 | [project-18-ump2.md](project-18-ump2.md) |
| #19 | CISD | [project-19-cisd.md](project-19-cisd.md) |
| #20 | CCSDT | [project-20-ccsdt.md](project-20-ccsdt.md) |
| #21 | HCI | [project-21-hci.md](project-21-hci.md) |
| #22 | FCI | [project-22-fci.md](project-22-fci.md) |
| #23 | DMRG/MPS | [project-23-dmrg.md](project-23-dmrg.md) |
| #24 | RHF analytic gradients | [project-24-rhf-gradients.md](project-24-rhf-gradients.md) |
| #25 | RHF geometry optimization | [project-25-rhf-geometry-optimization.md](project-25-rhf-geometry-optimization.md) |
| #26 | Integral-direct RHF | [project-26-integral-direct-rhf.md](project-26-integral-direct-rhf.md) |
| #27 | Density fitting / RI-MP2 | [project-27-density-fitting.md](project-27-density-fitting.md) |
| #28 | CIS/EOM-CCSD response properties | [project-28-response-properties.md](project-28-response-properties.md) |

General package entry point: `src/QuantumChem.jl`.

Run all checks:

```shell
julia --project=. -e 'using Pkg; Pkg.test()'
julia --threads=4 --project=. -e 'using Pkg; Pkg.test()'
```
