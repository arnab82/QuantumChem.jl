# Project #28: CIS and EOM-CCSD Response Properties

Code: `src/response_properties.jl`

Tests: `test/runtests.jl`, testset `Project #28 CIS and EOM-CCSD properties`

## Goal

Compute electric-dipole transition properties for excited states already
available in the package:

- spin-adapted singlet CIS transition dipoles and oscillator strengths
- EOM-CCSD right-vector transition dipoles and oscillator strengths
- RHF permanent dipole moments used as a ground-state property check

The Crawford future-project list mentions response properties, including
Hartree-Fock and coupled-cluster dipole polarizabilities.  This local project
starts with the excited-state property layer the repo can support directly from
its CIS and EOM-CCSD solvers.

## Dipole Integrals

PySCF supplies AO position integrals:

```text
r_alpha[mu,nu] = <chi_mu | alpha | chi_nu>,  alpha in {x,y,z}.
```

The spatial MO transformation is

```text
r_alpha[p,q] = sum_munu C_mup r_alpha[mu,nu] C_nuq.
```

The RHF permanent dipole is

```text
mu = sum_A Z_A R_A - 2 sum_munu D_munu <chi_mu|r|chi_nu>.
```

## CIS Transition Properties

For a normalized spin-adapted singlet CIS vector `C_ia^k`,

```text
mu_0k = -sqrt(2) sum_ia C_ia^k <i|r|a>.
```

The factor `sqrt(2)` comes from the normalized alpha-plus-beta singlet
excitation.  The oscillator strength is

```text
f_k = (2/3) omega_k |mu_0k|^2.
```

Use:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
excited = run_excited_states(rhf, mp2)
props = run_cis_properties(rhf, excited; nroots=5)
```

## EOM-CCSD Transition Properties

The local EOM-CCSD implementation stores right eigenvectors in the packed
singles plus unique-doubles space.  The property helper uses the singles block:

```text
mu_0k approx -sum_ia R_i^a(k) <i|r|a>.
```

This is intentionally an educational right-vector property.  A production
coupled-cluster response implementation would also build left eigenvectors and
similarity-transformed property operators.

Use:

```julia
ccsd = run_ccsd(rhf, mp2; diis=true)
eom = run_eom_ccsd(rhf, mp2, ccsd; nroots=5)
props = run_eom_ccsd_properties(rhf, eom; nroots=5)
```

## Reference Checks

For default STO-3G water, the RHF dipole moment in atomic units is:

```text
mu = (0.0, 0.603521296526, 0.0)
```

The first five singlet CIS oscillator strengths are:

```text
0.002341274012
0.000000000000
0.064926273239
0.015467355438
1.251936910632
```

The fourth local EOM-CCSD right-vector oscillator strength is:

```text
0.001950682062
```
