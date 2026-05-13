# Project #12: CIS and TDHF/RPA

Code: `src/excited_states.jl`

Tests: `test/runtests.jl`, testset `Project #12 CIS and TDHF/RPA`

## Goal

Compute excited states from RHF orbitals using:

- spin-orbital CIS
- spin-adapted singlet CIS
- spin-adapted triplet CIS
- TDHF/RPA using the full matrix
- TDHF/RPA using the reduced squared-energy equation

```julia
excited = run_excited_states(rhf, mp2)
```

## CIS Matrix

Spin-orbital CIS:

```text
A_ia,jb = delta_ij delta_ab (eps_a - eps_i) + <aj||ib>
```

Spin-adapted singlet CIS:

```text
H_ia,jb = delta_ij delta_ab (eps_a - eps_i) + 2(ai|jb) - (ab|ji)
```

Spin-adapted triplet CIS:

```text
H_ia,jb = delta_ij delta_ab (eps_a - eps_i) - (ab|ji)
```

## RPA

The full RPA matrix is:

```text
[ A   B]
[-B  -A]
```

The reduced form solves:

```text
(A - B)(A + B)(X + Y) = omega^2 (X + Y)
```

## Reference Checks

The tests compare the first STO-3G water CIS and RPA roots to Crawford Project
#12 reference values and verify that full and reduced RPA produce the same
positive excitation energies.

