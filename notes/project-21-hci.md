# Project #21: Heat-Bath Selected CI

Code: `src/selected_ci.jl`

Tests: `test/runtests.jl`, testset `Project #21 heat-bath selected CI`

## Status of Crawford Spec

The public Crawford ProgrammingProjects repository currently lists projects
through #14.  This repo follows the local roadmap entry `#21 SCI algorithms:
HCI`.

## Goal

Build a compact variational CI space by selecting only determinants that couple
strongly to the current CI wavefunction:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
hci = run_hci(rhf, mp2; epsilon1=1e-3, epsilon2=0.0)
```

## Selection Rule

The variational space starts with the Hartree-Fock determinant.  After
diagonalizing the current determinant-space Hamiltonian, every connected
external single or double excitation `a` is tested against each variational
determinant `i`:

```text
abs(H_ai * c_i) > epsilon1
```

If the test passes for any `i`, determinant `a` is added to the variational
space.  The process repeats until no new determinants pass the threshold.

## Hamiltonian Elements

The selected-CI code reuses the bitstring determinant machinery from CISD:

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r
```

For selection and PT2, the code evaluates only determinant pairs that differ by
zero, one, or two spin orbitals.  This gives the same matrix elements as the
older brute-force CISD Hamiltonian builder, but is much cheaper for sparse
selected spaces.

## PT2 Correction

After the variational HCI space converges, the deterministic Epstein-Nesbet
second-order correction is:

```text
E2 = sum_a (sum_i H_ai c_i)^2 / (E0 - H_aa)
```

The `epsilon2` parameter can screen the PT2 sum.  The default `epsilon2=0.0`
uses all connected external determinants.

## Reference

For default STO-3G water with `epsilon1=1e-3` and `epsilon2=0.0`:

```text
iterations                = 4
selected determinants     = 101
added per iteration       = [40, 54, 6, 0]
external PT2 determinants = 32
E_HCI corr                = -0.07089840260761093 Eh
E_PT2                     = -0.00000188725766157 Eh
E_HCI+PT2 total           = -75.01298021805813 Eh
```

With `epsilon1=0.0`, the selected space expands to the full connected symmetry
sector for this reference and gives zero PT2 correction.
