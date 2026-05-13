# Project #4: MP2

Code: `src/mp2.jl`

Tests: `test/runtests.jl`, testsets `MP2` and `MP2 in SO basis`

## Goal

Compute second-order Moller-Plesset correlation energy from RHF orbitals:

```julia
rhf = run_rhf()
mp2 = run_mp2(rhf)
```

## Main Steps

1. Transform AO electron-repulsion integrals into the MO basis:

   ```text
   (pq|rs) = C_mu,p C_nu,q (mu nu | lambda sigma) C_lambda,r C_sigma,s
   ```

2. Sum over occupied `i,j` and virtual `a,b` spatial orbitals:

   ```text
   E_MP2 = sum_ijab (ia|jb) * (2(ia|jb) - (ib|ja))
            / (eps_i + eps_j - eps_a - eps_b)
   ```

## Reference

For STO-3G water, the MP2 correlation energy is:

```text
-0.049149636120 Eh
```

