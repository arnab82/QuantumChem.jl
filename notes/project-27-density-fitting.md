# Project #27: Density Fitting / RI-MP2

Code: `src/density_fitting.jl`

Tests: `test/runtests.jl`, testset `Project #27 density-fitted MP2`

## Goal

Density fitting, also called resolution of the identity, replaces the full
four-center electron-repulsion integral tensor with contractions over an
auxiliary basis.  For MP2 this is useful because the expensive MO integral
access pattern can be evaluated from three-index factors.

## Equations

PySCF supplies AO three-center integrals and the auxiliary Coulomb metric:

```text
(mu nu | P)
J_PQ = (P | Q)
```

The metric is orthonormalized by diagonalization:

```text
J = U diag(j) U^T
J^(-1/2) = U diag(j^(-1/2)) U^T
```

The orthonormal AO factors are

```text
B_munu^P = sum_Q (mu nu | Q) (J^(-1/2))_QP
```

and are transformed to the MO basis as

```text
B_pq^P = sum_munu C_mup C_nuq B_munu^P.
```

The density-fitted approximation to an MO ERI is then

```text
(pq | rs)_DF = sum_P B_pq^P B_rs^P.
```

The RI-MP2 correlation energy is computed directly from these factors:

```text
E_RI-MP2 =
sum_ijab (ia|jb)_DF [2 (ia|jb)_DF - (ib|ja)_DF]
/ (eps_i + eps_j - eps_a - eps_b).
```

## Implementation Notes

`df_ao_factors` builds the PySCF auxiliary molecule, reads `(mu nu | P)` and
`(P|Q)`, and returns a `DensityFitInfo` object.  `df_mo_factors` performs the
AO-to-MO transformation with `TensorOperations.jl`.  `compute_df_mp2` loops over
occupied and virtual orbital combinations and uses threaded dot products over
the auxiliary dimension instead of building the full four-index MO ERI tensor.

`run_df_mp2(rhf; auxbasis="weigend")` is the public driver.  Set
`return_eri=true` for small validation calculations where the approximate
four-index tensor is useful.

## Reference Check

For the default STO-3G water system with the `weigend` auxiliary basis:

```text
E_RI-MP2 corr = -0.049090437203 Eh
```

The conventional MP2 correlation energy for the same orbital basis is
`-0.049149636120 Eh`, so the small difference is the expected auxiliary-basis
approximation error.
