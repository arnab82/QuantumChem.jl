# Project #7: Connecting Code to PSI4

Status: skipped in this Julia repo.

## Context

Crawford Project #7 focuses on connecting earlier code to PSI4.  This package
uses PySCF for molecular integrals instead, so there is no PSI4 bridge here.

## Local Replacement

The PySCF-backed path lives in `src/rhf.jl`:

- `make_molecule`
- `pyscf_1e`
- `pyscf_overlap`
- `pyscf_2e`
- `pyscf_nucr`

These functions provide the one-electron integrals, overlap, two-electron
integrals, and nuclear repulsion used by the RHF, MP2, CCSD, and excited-state
code.

## Why This Is Acceptable Here

The later projects need reliable integral generation, not PSI4 specifically.
Using PySCF keeps the workflow inside Julia plus PyCall and avoids requiring a
separate PSI4 installation.

