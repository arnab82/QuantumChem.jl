# Project #1: Molecular Geometry

Code: `src/geometry.jl`

Tests: `test/runtests.jl`, testset `Project #1 molecular geometry`

## Goal

Read a Cartesian molecular geometry and compute basic internal-coordinate and
rigid-rotor quantities:

- interatomic distances
- bond angles
- out-of-plane angles
- torsion angles
- center of mass
- inertia tensor
- principal moments
- rotor type
- rotational constants

## Main API

```julia
mol = read_molecule(geom_string)
distance(mol, 1, 2)
bond_angle(mol, 1, 2, 3)
torsion_angle(mol, 1, 2, 3, 4)
run_geometry_analysis(mol)
```

## Implementation Notes

The molecule is represented as atoms with masses and Cartesian coordinates.
Distances and angles are vector operations.  The center of mass is the
mass-weighted average of coordinates.  The inertia tensor is diagonalized to
obtain principal moments, which are then converted into rotational constants.

The test case uses acetaldehyde geometry and checks both internal coordinates
and rotational constants.

