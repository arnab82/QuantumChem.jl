"""
Harmonic vibrational analysis for Crawford Programming Project #2.
"""

using LinearAlgebra

const HARTREE_TO_J = 4.3597447222071e-18
const VIBRATION_CONVERSION_CM = sqrt(HARTREE_TO_J / (AMU_TO_KG * BOHR_TO_METER^2)) /
                                (2 * pi * SPEED_OF_LIGHT_CM_S)

"""
    read_hessian(io) -> (natoms, hessian)
    read_hessian(text) -> (natoms, hessian)

Read a Crawford-style Cartesian Hessian block.  The first token is `natoms`,
followed by `(3 natoms)^2` Hessian elements.
"""
function read_hessian(io::IO)
    tokens = split(read(io, String))
    isempty(tokens) && throw(ArgumentError("Hessian input is empty"))

    natoms = parse(Int, tokens[1])
    dim = 3 * natoms
    values = parse.(Float64, tokens[2:end])
    length(values) == dim^2 || throw(ArgumentError("Expected $(dim^2) Hessian values, got $(length(values))"))

    return (natoms=natoms, hessian=permutedims(reshape(values, dim, dim)))
end

read_hessian(text::AbstractString) = read_hessian(IOBuffer(text))

"""
    mass_weight_hessian(hessian, mol) -> Matrix{Float64}

Mass-weight the Cartesian Hessian:

```text
F^mw_{Ai,Bj} = F_{Ai,Bj} / sqrt(m_A m_B)
```

where `A,B` are atom indices and `i,j` are Cartesian components.
"""
function mass_weight_hessian(hessian, mol::Molecule)
    natoms = length(mol)
    size(hessian) == (3*natoms, 3*natoms) ||
        throw(ArgumentError("Hessian size does not match molecule size"))

    masses = atomic_masses(mol)
    coordinate_masses = repeat(masses, inner=3)
    weighted = similar(hessian, Float64)

    for i in axes(hessian, 1), j in axes(hessian, 2)
        weighted[i,j] = hessian[i,j] / sqrt(coordinate_masses[i] * coordinate_masses[j])
    end
    return weighted
end

"""
    harmonic_frequencies(eigenvalues) -> Vector{Float64}

Convert mass-weighted Hessian eigenvalues to harmonic frequencies in cm^-1:

```text
nu_k = sign(lambda_k) sqrt(|lambda_k|) / (2 pi c)
```

with the unit conversion from Hartree/(amu bohr^2) built into
`VIBRATION_CONVERSION_CM`.
"""
function harmonic_frequencies(eigenvalues)
    return [lambda < 0 ? -sqrt(abs(lambda)) * VIBRATION_CONVERSION_CM :
           sqrt(lambda) * VIBRATION_CONVERSION_CM for lambda in eigenvalues]
end

"""
    run_vibrational_analysis(mol, hessian) -> NamedTuple

Diagonalize the mass-weighted Hessian and return the sorted eigenvalues and
harmonic frequencies.
"""
function run_vibrational_analysis(mol::Molecule, hessian)
    weighted = mass_weight_hessian(hessian, mol)
    eigenvalues = eigvals(Symmetric(weighted))
    frequencies = harmonic_frequencies(eigenvalues)
    order = sortperm(frequencies; rev=true)

    return (
        mass_weighted_hessian = weighted,
        eigenvalues = eigenvalues[order],
        frequencies = frequencies[order],
    )
end

function run_vibrational_analysis(geom_text::AbstractString, hessian_text::AbstractString)
    mol = read_molecule(geom_text)
    hessian_data = read_hessian(hessian_text)
    hessian_data.natoms == length(mol) ||
        throw(ArgumentError("Geometry and Hessian atom counts do not match"))
    return run_vibrational_analysis(mol, hessian_data.hessian)
end
