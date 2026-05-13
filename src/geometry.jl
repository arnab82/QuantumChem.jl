"""
Molecular geometry analysis for Crawford Programming Project #1.

Coordinates are in bohr and atom labels are atomic numbers.
"""

using LinearAlgebra

const BOHR_TO_ANGSTROM = 0.529177210903
const AMU_TO_GRAM = 1.66053906660e-24
const AMU_TO_KG = 1.66053906660e-27
const BOHR_TO_CM = BOHR_TO_ANGSTROM * 1e-8
const BOHR_TO_METER = BOHR_TO_ANGSTROM * 1e-10
const PLANCK_J_S = 6.62607015e-34
const SPEED_OF_LIGHT_CM_S = 2.99792458e10

const ATOMIC_MASSES = Dict(
    1  => 1.00782503223,
    2  => 4.00260325413,
    3  => 7.0160034366,
    4  => 9.012183065,
    5  => 11.00930536,
    6  => 12.0,
    7  => 14.00307400443,
    8  => 15.99491461957,
    9  => 18.99840316273,
    10 => 19.9924401762,
    11 => 22.9897692820,
    12 => 23.985041697,
    13 => 26.98153853,
    14 => 27.97692653465,
    15 => 30.97376199842,
    16 => 31.9720711744,
    17 => 34.968852682,
    18 => 39.9623831237,
)

"""
    Molecule

Simple molecular geometry container for the early Crawford projects.

Fields:
- `atomic_numbers`: nuclear charges `Z_A`.
- `coordinates`: Cartesian coordinates in Bohr with shape `natom x 3`.
"""
struct Molecule
    atomic_numbers::Vector{Int}
    coordinates::Matrix{Float64}
end

Base.length(mol::Molecule) = length(mol.atomic_numbers)

"""
    atomic_mass(z) -> Float64

Return the standard atomic mass in atomic mass units for atomic number `z`.
"""
atomic_mass(z::Integer) = get(ATOMIC_MASSES, Int(z)) do
    throw(ArgumentError("No atomic mass is available for atomic number $z"))
end

"""
    atomic_masses(mol) -> Vector{Float64}

Return one atomic mass per atom in `mol`, in amu.
"""
atomic_masses(mol::Molecule) = [atomic_mass(z) for z in mol.atomic_numbers]

"""
    read_molecule(io) -> Molecule
    read_molecule(text) -> Molecule

Read a simple Crawford-style geometry block:

```text
natoms
Z1 x1 y1 z1
Z2 x2 y2 z2
...
```

Coordinates are interpreted as Bohr and atomic labels as atomic numbers.
"""
function read_molecule(io::IO)
    tokens = split(read(io, String))
    isempty(tokens) && throw(ArgumentError("Molecule input is empty"))

    natoms = parse(Int, tokens[1])
    expected = 1 + 4 * natoms
    length(tokens) < expected && throw(ArgumentError("Molecule input has fewer than $natoms atoms"))

    atomic_numbers = Vector{Int}(undef, natoms)
    coordinates = Matrix{Float64}(undef, natoms, 3)
    offset = 2
    for atom in 1:natoms
        atomic_numbers[atom] = round(Int, parse(Float64, tokens[offset]))
        coordinates[atom,1] = parse(Float64, tokens[offset + 1])
        coordinates[atom,2] = parse(Float64, tokens[offset + 2])
        coordinates[atom,3] = parse(Float64, tokens[offset + 3])
        offset += 4
    end

    return Molecule(atomic_numbers, coordinates)
end

read_molecule(text::AbstractString) = read_molecule(IOBuffer(text))

"""
    distance(mol, i, j) -> Float64

Compute the interatomic distance

```text
r_ij = |R_i - R_j|
```

in Bohr.
"""
function distance(mol::Molecule, i::Integer, j::Integer)
    return norm(@view(mol.coordinates[Int(i),:]) - @view(mol.coordinates[Int(j),:]))
end

"""
    distance_matrix(mol) -> Matrix{Float64}

Return the symmetric matrix of all pairwise distances `r_ij = |R_i - R_j|`.
"""
function distance_matrix(mol::Molecule)
    natoms = length(mol)
    distances = zeros(Float64, natoms, natoms)
    for i in 1:natoms, j in 1:(i-1)
        distances[i,j] = distances[j,i] = distance(mol, i, j)
    end
    return distances
end

"""
    bond_angle(mol, i, j, k) -> Float64

Return the angle `i-j-k` in degrees using

```text
theta = acos( ((R_i - R_j) dot (R_k - R_j)) /
              (|R_i - R_j| |R_k - R_j|) )
```
"""
function bond_angle(mol::Molecule, i::Integer, j::Integer, k::Integer)
    rij = @view(mol.coordinates[Int(i),:]) - @view(mol.coordinates[Int(j),:])
    rkj = @view(mol.coordinates[Int(k),:]) - @view(mol.coordinates[Int(j),:])
    cosine = clamp(dot(rij, rkj) / (norm(rij) * norm(rkj)), -1.0, 1.0)
    return acosd(cosine)
end

"""
    out_of_plane_angle(mol, i, j, k, l) -> Float64

Return the out-of-plane angle in degrees for atom `i` relative to the plane
formed by `j-k-l`.
"""
function out_of_plane_angle(mol::Molecule, i::Integer, j::Integer, k::Integer, l::Integer)
    rki = @view(mol.coordinates[Int(i),:]) - @view(mol.coordinates[Int(k),:])
    rkj = @view(mol.coordinates[Int(j),:]) - @view(mol.coordinates[Int(k),:])
    rkl = @view(mol.coordinates[Int(l),:]) - @view(mol.coordinates[Int(k),:])
    normal = cross(rkj, rkl)
    sine = clamp(dot(rki, normal) / (norm(rki) * norm(normal)), -1.0, 1.0)
    return asind(sine)
end

"""
    torsion_angle(mol, i, j, k, l) -> Float64

Return the signed dihedral angle in degrees between the planes `i-j-k` and
`j-k-l`.
"""
function torsion_angle(mol::Molecule, i::Integer, j::Integer, k::Integer, l::Integer)
    r1 = @view mol.coordinates[Int(i),:]
    r2 = @view mol.coordinates[Int(j),:]
    r3 = @view mol.coordinates[Int(k),:]
    r4 = @view mol.coordinates[Int(l),:]

    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    m1 = cross(n1, b2 / norm(b2))
    return atan(dot(m1, n2), dot(n1, n2)) * 180.0 / pi
end

"""
    center_of_mass(mol) -> Vector{Float64}

Compute the center of mass

```text
R_cm = (sum_A m_A R_A) / (sum_A m_A)
```

in Bohr.
"""
function center_of_mass(mol::Molecule)
    masses = atomic_masses(mol)
    total_mass = sum(masses)
    center = zeros(Float64, 3)
    for atom in 1:length(mol)
        center .+= masses[atom] .* @view(mol.coordinates[atom,:])
    end
    return center ./ total_mass
end

"""
    translate_to_center_of_mass(mol) -> Molecule

Return a copy of `mol` translated so that `center_of_mass(mol)` is at the
origin.
"""
function translate_to_center_of_mass(mol::Molecule)
    center = center_of_mass(mol)
    return Molecule(copy(mol.atomic_numbers), mol.coordinates .- center')
end

"""
    inertia_tensor(mol; center=true) -> Matrix{Float64}

Compute the Cartesian inertia tensor in amu bohr^2:

```text
I_xx = sum_A m_A (y_A^2 + z_A^2)
I_xy = -sum_A m_A x_A y_A
```

and cyclic permutations.  With `center=true`, coordinates are first shifted to
the center of mass.
"""
function inertia_tensor(mol::Molecule; center=true)
    working = center ? translate_to_center_of_mass(mol) : mol
    masses = atomic_masses(working)
    tensor = zeros(Float64, 3, 3)

    for atom in 1:length(working)
        mass = masses[atom]
        x, y, z = working.coordinates[atom, :]
        tensor[1,1] += mass * (y^2 + z^2)
        tensor[2,2] += mass * (x^2 + z^2)
        tensor[3,3] += mass * (x^2 + y^2)
        tensor[1,2] -= mass * x * y
        tensor[1,3] -= mass * x * z
        tensor[2,3] -= mass * y * z
    end

    tensor[2,1] = tensor[1,2]
    tensor[3,1] = tensor[1,3]
    tensor[3,2] = tensor[2,3]
    return tensor
end

"""
    principal_moments(mol; center=true)

Return the sorted eigenvalues of the inertia tensor.
"""
principal_moments(mol::Molecule; center=true) = sort(eigvals(Symmetric(inertia_tensor(mol; center))))

"""
    rotor_type(moments; atol=1e-8, rtol=1e-5) -> Symbol

Classify a molecule from its principal moments as `:linear`, `:spherical`,
`:oblate`, `:prolate`, or `:asymmetric`.
"""
function rotor_type(moments; atol=1e-8, rtol=1e-5)
    Ia, Ib, Ic = sort(collect(moments))
    if Ia < atol
        return :linear
    elseif isapprox(Ia, Ib; rtol, atol) && isapprox(Ib, Ic; rtol, atol)
        return :spherical
    elseif isapprox(Ia, Ib; rtol, atol)
        return :oblate
    elseif isapprox(Ib, Ic; rtol, atol)
        return :prolate
    else
        return :asymmetric
    end
end

"""
    rotational_constants(moments) -> (MHz, cm)

Convert principal moments to rotational constants:

```text
B_i = h / (8 pi^2 I_i)
```

reported in MHz and cm^-1.
"""
function rotational_constants(moments)
    sorted_moments = sort(collect(moments))
    mhz = similar(sorted_moments, Float64)
    wavenumbers = similar(sorted_moments, Float64)

    for i in eachindex(sorted_moments)
        moment = sorted_moments[i]
        if iszero(moment)
            mhz[i] = Inf
            wavenumbers[i] = Inf
        else
            inertia_si = moment * AMU_TO_KG * BOHR_TO_METER^2
            hz = PLANCK_J_S / (8 * pi^2 * inertia_si)
            mhz[i] = hz / 1e6
            wavenumbers[i] = hz / SPEED_OF_LIGHT_CM_S
        end
    end

    return (MHz=mhz, cm= wavenumbers)
end

"""Convert moments from amu bohr^2 to amu angstrom^2."""
moments_amu_angstrom2(moments) = collect(moments) .* BOHR_TO_ANGSTROM^2

"""Convert moments from amu bohr^2 to g cm^2."""
moments_g_cm2(moments) = collect(moments) .* AMU_TO_GRAM .* BOHR_TO_CM^2

"""
    run_geometry_analysis(mol) -> NamedTuple

Compute the Project #1 geometry summary: distance matrix, center of mass,
inertia tensor, principal moments, rotor classification, and rotational
constants.
"""
function run_geometry_analysis(mol::Molecule)
    moments = principal_moments(mol)
    return (
        distances = distance_matrix(mol),
        center_of_mass = center_of_mass(mol),
        inertia_tensor = inertia_tensor(mol),
        principal_moments = moments,
        principal_moments_amu_angstrom2 = moments_amu_angstrom2(moments),
        principal_moments_g_cm2 = moments_g_cm2(moments),
        rotor_type = rotor_type(moments),
        rotational_constants = rotational_constants(moments),
    )
end

run_geometry_analysis(text::AbstractString) = run_geometry_analysis(read_molecule(text))
