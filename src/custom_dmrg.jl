"""
Custom Hartree-Fock-basis DMRG.

This file owns the production DMRG path: a Jordan-Wigner product-term MPO is
built directly from RHF molecular-orbital integrals, then optimized with
in-repo two-site MPS sweeps.  It does not enumerate FCI determinants.
"""

using LinearAlgebra, Printf

"""
    ProductMPOTerm

A single Jordan-Wigner product operator contribution to a product-term MPO:

```text
c * O_1 O_2 ... O_L.
```

Each local operator is a `2 x 2` matrix acting on the occupation basis
`|0>, |1>` of one spin orbital.
"""
struct ProductMPOTerm
    coefficient::Float64
    ops::Vector{Matrix{Float64}}
end

"""
    ProductMPO

Product-term molecular Hamiltonian used by the custom two-site DMRG sweeps:

```text
H = constant + sum_t c_t prod_site O_site^(t).
```

The original spin-orbital one- and two-electron tensors are cached for
inspection, while the sweep code contracts the local product operators.
"""
struct ProductMPO
    nspin::Int
    n_elec::Int
    terms::Vector{ProductMPOTerm}
    constant::Float64
    h_so::Matrix{Float64}
    aso::Array{Float64, 4}
end

const _DMRG_LOCAL_I = [1.0 0.0; 0.0 1.0]
const _DMRG_LOCAL_Z = [1.0 0.0; 0.0 -1.0]
const _DMRG_LOCAL_C = [0.0 1.0; 0.0 0.0]
const _DMRG_LOCAL_CDAG = [0.0 0.0; 1.0 0.0]
const _DMRG_LOCAL_N = [0.0 0.0; 0.0 1.0]

function _is_zero_local_operator(op)
    return all(abs.(op) .<= 10eps(Float64))
end

function _jw_product_ops(nspin::Integer, fermion_ops)
    ops = [copy(_DMRG_LOCAL_I) for _ in 1:nspin]

    for (kind, site) in fermion_ops
        1 <= site <= nspin ||
            throw(ArgumentError("Fermion operator site $site is outside 1:$nspin"))
        for parity_site in 1:(site - 1)
            ops[parity_site] = ops[parity_site] * _DMRG_LOCAL_Z
        end
        local_operator = kind === :create ? _DMRG_LOCAL_CDAG :
                         kind === :annihilate ? _DMRG_LOCAL_C :
                         throw(ArgumentError("Unknown fermion operator kind $kind"))
        ops[site] = ops[site] * local_operator
    end

    any(_is_zero_local_operator, ops) && return nothing
    return ops
end

function _number_product_term(nspin::Integer, site::Integer, coefficient::Real)
    ops = [copy(_DMRG_LOCAL_I) for _ in 1:nspin]
    ops[site] = copy(_DMRG_LOCAL_N)
    return ProductMPOTerm(Float64(coefficient), ops)
end

function _number_number_product_term(nspin::Integer, site1::Integer, site2::Integer,
                                     coefficient::Real)
    ops = [copy(_DMRG_LOCAL_I) for _ in 1:nspin]
    ops[site1] = copy(_DMRG_LOCAL_N)
    ops[site2] = copy(_DMRG_LOCAL_N)
    return ProductMPOTerm(Float64(coefficient), ops)
end

function _build_product_mpo_terms(h_so, aso; integral_cutoff=1e-12)
    nspin = size(h_so, 1)
    terms = ProductMPOTerm[]

    for q in 1:nspin, p in 1:nspin
        value = h_so[p,q]
        abs(value) <= integral_cutoff && continue
        ops = _jw_product_ops(nspin, ((:create, p), (:annihilate, q)))
        ops === nothing && continue
        push!(terms, ProductMPOTerm(value, ops))
    end

    for r in 1:nspin, s in 1:nspin, q in 1:nspin, p in 1:nspin
        value = 0.25 * aso[p,q,r,s]
        abs(value) <= integral_cutoff && continue
        ops = _jw_product_ops(nspin, ((:create, p), (:create, q),
                                      (:annihilate, s), (:annihilate, r)))
        ops === nothing && continue
        push!(terms, ProductMPOTerm(value, ops))
    end

    return terms
end

function _with_particle_penalty(mpo::ProductMPO, penalty::Real)
    penalty = Float64(penalty)
    penalty == 0.0 && return mpo

    terms = copy(mpo.terms)
    nspin = mpo.nspin
    n_elec = mpo.n_elec
    for site in 1:nspin
        push!(terms, _number_product_term(nspin, site, penalty * (1 - 2n_elec)))
    end
    for site1 in 1:(nspin - 1), site2 in (site1 + 1):nspin
        push!(terms, _number_number_product_term(nspin, site1, site2, 2penalty))
    end
    return ProductMPO(mpo.nspin, mpo.n_elec, terms,
                      mpo.constant + penalty * n_elec^2, mpo.h_so, mpo.aso)
end

"""
    build_product_dmrg_mpo(rhf, mp2_result; integral_cutoff=1e-12)

Build the Hartree-Fock molecular-orbital spin-orbital Hamiltonian as a custom
Jordan-Wigner product-term MPO.  No FCI determinant space is constructed.

```text
H = sum_pq h_pq a_p^+ a_q
  + 1/4 sum_pqrs <pq||rs> a_p^+ a_q^+ a_s a_r.
```

Each fermion string is mapped to qubit/occupation operators by the
Jordan-Wigner parity string.
"""
function build_product_dmrg_mpo(rhf, mp2_result; integral_cutoff=1e-12)
    h_mo = Mat_aotoMat_mo(rhf.mo_coeffs, rhf.h1e)
    h_so = spatial_to_spinorbital_1e(h_mo)
    aso = mo_to_aso(mp2_result.new_eri)
    terms = _build_product_mpo_terms(h_so, aso; integral_cutoff)
    return ProductMPO(size(h_so, 1), rhf.n_elec, terms, 0.0, h_so, aso)
end

"""
    hartree_fock_mps(nspin, n_elec) -> Vector{Array{Float64,3}}

Build the bond-dimension-one MPS for the Hartree-Fock occupation string:
spin orbitals `1:n_elec` are occupied and the remaining spin orbitals are
empty.
"""
function hartree_fock_mps(nspin::Integer, n_elec::Integer)
    0 <= n_elec <= nspin ||
        throw(ArgumentError("n_elec must be between 0 and nspin"))
    tensors = Array{Float64, 3}[]
    for site in 1:nspin
        tensor = zeros(Float64, 1, 2, 1)
        occ = site <= n_elec ? 2 : 1
        tensor[1, occ, 1] = 1.0
        push!(tensors, tensor)
    end
    return tensors
end

function _left_product_environment(tensors, ops, site::Integer)
    env = ones(Float64, 1, 1)
    for tensor_site in 1:(site - 1)
        A = tensors[tensor_site]
        O = ops[tensor_site]
        next_env = zeros(Float64, size(A, 3), size(A, 3))
        @inbounds for lb in 1:size(A, 1), lk in 1:size(A, 1),
                      rb in 1:size(A, 3), rk in 1:size(A, 3),
                      sb in 1:2, sk in 1:2
            next_env[rb, rk] += A[lb, sb, rb] * O[sb, sk] * env[lb, lk] * A[lk, sk, rk]
        end
        env = next_env
    end
    return env
end

function _right_product_environment(tensors, ops, site::Integer)
    env = ones(Float64, 1, 1)
    for tensor_site in length(tensors):-1:(site + 2)
        A = tensors[tensor_site]
        O = ops[tensor_site]
        next_env = zeros(Float64, size(A, 1), size(A, 1))
        @inbounds for lb in 1:size(A, 1), lk in 1:size(A, 1),
                      rb in 1:size(A, 3), rk in 1:size(A, 3),
                      sb in 1:2, sk in 1:2
            next_env[lb, lk] += A[lb, sb, rb] * O[sb, sk] * env[rb, rk] * A[lk, sk, rk]
        end
        env = next_env
    end
    return env
end

"""
    product_mpo_expectation(mpo, tensors) -> Float64

Evaluate an MPS expectation value for the product-term MPO:

```text
<H> = constant + sum_t c_t <Psi| prod_site O_site^(t) |Psi>.
```

The contraction walks left to right, accumulating the one-term transfer
environment for each product operator.
"""
function product_mpo_expectation(mpo::ProductMPO, tensors::AbstractVector)
    value = mpo.constant
    for term in mpo.terms
        env = ones(Float64, 1, 1)
        for site in 1:length(tensors)
            A = tensors[site]
            O = term.ops[site]
            next_env = zeros(Float64, size(A, 3), size(A, 3))
            @inbounds for lb in 1:size(A, 1), lk in 1:size(A, 1),
                          rb in 1:size(A, 3), rk in 1:size(A, 3),
                          sb in 1:2, sk in 1:2
                next_env[rb, rk] += A[lb, sb, rb] * O[sb, sk] * env[lb, lk] * A[lk, sk, rk]
            end
            env = next_env
        end
        value += term.coefficient * env[1, 1]
    end
    return value
end

"""
    mps_particle_number(tensors) -> Float64

Return `<N>` for an MPS by contracting a product-term number MPO:

```text
N = sum_p n_p.
```
"""
function mps_particle_number(tensors::AbstractVector)
    nspin = length(tensors)
    number_mpo = ProductMPO(
        nspin, 0,
        [_number_product_term(nspin, site, 1.0) for site in 1:nspin],
        0.0,
        zeros(Float64, nspin, nspin),
        zeros(Float64, nspin, nspin, nspin, nspin),
    )
    return product_mpo_expectation(number_mpo, tensors)
end

function _two_site_effective_matrix(mpo::ProductMPO, tensors::AbstractVector, site::Integer)
    site = _check_two_site(tensors, site)
    left_bond = size(tensors[site], 1)
    right_bond = size(tensors[site + 1], 3)
    dim = left_bond * 2 * 2 * right_bond
    H = zeros(Float64, dim, dim)

    for term in mpo.terms
        L = _left_product_environment(tensors, term.ops, site)
        R = _right_product_environment(tensors, term.ops, site)
        local_matrix = kron(R, term.ops[site + 1], term.ops[site], L)
        H .+= term.coefficient .* local_matrix
    end
    return Symmetric((H .+ H') ./ 2)
end

function _optimize_product_two_site!(tensors, mpo::ProductMPO, site::Integer;
                                     maxdim::Integer, cutoff=1e-10,
                                     direction=:right, noise=0.0)
    theta = two_site_tensor(tensors, site)
    H_eff = _two_site_effective_matrix(mpo, tensors, site)
    eig = eigen(H_eff)
    theta_opt = reshape(eig.vectors[:, 1], size(theta))
    if noise > 0.0
        @inbounds for idx in eachindex(theta_opt)
            theta_opt[idx] += noise * sin(0.61803398875 * idx + site)
        end
        theta_norm = norm(theta_opt)
        theta_norm > eps(Float64) && (theta_opt ./= theta_norm)
    end
    split = replace_two_site_tensor(tensors, site, theta_opt;
                                    max_bond=maxdim, cutoff, direction)
    tensors[site] = split.tensors[site]
    tensors[site + 1] = split.tensors[site + 1]
    return (local_energy=eig.values[1], discarded_weight=split.discarded_weight)
end

"""
    dmrg_sweep!(tensors, optimization_mpo; maxdim, cutoff=1e-10, noise=0.0)

Perform one left-to-right and one right-to-left two-site DMRG sweep.  Each local
step diagonalizes the two-site effective Hamiltonian,

```text
H_eff theta = E theta,
```

then truncates the optimized two-site tensor back to the requested MPS bond
dimension.
"""
function dmrg_sweep!(tensors, optimization_mpo::ProductMPO;
                     maxdim::Integer, cutoff=1e-10, noise=0.0)
    discarded = 0.0
    local_energy = 0.0
    for site in 1:(length(tensors) - 1)
        result = _optimize_product_two_site!(tensors, optimization_mpo, site;
                                             maxdim, cutoff, direction=:right, noise)
        discarded += result.discarded_weight
        local_energy = result.local_energy
    end
    for site in (length(tensors) - 1):-1:1
        result = _optimize_product_two_site!(tensors, optimization_mpo, site;
                                             maxdim, cutoff, direction=:left, noise)
        discarded += result.discarded_weight
        local_energy = result.local_energy
    end
    return (local_energy=local_energy, discarded_weight=discarded)
end

function _max_mps_bond_dimension(tensors)
    isempty(tensors) && return 1
    return maximum(size(tensor, 3) for tensor in tensors)
end

function _as_sweep_vector(value, nsweeps::Integer)
    if value isa Number
        return fill(value, Int(nsweeps))
    end
    collected = collect(value)
    isempty(collected) && throw(ArgumentError("Sweep parameter vectors cannot be empty"))
    return collected
end

"""
    build_dmrg_mpo(rhf, mp2_result; kwargs...)

Build the direct Hartree-Fock-basis custom product-term MPO used by `run_dmrg`.
"""
function build_dmrg_mpo(rhf, mp2_result; kwargs...)
    return build_product_dmrg_mpo(rhf, mp2_result; kwargs...)
end

"""
    run_dmrg(rhf, mp2_result; maxdim=(10, 20, 40), cutoff=1e-10, nsweeps=nothing)

Run DMRG directly in the Hartree-Fock molecular-orbital spin-orbital basis.
This custom implementation builds a Jordan-Wigner product-term MPO and performs
two-site MPS sweeps in this repo.  It does not build FCI determinants.
"""
function run_dmrg(rhf, mp2_result;
                  maxdim=(10, 20, 40),
                  cutoff=1e-10,
                  noise=(1e-4, 1e-5, 0.0),
                  nsweeps=nothing,
                  integral_cutoff=1e-10,
                  particle_penalty=100.0,
                  verbose=true)
    maxdim_values = collect(maxdim)
    isempty(maxdim_values) && throw(ArgumentError("maxdim cannot be empty"))
    nsweeps_value = isnothing(nsweeps) ? length(maxdim_values) : Int(nsweeps)
    nsweeps_value >= 1 || throw(ArgumentError("nsweeps must be positive"))
    noise_values = _as_sweep_vector(noise, nsweeps_value)

    physical_mpo = build_product_dmrg_mpo(rhf, mp2_result; integral_cutoff)
    optimization_mpo = _with_particle_penalty(physical_mpo, particle_penalty)
    tensors = hartree_fock_mps(physical_mpo.nspin, physical_mpo.n_elec)

    sweep_energies = Float64[]
    particle_numbers = Float64[]
    discarded_weights = Float64[]

    for sweep in 1:nsweeps_value
        maxdim_sweep = Int(maxdim_values[min(sweep, length(maxdim_values))])
        noise_sweep = Float64(noise_values[min(sweep, length(noise_values))])
        sweep_result = dmrg_sweep!(tensors, optimization_mpo;
                                   maxdim=maxdim_sweep, cutoff, noise=noise_sweep)
        electronic_energy = product_mpo_expectation(physical_mpo, tensors)
        particle_number = mps_particle_number(tensors)
        push!(sweep_energies, electronic_energy)
        push!(particle_numbers, particle_number)
        push!(discarded_weights, sweep_result.discarded_weight)

        verbose && @printf("DMRG sweep %3d  maxdim=%5d  E_elec=%20.12f  <N>=%12.8f  noise=%.1e  disc=%.3e\n",
                           sweep, maxdim_sweep, electronic_energy,
                           particle_number, noise_sweep, sweep_result.discarded_weight)
    end

    electronic_energy = sweep_energies[end]
    nuclear_repulsion = rhf.total_energy - rhf.energy
    total_energy = electronic_energy + nuclear_repulsion
    correlation_energy = total_energy - rhf.total_energy

    verbose && begin
        @printf("Custom DMRG sites         = %d\n", physical_mpo.nspin)
        @printf("Custom DMRG terms         = %d\n", length(physical_mpo.terms))
        @printf("Custom DMRG max bond      = %d\n", _max_mps_bond_dimension(tensors))
        @printf("Custom DMRG corr energy   = %20.12f Eh\n", correlation_energy)
        @printf("Custom DMRG total energy  = %20.12f Eh\n", total_energy)
    end

    return (
        E_dmrg = correlation_energy,
        total_energy = total_energy,
        electronic_energy = electronic_energy,
        tensors = tensors,
        mps = tensors,
        mpo = physical_mpo,
        optimization_mpo = optimization_mpo,
        nspin = physical_mpo.nspin,
        n_elec = physical_mpo.n_elec,
        max_bond_dimension = _max_mps_bond_dimension(tensors),
        maxdim = maxdim_values,
        nsweeps = nsweeps_value,
        cutoff = cutoff,
        noise = noise_values,
        integral_cutoff = integral_cutoff,
        particle_penalty = particle_penalty,
        particle_number = particle_numbers[end],
        particle_numbers = particle_numbers,
        sweep_energies = sweep_energies,
        discarded_weights = discarded_weights,
        converged = nsweeps_value > 1 ? abs(sweep_energies[end] - sweep_energies[end - 1]) < 1e-8 : false,
    )
end
