"""
Unrestricted MP2 (UMP2) from a UHF reference.

Project #18 evaluates the second-order Møller-Plesset correction using explicit
alpha/beta spin orbitals from `run_uhf`.
"""

using Printf

"""
    compute_ump2(aso_oovv, eps_occ, eps_vir) -> (emp2, td)

Compute the UMP2 correlation energy and doubles amplitudes from an
antisymmetrized spin-orbital OOVV block and explicit occupied/virtual
spin-orbital energies:

```text
t_ij^ab = <ij||ab> / (eps_i + eps_j - eps_a - eps_b)
E_UMP2  = 1/4 sum_ijab <ij||ab> t_ij^ab.
```
"""
function compute_ump2(aso_oovv, eps_occ, eps_vir)
    td = make_td_spinorbital(aso_oovv, eps_occ, eps_vir)
    return (emp2 = mp2_so(aso_oovv, td), td = td)
end

"""
    run_ump2(uhf; verbose=true) -> NamedTuple

Run unrestricted MP2 from the NamedTuple returned by `run_uhf`.  UHF alpha and
beta orbitals are ordered into an occupied-first spin-orbital basis before the
same spin-orbital MP2 formula is evaluated.
"""
function run_ump2(uhf; verbose=true)
    o = uhf.n_elec
    v = 2 * uhf.nbasis - o
    inputs = build_uccsd_spinorbital_inputs(uhf)

    oi = 1:o
    vi = (o + 1):(o + v)
    aso_oovv = inputs.aso[oi, oi, vi, vi]
    result = compute_ump2(aso_oovv, inputs.orbital_energies[oi], inputs.orbital_energies[vi])
    total = uhf.total_energy + result.emp2

    if verbose
        @printf("UMP2 correlation energy = %20.12f Eh\n", result.emp2)
        @printf("UMP2 total energy       = %20.12f Eh\n", total)
    end

    return (
        emp2 = result.emp2,
        total_energy = total,
        reference_energy = uhf.total_energy,
        td = result.td,
        oovv = aso_oovv,
        orbital_energies = inputs.orbital_energies,
        orbital_order = inputs.order,
        spins = inputs.spins,
    )
end
