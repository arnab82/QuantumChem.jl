"""
RHF geometry optimization.

Project #25 uses the analytic RHF gradients from Project #24 to optimize
Cartesian nuclear coordinates.  The implementation keeps the algorithm explicit:
RHF is rebuilt at each trial geometry, the step is capped in Cartesian space,
and a backtracking line search enforces downhill energy movement.
"""

using LinearAlgebra, Printf, PyCall, Statistics

"""
    rhf_geometry_coordinates(rhf) -> Matrix{Float64}

Return the current PySCF nuclear coordinates from an RHF result in Bohr, with
shape `natom x 3`.
"""
function rhf_geometry_coordinates(rhf)
    mol = _require_rhf_molecule(rhf)
    return _pyarray_call(mol.atom_coords, Float64, Val(2))
end

"""
    rhf_atom_symbols(rhf) -> Vector{String}

Return the element symbols associated with the RHF molecule in atom order.
"""
function rhf_atom_symbols(rhf)
    mol = _require_rhf_molecule(rhf)
    natoms = pyscf_getattr(mol, "natm", Int)
    symbols = Vector{String}(undef, natoms)
    for atom in 1:natoms
        symbols[atom] = pycall(mol.atom_pure_symbol, String, atom - 1)
    end
    return symbols
end

"""
    format_atoms_bohr(symbols, coordinates)

Format a PySCF atom string from element symbols and Bohr coordinates.
"""
function format_atoms_bohr(symbols, coordinates)
    length(symbols) == size(coordinates, 1) ||
        throw(DimensionMismatch("Number of atom symbols must match coordinate rows"))
    size(coordinates, 2) == 3 ||
        throw(DimensionMismatch("Coordinates must have shape natom x 3"))

    parts = String[]
    for atom in eachindex(symbols)
        push!(parts, @sprintf("%s %.14f %.14f %.14f",
                              symbols[atom],
                              coordinates[atom, 1],
                              coordinates[atom, 2],
                              coordinates[atom, 3]))
    end
    return join(parts, ";")
end

function _require_rhf_field(rhf, name::Symbol)
    name in propertynames(rhf) ||
        throw(ArgumentError("RHF geometry optimization needs field '$name' from run_rhf"))
    return getproperty(rhf, name)
end

function _max_row_norm(matrix)
    isempty(matrix) && return 0.0
    return maximum(norm(@view matrix[row, :]) for row in axes(matrix, 1))
end

function _bounded_displacement(direction, step_scale, max_step)
    displacement = step_scale .* direction
    max_displacement = _max_row_norm(displacement)
    if max_displacement > max_step && max_displacement > eps(Float64)
        displacement .*= max_step / max_displacement
    end
    return displacement
end

function _remove_translation!(direction)
    shift = vec(mean(direction; dims=1))
    for atom in axes(direction, 1)
        direction[atom, :] .-= shift
    end
    return direction
end

function _descent_direction(method::Symbol, gradient, inverse_hessian)
    if method === :steepest_descent
        return -copy(gradient)
    elseif method === :bfgs
        direction = reshape(-(inverse_hessian * vec(gradient)), size(gradient))
        dot(vec(direction), vec(gradient)) < 0.0 && return direction
        return -copy(gradient)
    else
        throw(ArgumentError("Unknown geometry optimization method $method; use :bfgs or :steepest_descent"))
    end
end

function _rebuild_rhf_at_geometry(rhf, symbols, coordinates;
                                  scf_diis=true, rhf_verbose=false)
    atoms = format_atoms_bohr(symbols, coordinates)
    return run_rhf(
        atoms=atoms,
        basis=_require_rhf_field(rhf, :basis),
        charge=Int(_require_rhf_field(rhf, :charge)),
        spin=Int(_require_rhf_field(rhf, :spin)),
        unit="Bohr",
        n_elec=Int(rhf.n_elec),
        diis=scf_diis,
        verbose=rhf_verbose,
    )
end

function _try_geometry_step(rhf, symbols, coordinates, gradient, direction;
                            initial_step, max_step, min_step, backtrack,
                            armijo, line_search, scf_diis, rhf_verbose)
    current_energy = rhf.total_energy
    step_scale = Float64(initial_step)
    last_error = nothing

    while step_scale >= min_step
        displacement = _bounded_displacement(direction, step_scale, max_step)
        predicted_change = dot(vec(gradient), vec(displacement))
        if predicted_change >= 0.0
            return (accepted=false, reason="search direction is not downhill",
                    last_error=last_error)
        end

        trial_coordinates = coordinates .+ displacement
        try
            trial_rhf = _rebuild_rhf_at_geometry(rhf, symbols, trial_coordinates;
                                                scf_diis, rhf_verbose)
            sufficient_decrease =
                trial_rhf.total_energy <= current_energy + armijo * predicted_change
            if !line_search || sufficient_decrease
                trial_gradient = run_rhf_gradient(trial_rhf; verbose=false)
                return (
                    accepted = true,
                    rhf = trial_rhf,
                    gradient_result = trial_gradient,
                    coordinates = trial_coordinates,
                    displacement = displacement,
                    step_scale = step_scale,
                    predicted_change = predicted_change,
                    energy_change = trial_rhf.total_energy - current_energy,
                    reason = "accepted",
                    last_error = nothing,
                )
            end
        catch err
            last_error = err
        end

        step_scale *= backtrack
    end

    reason = "line search failed below min_step=$min_step"
    last_error === nothing || (reason *= "; last error: $(sprint(showerror, last_error))")
    return (accepted=false, reason=reason, last_error=last_error)
end

function _update_inverse_hessian(inverse_hessian, displacement, old_gradient,
                                 new_gradient; curvature_tol=1e-10)
    s = vec(displacement)
    y = vec(new_gradient .- old_gradient)
    ys = dot(y, s)
    ys > curvature_tol || return inverse_hessian

    rho = 1.0 / ys
    ndof = length(s)
    identity = Matrix{Float64}(I, ndof, ndof)
    transform = identity .- rho .* (s * y')
    return transform * inverse_hessian * transform' .+ rho .* (s * s')
end

"""
    run_rhf_geometry_optimization([rhf]; kwargs...)

Optimize Cartesian coordinates with RHF analytic gradients.  If `rhf` is not
provided, the remaining keywords are passed to `run_rhf` for the initial
geometry.

Important controls:
- `method=:bfgs` or `:steepest_descent`
- `maxiter=20`
- `gradient_tol=3e-4`
- `initial_step=0.5`
- `max_step=0.2`
- `line_search=true`

For BFGS steps the search direction is

```text
p_k = -H_k^{-1} g_k,
```

and the inverse Hessian update uses

```text
H_{k+1}^{-1} = (I - rho s y') H_k^{-1} (I - rho y s') + rho s s',
rho = 1/(y's).
```
"""
function run_rhf_geometry_optimization(rhf=nothing;
                                       method=:bfgs,
                                       maxiter=20,
                                       gradient_tol=3e-4,
                                       rms_gradient_tol=2e-4,
                                       energy_tol=1e-8,
                                       initial_step=0.5,
                                       max_step=0.2,
                                       min_step=1e-5,
                                       backtrack=0.5,
                                       armijo=1e-4,
                                       line_search=true,
                                       remove_translation=true,
                                       scf_diis=true,
                                       verbose=true,
                                       rhf_verbose=false,
                                       kwargs...)
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative"))
    initial_step > 0 || throw(ArgumentError("initial_step must be positive"))
    max_step > 0 || throw(ArgumentError("max_step must be positive"))
    min_step > 0 || throw(ArgumentError("min_step must be positive"))
    0 < backtrack < 1 || throw(ArgumentError("backtrack must be between 0 and 1"))
    armijo >= 0 || throw(ArgumentError("armijo must be nonnegative"))

    if rhf === nothing
        rhf = run_rhf(; verbose=rhf_verbose, kwargs...)
    elseif !isempty(kwargs)
        throw(ArgumentError("When an RHF result is supplied, run_rhf_geometry_optimization does not accept run_rhf keywords"))
    end

    symbols = rhf_atom_symbols(rhf)
    coordinates = rhf_geometry_coordinates(rhf)
    gradient_result = run_rhf_gradient(rhf; verbose=false)
    gradient = gradient_result.gradient
    ndof = length(gradient)
    inverse_hessian = Matrix{Float64}(I, ndof, ndof)

    energies = [rhf.total_energy]
    max_forces = [gradient_result.max_force]
    rms_forces = [gradient_result.rms_force]
    geometries = [copy(coordinates)]
    gradients = [copy(gradient)]
    displacements = Matrix{Float64}[]
    step_scales = Float64[]
    energy_changes = Float64[]
    accepted_steps = 0
    converged = gradient_result.max_force <= gradient_tol &&
                gradient_result.rms_force <= rms_gradient_tol
    reason = converged ? "initial geometry converged" : "maxiter reached"

    verbose && @printf("RHF geometry optimization: iter=%3d  E=%20.12f  max|g|=%.6e  rms|g|=%.6e\n",
                       0, rhf.total_energy, gradient_result.max_force,
                       gradient_result.rms_force)

    for iteration in 1:maxiter
        converged && break

        direction = _descent_direction(Symbol(method), gradient, inverse_hessian)
        remove_translation && _remove_translation!(direction)

        step = _try_geometry_step(rhf, symbols, coordinates, gradient, direction;
                                  initial_step, max_step, min_step, backtrack,
                                  armijo, line_search, scf_diis, rhf_verbose)
        if !step.accepted
            reason = step.reason
            break
        end

        old_gradient = gradient
        rhf = step.rhf
        gradient_result = step.gradient_result
        gradient = gradient_result.gradient
        coordinates = step.coordinates
        accepted_steps += 1

        push!(energies, rhf.total_energy)
        push!(max_forces, gradient_result.max_force)
        push!(rms_forces, gradient_result.rms_force)
        push!(geometries, copy(coordinates))
        push!(gradients, copy(gradient))
        push!(displacements, copy(step.displacement))
        push!(step_scales, step.step_scale)
        push!(energy_changes, step.energy_change)

        if Symbol(method) === :bfgs
            inverse_hessian = _update_inverse_hessian(inverse_hessian,
                                                      step.displacement,
                                                      old_gradient,
                                                      gradient)
        end

        converged = gradient_result.max_force <= gradient_tol &&
                    gradient_result.rms_force <= rms_gradient_tol
        energy_converged = abs(step.energy_change) <= energy_tol &&
                           gradient_result.max_force <= 10gradient_tol
        if converged
            reason = "gradient converged"
        elseif energy_converged
            converged = true
            reason = "energy and loose gradient converged"
        end

        verbose && @printf("RHF geometry optimization: iter=%3d  E=%20.12f  dE=% .3e  max|g|=%.6e  step=%.3e\n",
                           iteration, rhf.total_energy, step.energy_change,
                           gradient_result.max_force,
                           _max_row_norm(step.displacement))
    end

    atoms = format_atoms_bohr(symbols, coordinates)
    return (
        converged = converged,
        reason = reason,
        iterations = accepted_steps,
        method = Symbol(method),
        rhf = rhf,
        gradient_result = gradient_result,
        gradient = gradient,
        coordinates = coordinates,
        atoms = atoms,
        symbols = symbols,
        unit = "Bohr",
        energies = energies,
        max_forces = max_forces,
        rms_forces = rms_forces,
        geometries = geometries,
        gradients = gradients,
        displacements = displacements,
        step_scales = step_scales,
        energy_changes = energy_changes,
        max_force = max_forces[end],
        rms_force = rms_forces[end],
        gradient_tol = gradient_tol,
        rms_gradient_tol = rms_gradient_tol,
        energy_tol = energy_tol,
        max_step = max_step,
    )
end
