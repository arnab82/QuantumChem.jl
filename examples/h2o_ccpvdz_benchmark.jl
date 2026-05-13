#!/usr/bin/env julia

using Printf
using QuantumChem

const H2O_CRAWFORD_GEOM =
    "O 0.000000000000 -0.143225816552 0.000000000000;" *
    "H 1.638036840407 1.136548822547 0.000000000000;" *
    "H -1.638036840407 1.136548822547 0.000000000000"

const METHOD_ORDER = ["rhf", "mp2", "ccsd", "ccsd_t", "ccsdt", "cisd", "hci", "dmrg"]

function usage()
    println("""
    H2O/cc-pVDZ method comparison.

    Examples:
      julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl
      julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl --all
      julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl --all --force-heavy
      julia --threads=auto --project=. examples/h2o_ccpvdz_benchmark.jl --methods=rhf,mp2,ccsd,ccsd_t,dmrg

    Options:
      --basis=cc-pvdz              Basis set passed to PySCF.
      --methods=list               Comma-separated methods: rhf, mp2, ccsd, ccsd_t,
                                   ccsdt, cisd, hci, dmrg, or all.
      --all                        Request every method.
      --force-heavy                Actually run full-basis CCSDT and large dense CISD.
      --cisd-max-gb=2.0            Dense CISD Hamiltonian memory guard.
      --ccsd-maxiter=100           CCSD iteration cap.
      --ccsdt-maxiter=25           CCSDT iteration cap when --force-heavy is set.
      --hci-epsilon1=5e-3          HCI selection threshold.
      --hci-epsilon2=0.0           HCI PT2 threshold.
      --hci-maxiter=8              HCI selection iteration cap.
      --dmrg-maxdim=4,8            Custom DMRG sweep bond dimensions.
      --dmrg-nsweeps=2             Custom DMRG sweep count.
      --dmrg-integral-cutoff=1e-4  Drop smaller spin-orbital Hamiltonian terms.
      --dmrg-particle-penalty=200  Penalty for leaving the target electron sector.
      --verbose-methods            Print each method's internal iterations.
    """)
end

function normalize_method(method::AbstractString)
    raw = lowercase(strip(method))
    raw == "all" && return "all"
    raw in ("ccsd(t)", "ccsd_t", "ccsd-t", "perturbative_triples") && return "ccsd_t"
    raw in ("ccsdt", "full_ccsdt", "full-ccsdt") && return "ccsdt"
    raw in ("rhf", "hf") && return "rhf"
    raw in ("mp2", "ccsd", "cisd", "hci", "dmrg") && return raw
    throw(ArgumentError("Unknown method '$method'"))
end

function parse_method_list(value::AbstractString)
    methods = [normalize_method(part) for part in split(value, ",") if !isempty(strip(part))]
    return "all" in methods ? copy(METHOD_ORDER) : methods
end

parse_int_list(value::AbstractString) = [parse(Int, strip(part)) for part in split(value, ",") if !isempty(strip(part))]

function parse_args(args)
    opts = Dict{String,Any}(
        "basis" => "cc-pvdz",
        "methods" => String[],
        "force_heavy" => false,
        "cisd_max_gb" => 2.0,
        "ccsd_maxiter" => 100,
        "ccsdt_maxiter" => 25,
        "hci_epsilon1" => 5e-3,
        "hci_epsilon2" => 0.0,
        "hci_maxiter" => 8,
        "dmrg_maxdim" => [4, 8],
        "dmrg_nsweeps" => 2,
        "dmrg_integral_cutoff" => 1e-4,
        "dmrg_particle_penalty" => 200.0,
        "verbose_methods" => false,
    )

    for arg in args
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--all"
            opts["methods"] = copy(METHOD_ORDER)
        elseif arg == "--force-heavy"
            opts["force_heavy"] = true
        elseif arg == "--verbose-methods"
            opts["verbose_methods"] = true
        elseif startswith(arg, "--basis=")
            opts["basis"] = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--methods=")
            opts["methods"] = parse_method_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cisd-max-gb=")
            opts["cisd_max_gb"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ccsd-maxiter=")
            opts["ccsd_maxiter"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--ccsdt-maxiter=")
            opts["ccsdt_maxiter"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--hci-epsilon1=")
            opts["hci_epsilon1"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--hci-epsilon2=")
            opts["hci_epsilon2"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--hci-maxiter=")
            opts["hci_maxiter"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dmrg-maxdim=")
            opts["dmrg_maxdim"] = parse_int_list(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dmrg-nsweeps=")
            opts["dmrg_nsweeps"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dmrg-integral-cutoff=")
            opts["dmrg_integral_cutoff"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--dmrg-particle-penalty=")
            opts["dmrg_particle_penalty"] = parse(Float64, split(arg, "=", limit=2)[2])
        else
            throw(ArgumentError("Unknown option '$arg'. Run with --help for usage."))
        end
    end

    if isempty(opts["methods"])
        opts["methods"] = ["rhf", "mp2", "ccsd", "ccsd_t", "hci", "dmrg"]
    end
    return opts
end

function expanded_methods(methods)
    wanted = Set(methods)
    if any(method != "rhf" for method in wanted)
        push!(wanted, "rhf")
    end
    if any(method in ("mp2", "ccsd", "ccsd_t", "ccsdt", "cisd", "hci", "dmrg") for method in wanted)
        push!(wanted, "mp2")
    end
    if "ccsd_t" in wanted
        push!(wanted, "ccsd")
    end
    return [method for method in METHOD_ORDER if method in wanted]
end

estimate_cisd_dimension(nspin, nelec) =
    1 + nelec * (nspin - nelec) + binomial(nelec, 2) * binomial(nspin - nelec, 2)

estimate_dense_gb(dimension) = 8.0 * dimension * dimension / 1024.0^3

format_energy(value) = isfinite(value) ? @sprintf("% .12f", value) : ""
format_seconds(value) = @sprintf("%.2f", value)

function push_record!(records; method, status, total=NaN, correlation=NaN,
                      dimension="", seconds=0.0, notes="")
    push!(records, (
        method=method,
        status=status,
        total=Float64(total),
        correlation=Float64(correlation),
        dimension=string(dimension),
        seconds=Float64(seconds),
        notes=notes,
    ))
end

function run_and_record!(records, method, runner, summarizer)
    start = time()
    try
        result = runner()
        summary = summarizer(result)
        push_record!(records;
                     method=method,
                     status="ok",
                     total=get(summary, :total, NaN),
                     correlation=get(summary, :correlation, NaN),
                     dimension=get(summary, :dimension, ""),
                     seconds=time() - start,
                     notes=get(summary, :notes, ""))
        return result
    catch err
        push_record!(records;
                     method=method,
                     status="error",
                     seconds=time() - start,
                     notes=sprint(showerror, err))
        return nothing
    end
end

function print_records(records)
    println()
    println("H2O cc-pVDZ method comparison")
    @printf("%-10s %-8s %18s %18s %12s %10s  %s\n",
            "method", "status", "total_Eh", "corr_Eh", "dimension", "seconds", "notes")
    println(repeat("-", 104))
    for record in records
        @printf("%-10s %-8s %18s %18s %12s %10s  %s\n",
                record.method,
                record.status,
                format_energy(record.total),
                format_energy(record.correlation),
                record.dimension,
                format_seconds(record.seconds),
                record.notes)
    end
end

function main(args)
    opts = parse_args(args)
    methods = expanded_methods(opts["methods"])
    verbose = Bool(opts["verbose_methods"])
    records = NamedTuple[]

    println("System: H2O, basis=$(opts["basis"]), unit=Bohr")
    println("Methods: $(join(methods, ", "))")

    rhf = nothing
    mp2 = nothing
    ccsd = nothing

    if "rhf" in methods
        rhf = run_and_record!(records, "RHF",
            () -> run_rhf(atoms=H2O_CRAWFORD_GEOM, basis=opts["basis"],
                          unit="Bohr", charge=0, spin=0, n_elec=10,
                          diis=true, verbose=verbose),
            result -> (total=result.total_energy,
                       correlation=0.0,
                       dimension=result.nbasis,
                       notes="nbasis=$(result.nbasis), nelec=$(result.n_elec)"))
    end

    if "mp2" in methods && rhf !== nothing
        mp2 = run_and_record!(records, "MP2",
            () -> run_mp2(rhf; verbose=verbose),
            result -> (total=rhf.total_energy + result.emp2,
                       correlation=result.emp2,
                       dimension=size(result.new_eri, 1),
                       notes="MO ERI size=$(size(result.new_eri))"))
    end

    if "ccsd" in methods && rhf !== nothing && mp2 !== nothing
        ccsd = run_and_record!(records, "CCSD",
            () -> run_ccsd(rhf, mp2; maxiter=opts["ccsd_maxiter"],
                           diis=true, verbose=verbose),
            result -> (total=result.total_energy,
                       correlation=result.E_ccsd,
                       dimension="$(rhf.n_elec)x$(2 * rhf.nbasis - rhf.n_elec)",
                       notes="converged=$(result.converged), iterations=$(result.iterations)"))
    end

    if "ccsd_t" in methods && rhf !== nothing && mp2 !== nothing && ccsd !== nothing
        run_and_record!(records, "CCSD(T)",
            () -> run_ccsd_t(rhf, mp2, ccsd; verbose=verbose),
            result -> (total=result.total_energy,
                       correlation=result.total_energy - rhf.total_energy,
                       dimension="$(rhf.n_elec)^3 x $(2 * rhf.nbasis - rhf.n_elec)^3",
                       notes="E(T)=$(result.E_triples)"))
    end

    if "ccsdt" in methods && rhf !== nothing && mp2 !== nothing
        o = rhf.n_elec
        v = 2 * rhf.nbasis - rhf.n_elec
        t3_elements = o^3 * v^3
        if !Bool(opts["force_heavy"])
            push_record!(records;
                         method="CCSDT",
                         status="skipped",
                         dimension=t3_elements,
                         notes="full-basis iterative CCSDT is expensive; rerun with --force-heavy")
        else
            run_and_record!(records, "CCSDT",
                () -> run_ccsdt(rhf, mp2; maxiter=opts["ccsdt_maxiter"],
                                diis=true, verbose=verbose),
                result -> (total=result.total_energy,
                           correlation=result.E_ccsdt,
                           dimension=t3_elements,
                           notes="converged=$(result.converged), iterations=$(result.iterations)"))
        end
    end

    if "cisd" in methods && rhf !== nothing && mp2 !== nothing
        nspin = 2 * rhf.nbasis
        dim = estimate_cisd_dimension(nspin, rhf.n_elec)
        dense_gb = estimate_dense_gb(dim)
        if dense_gb > opts["cisd_max_gb"] && !Bool(opts["force_heavy"])
            push_record!(records;
                         method="CISD",
                         status="skipped",
                         dimension=dim,
                         notes=@sprintf("dense Hamiltonian estimate %.2f GiB; raise --cisd-max-gb or use --force-heavy", dense_gb))
        else
            run_and_record!(records, "CISD",
                () -> run_cisd(rhf, mp2; verbose=verbose),
                result -> (total=result.total_energy,
                           correlation=result.E_cisd,
                           dimension=result.dimension,
                           notes="reference_weight=$(result.reference_weight)"))
        end
    end

    if "hci" in methods && rhf !== nothing && mp2 !== nothing
        run_and_record!(records, "HCI",
            () -> run_hci(rhf, mp2; epsilon1=opts["hci_epsilon1"],
                          epsilon2=opts["hci_epsilon2"],
                          maxiter=opts["hci_maxiter"],
                          verbose=verbose),
            result -> (total=result.total_energy,
                       correlation=result.E_hci_pt2,
                       dimension=result.dimension,
                       notes="epsilon1=$(result.epsilon1), converged=$(result.converged)"))
    end

    if "dmrg" in methods && rhf !== nothing && mp2 !== nothing
        maxdim = opts["dmrg_maxdim"]
        nsweeps = opts["dmrg_nsweeps"]
        integral_cutoff = opts["dmrg_integral_cutoff"]
        run_and_record!(records, "DMRG",
            () -> run_dmrg(rhf, mp2; maxdim=maxdim,
                           nsweeps=nsweeps,
                           integral_cutoff=integral_cutoff,
                           particle_penalty=opts["dmrg_particle_penalty"],
                           verbose=verbose),
            result -> (total=result.total_energy,
                       correlation=result.E_dmrg,
                       dimension=result.max_bond_dimension,
                       notes="terms=$(length(result.mpo.terms)), <N>=$(result.particle_number)"))
    end

    print_records(records)
end

main(ARGS)
