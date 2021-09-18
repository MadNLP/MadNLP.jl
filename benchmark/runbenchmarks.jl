try
    using ArgParse
catch e
    println("Package ArgParse is required, but not installed. Install now? [y/n]")
    if readline() == "y"
        import Pkg
        Pkg.add("ArgParse")
        using ArgParse
    else
        exit()
    end
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--nprocs", "-p"
            help = "number of worker processors"
            arg_type = Int
            default = 1
        "--verbose", "-v"
            help = "print the solver outputs"
            action = :store_true
        "--gcoff", "-g"
            help = "turn of the julia garbage collector"
            action = :store_true
        "--quick", "-q"
            help = "run tests with reduced number of instances"
            action = :store_true
        "--decode", "-d"
            help = "decode the cutest instances"
            action = :store_true
        "testsets"
            help = "testsets for benchmark (separated by comma). possible values: cutest, power"
            required = true
        "solvers"
            help = "solvers to benchmark (separated by comma). possible values: current, master, ipopt"
            required = true
    end

    return parse_args(s)
end

function main()
    pargs = parse_commandline()

    CLASSES = split(pargs["testsets"],",")
    SOLVERS = split(pargs["solvers"],",")

    # sanity check
    issubset(CLASSES,["cutest","power"]) || error("argument testsets is incorrect")
    issubset(SOLVERS,["current","master","ipopt","knitro"]) || error("argument solvers is incorrect")

    PROJECT_PATH = dirname(@__FILE__)
    
    cp(
        joinpath(PROJECT_PATH, ".Project.toml"),
        joinpath(PROJECT_PATH, "Project.toml"),
        force=true
    )

    for class in CLASSES
        for solver in SOLVERS
            launch_script = joinpath(PROJECT_PATH, "benchmark-$class.jl")
            run(`julia --project=$PROJECT_PATH $launch_script $(pargs["nprocs"]) $solver $(pargs["verbose"]) $(pargs["quick"]) $(pargs["gcoff"]) $(pargs["decode"])`)
        end
    end

    run(Cmd([["julia","--project=$PROJECT_PATH","plot.jl"];String.(CLASSES);String.(SOLVERS)]))

end

main()
