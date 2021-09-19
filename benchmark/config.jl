using Pkg, Distributed, DelimitedFiles

const NP = ARGS[1]
const SOLVER = ARGS[2]
const VERBOSE = ARGS[3] == "true"
const QUICK = ARGS[4] == "true"
const GCOFF = ARGS[5] == "true"
const DECODE = ARGS[6] == "true"

addprocs(parse(Int,NP),exeflags="--project=.")
Pkg.instantiate()

if SOLVER == "master"
    Pkg.add(PackageSpec(name="MadNLP",rev="master"))
    Pkg.add(PackageSpec(name="MadNLPHSL",rev="master"))
    Pkg.build("MadNLPHSL")
elseif SOLVER == "current"
    Pkg.develop(path=joinpath(@__DIR__,".."))
    Pkg.develop(path=joinpath(@__DIR__,"..","lib","MadNLPHSL"))
    Pkg.build("MadNLPHSL")
elseif SOLVER == "ipopt"
elseif SOLVER == "knitro"
else
    error("Proper ARGS should be given")
end

# Set verbose option
if SOLVER == "ipopt"
    const PRINT_LEVEL = VERBOSE ? 5 : 0
elseif SOLVER == "knitro"
    const PRINT_LEVEL = VERBOSE ? 3 : 0
else
    using MadNLP
    const PRINT_LEVEL = VERBOSE ? MadNLP.INFO : MadNLP.ERROR
end

# Set quick option

