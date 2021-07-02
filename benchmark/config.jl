using Pkg, Distributed, JLD2

const NP = ARGS[1]
const SOLVER = ARGS[2]

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
