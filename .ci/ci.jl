using Pkg

if ARGS[1] == "full"
    pkgs = ["MadNLPHSL","MadNLPPardiso","MadNLPMumps","MadNLPGPU","MadNLPGraphs","MadNLPKrylov"]
elseif ARGS[1] == "basic"
    pkgs = ["MadNLPMumps","MadNLPGraphs","MadNLPKrylov"]
else
    error("proper argument should be given - full or basic")
end

Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..")))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..","lib","MadNLPTests")))
Pkg.develop.([PackageSpec(path=joinpath(@__DIR__,"..","lib",pkg)) for pkg in pkgs])
Pkg.build()

Pkg.test.("MadNLP", coverage=true)
Pkg.test.(pkgs, coverage=true)
