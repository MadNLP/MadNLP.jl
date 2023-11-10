
rm(joinpath(@__DIR__, "Project.toml"); force=true)
rm(joinpath(@__DIR__, "Manifest.toml"); force=true)

using Pkg

Pkg.activate(@__DIR__)


if ARGS[1] == "full"
    pkgs = ["MadNLPHSL","MadNLPPardiso","MadNLPMumps"]
            # ,"MadNLPKrylov"] # Krylov has been discontinued since the introduction of iterative refinement on the full space.
elseif ARGS[1] == "basic"
    pkgs = ["MadNLPMumps","MadNLPKrylov"]
elseif ARGS[1] == "cuda"
    pkgs = ["MadNLPGPU"]
else
    error("proper argument should be given - full or basic")
end

Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..")))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..","lib","MadNLPTests")))
Pkg.develop.([PackageSpec(path=joinpath(@__DIR__,"..","lib",pkg)) for pkg in pkgs])
Pkg.build()

if ARGS[1] == "full" || ARGS[1] == "basic"
    Pkg.test("MadNLP", coverage=true)
end
Pkg.test.(pkgs, coverage=true)
