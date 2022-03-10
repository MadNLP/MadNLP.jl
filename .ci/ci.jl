
rm(joinpath(@__DIR__, "Project.toml"); force=true)
rm(joinpath(@__DIR__, "Manifest.toml"); force=true)

using Pkg

# Delete MathOptInterface directory to avoid side effect
# with coverage files generated in a previous CI run.
for _dep in Pkg.depots()
    moi_directory = joinpath(_dep, "packages", "MathOptInterface")
    if isdir(moi_directory)
        rm(moi_directory; force=true, recursive=true)
    end
end

Pkg.activate(@__DIR__)


if ARGS[1] == "full"
    pkgs = ["MadNLPHSL","MadNLPPardiso","MadNLPMumps","MadNLPGPU","MadNLPKrylov"]
elseif ARGS[1] == "basic"
    pkgs = ["MadNLPMumps","MadNLPKrylov"]
else
    error("proper argument should be given - full or basic")
end

Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..")))
Pkg.develop(PackageSpec(path=joinpath(@__DIR__,"..","lib","MadNLPTests")))
Pkg.develop.([PackageSpec(path=joinpath(@__DIR__,"..","lib",pkg)) for pkg in pkgs])
Pkg.build()

Pkg.test.("MadNLP", coverage=true)
Pkg.test.(pkgs, coverage=true)
