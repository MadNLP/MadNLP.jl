using Pkg

MADNLP_DIR = pwd()

# Dev the local (unregistered, monorepo) packages FIRST. They appear in the docs
# [deps], so a Pkg.update()/resolve() before this would try to find e.g. cuMadNLP
# in a registry and fail. Their [sources] pull in MadCore + the lib backends.
Pkg.develop(
    [
        PackageSpec(path = MADNLP_DIR),
        PackageSpec(path = joinpath(MADNLP_DIR, "MadCore")),
        PackageSpec(path = joinpath(MADNLP_DIR, "lib", "MadNLPCore")),
        PackageSpec(path = joinpath(MADNLP_DIR, "lib", "cuMadNLP")),
        PackageSpec(path = joinpath(MADNLP_DIR, "lib", "MadNLPTests")),
    ]
)
Pkg.instantiate()
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("cuMadNLP"))
Base.compilecache(Base.identify_package("MadNLP"))
