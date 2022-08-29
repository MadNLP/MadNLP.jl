using Pkg

MADNLP_DIR = pwd()

Pkg.develop(PackageSpec(path=MADNLP_DIR))
Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "../lib/MadNLPTests/")))
Pkg.instantiate()

