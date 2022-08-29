using Pkg

MADNLP_DIR = pwd()

Pkg.develop(path=MADNLP_DIR)
Pkg.develop(PackageSpec(path=joinpath(dirname(@__FILE__), "../lib/MadNLPTests/")))
Pkg.instantiate()

