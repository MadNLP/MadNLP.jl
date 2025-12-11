using Pkg

MADNLP_DIR = pwd()

Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests"))
Pkg.develop(path="https://github.com/MadNLP/HybridKKT.jl")
Pkg.develop(path=MADNLP_DIR)
Pkg.instantiate()

