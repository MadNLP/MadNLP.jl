using Pkg

MADNLP_DIR = pwd()

Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests"))
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU"))
Pkg.develop(path=MADNLP_DIR)
Pkg.instantiate()

