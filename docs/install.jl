using Pkg

MADNLP_DIR = pwd()

Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests"))
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU"))
Pkg.develop(path=MADNLP_DIR)
# Compile cache for Documenter to avoid issue with MbedTLS
Base.compilecache(Base.identify_package("Documenter"))
Pkg.instantiate()

