using Pkg

MADNLP_DIR = pwd()

Pkg.update()
Pkg.resolve()
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPTests"))
Pkg.develop(path=joinpath(MADNLP_DIR, "lib", "MadNLPGPU"))
Pkg.develop(path=MADNLP_DIR)
Base.compilecache(Base.identify_package("MadNLPTests"))
Base.compilecache(Base.identify_package("MadNLPGPU"))
Base.compilecache(Base.identify_package("MadNLP"))
Pkg.instantiate()

# Pin the CUDA runtime to a 12.x toolkit so the self-hosted V100 runner
# keeps working: CUDA 13 dropped sm_70 support, and without this pin
# CUDA_Runtime_jll may auto-select CUDA 13. See JuliaGPU/CUDA.jl#3134 —
# once that fix is in a tagged CUDA.jl release, this pin can be removed.
using CUDA
CUDA.set_runtime_version!(v"12.9")
