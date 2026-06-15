module cuMadNLP

# CUDA backend layer for MadNLP's interior-point solver. The device-agnostic GPU
# IPM kernels/scaling live in the shared MadNLPGPU base (re-exported here); this
# package adds only the CUDA-specific pieces: the GPU-default MadNLPOptions
# constructor (CuVector dispatch, LapackCUDASolver / CUDSSSolver defaults) and the
# CUDSS sparse-solver defaults for the condensed/hybrid formulations.

using Reexport
@reexport using MadNLPGPU
import MadNLP  # options.jl extends MadNLP.MadNLPOptions / MadNLP.default_sparse_solver

import CUDACore: CuVector, CUDABackend
import MadCoreCUDA: LapackCUDASolver
import MadCoreCUDSS: CUDSSSolver

include("options.jl")

# Re-export the CUDA KernelAbstractions backend so `using cuMadNLP` gives users
# CUDABackend() for constructing GPU KKT systems (e.g. MadNLPHybridKKT on CUDA).
export CUDABackend

end # module
