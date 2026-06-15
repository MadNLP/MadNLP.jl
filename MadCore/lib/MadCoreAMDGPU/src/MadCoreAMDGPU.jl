module MadCoreAMDGPU

# ROCm/AMDGPU backend for MadCore: LapackROCmSolver and the GPU sparse glue,
# migrated from MadNLPGPU/ext/MadNLPGPUAMDGPUExt and promoted from a weakdep
# extension to a standalone package (AMDGPU is a hard dependency). Builds on
# MadCore + MadCoreKernelAbstractions. The IPM-specific GPU-default MadNLPOptions
# constructor was split out (a future RocMadNLP, mirroring cuMadNLP).

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadCore
import MadCoreKernelAbstractions

import KernelAbstractions: synchronize, get_backend

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

include("rocm_sparse.jl")
include("rocsolver.jl")
include("rocm.jl")

export LapackROCmSolver

end # module
