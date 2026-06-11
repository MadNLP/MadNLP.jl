module MadCoreGPUArrays

# Backend-agnostic GPU-array utilities for MadCore. Currently the data-transfer
# kernels (index_copy!, fixed!, transfer_coef!) used by MadNLPHybridKKT, written
# against GPUArraysCore + KernelAbstractions so they run on any GPU backend
# without a hard CUDA dependency.

import MadCore
import SparseArrays: SparseMatrixCSC, nonzeros
import GPUArraysCore: AbstractGPUVector
import KernelAbstractions
import KernelAbstractions: @kernel, @index, get_backend
import Atomix

include("kernels.jl")

export index_copy!, fixed!, transfer_coef!

end # module
