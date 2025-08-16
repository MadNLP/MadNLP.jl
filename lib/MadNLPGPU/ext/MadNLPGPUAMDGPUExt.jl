module MadNLPGPUAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadNLP
import MadNLPGPU
import MadNLPGPU: LapackGPUSolver

import KernelAbstractions: synchronize

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

include("utils.jl")
include("KKT/rocm_dense.jl")
include("LinearSolvers/rocsolver.jl")

end
