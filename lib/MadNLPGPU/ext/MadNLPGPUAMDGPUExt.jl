module MadNLPGPUAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadNLP
import MadNLPGPU

import KernelAbstractions: synchronize

using AMDGPU
using AMDGPU.rocBLAS, AMDGPU.rocSOLVER, AMDGPU.rocSPARSE

function __init__()
    setglobal!(MadNLPGPU, :LapackROCSolver, LapackROCSolver)
    return
end

include("utils.jl")
include("KKT/rocm_dense.jl")
include("LinearSolvers/rocsolver.jl")

end
