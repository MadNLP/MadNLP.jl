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
    setglobal!(MadNLPGPU, :LapackROCmSolver, LapackROCmSolver)
    return
end

include("rocm_dense.jl")
include("rocm_sparse.jl")
include("rocm_qn.jl")
include("rocsolver.jl")
include("rocm.jl")

end
