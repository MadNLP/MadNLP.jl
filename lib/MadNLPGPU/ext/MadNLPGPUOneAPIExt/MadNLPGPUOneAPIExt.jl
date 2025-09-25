module MadNLPGPUOneAPIExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadNLP
import MadNLPGPU

import KernelAbstractions: synchronize
import GPUArraysCore: @allowscalar

using oneAPI
using oneAPI.oneMKL, oneAPI.Support

function __init__()
    setglobal!(MadNLPGPU, :LapackOneMKLSolver, LapackOneMKLSolver)
    return
end

include("oneapi_dense.jl")
include("oneapi_sparse.jl")
include("onemkl.jl")
include("oneapi.jl")

end
