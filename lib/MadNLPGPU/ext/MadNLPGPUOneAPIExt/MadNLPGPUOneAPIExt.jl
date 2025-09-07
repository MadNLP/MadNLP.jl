module MadNLPGPUAMDGPUExt

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric

import MadNLP
import MadNLPGPU

import KernelAbstractions: synchronize

using oneAPI
using oneAPI.oneMKL, onAPI.Support

function __init__()
    setglobal!(MadNLPGPU, :LapackOneMKLSolver, LapackOneMKLSolver)
    return
end

include("oneapi_dense.jl")
include("oneapi_sparse.jl")
include("onemkl.jl")
include("oneapi.jl")

end
