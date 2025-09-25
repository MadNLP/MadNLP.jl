module MadNLPGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric
# CUDA
import CUDA: CUDA, CUSPARSE, CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray,
    has_cuda, @allowscalar, runtime_version, CUDABackend
import .CUSOLVER: cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t,
    dense_handle, CuSolverParameters, CUSOLVER_EIG_MODE_VECTOR
import .CUBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T

# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const

import MadNLP: NLPModels
import MadNLP
import MadNLP:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options, symul!

include("utils.jl")
include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/cuda_dense.jl")
include("KKT/cuda_sparse.jl")
include("LinearSolvers/lapackgpu.jl")
include("LinearSolvers/cusolver.jl")
include("LinearSolvers/cudss.jl")
include("cuda.jl")

global LapackROCSolver
export LapackGPUSolver, CUDSSSolver, LapackROCSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
