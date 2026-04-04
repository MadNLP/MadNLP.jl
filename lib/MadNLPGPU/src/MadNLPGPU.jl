module MadNLPGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!, norm, dot

# GPU abstractions
import Adapt: adapt
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
# Kernels
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend

import MadNLP: NLPModels
import MadNLP
import MadNLP:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!

# AMD and Metis
import AMD, Metis

include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/kernels_qn.jl")
include("utils.jl")
include("KKT/gpu_dense.jl")
include("KKT/gpu_sparse.jl")
include("KKT/gpu_qn.jl")

# GPU solver types are provided by backend extensions (CUDA, ROCm).
# We store them in typed Refs to avoid untyped globals (AOT-incompatible).
const _LapackCUDASolver_type = Ref{Type}(Nothing)
const _CUDSSSolver_type = Ref{Type}(Nothing)
const _LapackROCmSolver_type = Ref{Type}(Nothing)

"""
    LapackCUDASolver(args...; kwargs...)
Construct a CUDA-based Lapack solver (requires CUDA.jl to be loaded).
"""
function LapackCUDASolver(args...; kwargs...)
    T = _LapackCUDASolver_type[]
    T === Nothing && error("LapackCUDASolver requires CUDA.jl. Please run `using CUDA` first.")
    return T(args...; kwargs...)
end

"""
    CUDSSSolver(args...; kwargs...)
Construct a CUDSS solver (requires CUDA.jl to be loaded).
"""
function CUDSSSolver(args...; kwargs...)
    T = _CUDSSSolver_type[]
    T === Nothing && error("CUDSSSolver requires CUDA.jl. Please run `using CUDA` first.")
    return T(args...; kwargs...)
end

"""
    LapackROCmSolver(args...; kwargs...)
Construct an ROCm-based Lapack solver (requires AMDGPU.jl to be loaded).
"""
function LapackROCmSolver(args...; kwargs...)
    T = _LapackROCmSolver_type[]
    T === Nothing && error("LapackROCmSolver requires AMDGPU.jl. Please run `using AMDGPU` first.")
    return T(args...; kwargs...)
end

export LapackCUDASolver, CUDSSSolver, LapackROCmSolver

# Re-export all exported names from MadNLP.
# Using a loop with @eval at module-definition time is AOT-safe since
# it runs during precompilation, not at runtime.
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
