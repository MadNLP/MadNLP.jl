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

# GPU solver placeholder types. These are abstract types defined here
# so they can be referenced as `linear_solver=LapackCUDASolver` in the
# options system, and support dispatch via `is_supported`, `input_type`,
# `default_options`, etc. Backend extensions (CUDA, ROCm) define concrete
# subtypes with the actual implementations.
abstract type LapackCUDASolver{T, MT, Alg} <: MadNLP.AbstractLapackSolver{T, Alg} end
abstract type CUDSSSolver{T, V} <: MadNLP.AbstractLinearSolver{T} end
abstract type LapackROCmSolver{T, MT, Alg} <: MadNLP.AbstractLapackSolver{T, Alg} end

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
