module MadCoreKernelAbstractions

# Solver-agnostic, KernelAbstractions-based GPU support: the GPU KKT systems
# (dense/sparse/quasi-Newton) and their kernels, migrated from MadNLPGPU. Backend
# packages (MadCoreCUDA, MadCoreAMDGPU) build on this; IPM-specific GPU code lives
# in MadNLP/lib/cuMadNLP.

import LinearAlgebra
import LinearAlgebra: Symmetric, mul!, norm, dot
import SparseArrays: SparseMatrixCSC, nonzeros, nnz

import Adapt: adapt
import GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray, @allowscalar
import KernelAbstractions: @kernel, @index, synchronize, @Const, get_backend

import MadCore
import MadCore:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!,
    SymbolicException, FactorizationException, SolveException, InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!,
    SubVector

import AMD, Metis

const AbstractGPUVectorOrSubVector{T, VT <: AbstractGPUVector{T}} =
    Union{AbstractGPUVector{T}, SubVector{T, VT}}

include("KKT/kernels_dense.jl")
include("KKT/kernels_sparse.jl")
include("KKT/kernels_qn.jl")
include("utils.jl")
include("KKT/gpu_dense.jl")
include("KKT/gpu_sparse.jl")
include("KKT/gpu_qn.jl")

end # module
