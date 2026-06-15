module MadCoreCUDSS

# CUDSS backend for MadCore: the cuDSS sparse direct solver (CUDSSSolver) and the
# GPU Schur-complement KKT system, split out of MadCoreCUDA so the cuSOLVER-based
# LapackCUDASolver / dense GPU path doesn't pull in the heavy cuDSS library. Builds
# on MadCore + MadCoreCUDA (LapackCUDASolver) + the CUDA stack + CUDSS.

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!

import MadCore
import MadCore:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException, FactorizationException, SolveException, InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options

import MadCoreKernelAbstractions
import MadCoreKernelAbstractions: ORDERING, DEFAULT_ORDERING, METIS_ORDERING, AMD_ORDERING,
    USER_ORDERING, SYMAMD_ORDERING, COLAMD_ORDERING, gpu_transfer!

import KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

import AMD, Metis

import MadCoreCUDA: LapackCUDASolver

using CUDACore
using cuSPARSE, cuBLAS, cuSOLVER
import CUDSS

include("cudss.jl")
include("kernels_schur.jl")
include("cuda_schur.jl")

# GPU-default options for CUDSSSolver + SparseCondensedKKTSystem (moved from
# MadCoreCUDA/cuda.jl).
function MadCore.default_options(::MadCore.AbstractNLPModel{T, VT}, ::Type{MadCore.SparseCondensedKKTSystem}, linear_solver::Type{CUDSSSolver}) where {T, VT <: CuVector{T}}
    opt = MadCore.default_options(linear_solver)
    # MadCore.set_options!(opt, Dict(:cudss_algorithm => MadCore.CHOLESKY)) # commented out due to issue #539
    return opt
end

export CUDSSSolver, GPUSchurComplementKKTSystem

end # module
