module MadCoreCUDA

# CUDA backend for MadCore: GPU dense/sparse linear solvers (LapackCUDASolver via
# cuSOLVER, CUDSSSolver via CUDSS) and the GPU Schur solver. Migrated from
# MadNLPGPU/ext/MadNLPGPUCUDAExt and promoted from a weakdep extension to a
# standalone package — the CUDA stack is a hard dependency here. Builds on
# MadCore + MadCoreKernelAbstractions. IPM-specific GPU code lives in
# MadNLP/lib/cuMadNLP.

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros, nnz
import LinearAlgebra: Symmetric, mul!

import MadCore
import MadCore:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException, FactorizationException, SolveException, InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options,
    setup_cholesky!, setup_lu!, setup_qr!, setup_evd!, setup_bunchkaufman!,
    factorize_cholesky!, factorize_lu!, factorize_qr!, factorize_evd!, factorize_bunchkaufman!,
    solve_cholesky!, solve_lu!, solve_qr!, solve_evd!, solve_bunchkaufman!

import MadCoreKernelAbstractions
import MadCoreKernelAbstractions: ORDERING, DEFAULT_ORDERING, METIS_ORDERING, AMD_ORDERING,
    USER_ORDERING, SYMAMD_ORDERING, COLAMD_ORDERING, gpu_transfer!

import KernelAbstractions: @kernel, @index, @Const, synchronize, get_backend

using CUDACore
using cuSPARSE, cuBLAS, cuSOLVER
import CUDSS

import AMD, Metis

import .cuSOLVER: cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t,
    dense_handle, CuSolverParameters, CUSOLVER_EIG_MODE_VECTOR
import .cuBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T

include("cuda_sparse.jl")
include("lapackgpu.jl")
include("cusolver.jl")
include("cudss.jl")
include("cuda.jl")
include("kernels_schur.jl")
include("cuda_schur.jl")

export LapackCUDASolver, CUDSSSolver

end # module
