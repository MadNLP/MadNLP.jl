module MadNLPGPU

import LinearAlgebra
# CUDA
import CUDA: CUDA, CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, R_64F, has_cuda, @allowscalar, runtime_version
import .CUSOLVER:
    libcusolver, cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t, dense_handle
import .CUBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T

# Kernels
import KernelAbstractions: @kernel, @index, wait, Event
import CUDAKernels: CUDADevice

import MadNLP
import MadNLP:
    @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, AbstractNLPModel, set_options!,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, tril_to_full!,
    LapackOptions, input_type, is_supported, default_options, symul!

symul!(y, A, x::CuVector{T}, α = 1., β = 0.) where T = CUBLAS.symv!('L', T(α), A, x, T(β), y)
MadNLP._ger!(alpha::Number, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where T = CUBLAS.ger!(alpha, x, y, A)


include("kernels.jl")
include("callbacks.jl")

if has_cuda()
    include("lapackgpu.jl")
    export LapackGPUSolver
end
export CuMadNLPSolver

include("interface.jl")

end # module
