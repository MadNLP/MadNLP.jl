module MadNLPGPU

import LinearAlgebra
# CUDA
import CUDA: CUDA, CUSPARSE, CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, R_64F,
    has_cuda, @allowscalar, runtime_version, CUDABackend
import .CUSOLVER:
    libcusolver, cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t, dense_handle
import .CUBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T
import CUSOLVERRF

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

# AMD and Metis
import AMD, Metis

symul!(y, A, x::CuVector{T}, α = 1., β = 0.) where T = CUBLAS.symv!('L', T(α), A, x, T(β), y)
MadNLP._ger!(alpha::Number, x::CuVector{T}, y::CuVector{T}, A::CuMatrix{T}) where T = CUBLAS.ger!(alpha, x, y, A)
function MadNLP._madnlp_unsafe_wrap(vec::VT, n, shift=1) where {T, VT <: CuVector{T}}
    return view(vec,shift:shift+n-1)
end

include("kernels.jl")
include("interface.jl")
include("lapackgpu.jl")
include("cusolverrf.jl")
include("cudss.jl")

# option preset
function MadNLP.MadNLPOptions(
    nlp::AbstractNLPModel{T,VT};
    dense_callback = MadNLP.is_dense_callback(nlp),
    callback = dense_callback ? MadNLP.DenseCallback : MadNLP.SparseCallback,
    kkt_system = dense_callback ? MadNLP.DenseCondensedKKTSystem : MadNLP.SparseCondensedKKTSystem,
    linear_solver = dense_callback ? LapackGPUSolver : CUDSSSolver,
    tol = MadNLP.get_tolerance(T,kkt_system),
    ) where {T, VT <: CuVector{T}}
    
    return MadNLP.MadNLPOptions(
        tol = tol,
        callback = callback,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
    )
end

export LapackGPUSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
