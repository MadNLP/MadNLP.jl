module MadNLPGPU

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nonzeros
import LinearAlgebra: Symmetric
# CUDA
import CUDA: CUDA, CUSPARSE, CUBLAS, CUSOLVER, CuVector, CuMatrix, CuArray, R_64F,
    has_cuda, @allowscalar, runtime_version, CUDABackend
import .CUSOLVER:
    libcusolver, cusolverStatus_t, CuPtr, cudaDataType, cublasFillMode_t, cusolverDnHandle_t, dense_handle
import .CUBLAS: handle, CUBLAS_DIAG_NON_UNIT,
    CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, CUBLAS_OP_T
import CUSOLVERRF

# Kernels
import Atomix
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

include("utils.jl")
include("KKT/dense.jl")
include("KKT/sparse.jl")
include("LinearSolvers/lapackgpu.jl")
include("LinearSolvers/cusolverrf.jl")
include("LinearSolvers/cudss.jl")

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

export LapackGPUSolver, CuCholeskySolver, RFSolver

# re-export MadNLP, including deprecated names
for name in names(MadNLP, all=true)
    if Base.isexported(MadNLP, name)
        @eval using MadNLP: $(name)
        @eval export $(name)
    end
end

end # module
