module MadNLP

import Printf: @sprintf
import LinearAlgebra: BLAS, LAPACK, Adjoint, Symmetric, Diagonal, mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind, normInf, transpose!, issuccess
import LinearAlgebra: BlasReal, bunchkaufman, cholesky, qr, lu, bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
Base.USE_GPL_LIBS && import SuiteSparse: UMFPACK, CHOLMOD
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, hess_dense!, jac_dense!, NLPModelMeta, get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: getStatus, AbstractOptimizationSolver, AbstractExecutionStats, solve!
import LDLFactorizations
import MUMPS_seq_jll, OpenBLAS32_jll

export MadNLPSolver, MadNLPOptions, LDLSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!, madsuite
Base.USE_GPL_LIBS && export UmfpackSolver, CHOLMODSolver

function __init__()
    try
        config = BLAS.lbt_get_config()
        if !any(lib -> lib.interface == :lp64, config.loaded_libs)
            BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
        end
    catch e
        # In AOT-compiled binaries, JLL artifacts may not be available.
        # BLAS should already be linked statically in that case.
        @debug "BLAS lbt_forward skipped: $e"
    end
end
using PrecompileTools: @setup_workload, @compile_workload

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include(joinpath("Callbacks", "nlpmodels.jl"))
include(joinpath("Callbacks", "wrappers.jl"))
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers", "linearsolvers.jl"))
include(joinpath("IPM", "IPM.jl"))
include("precompile.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

# The Optimizer type is provided by the MadNLPMOI extension.
# We store it in a typed Ref to avoid untyped globals (AOT-incompatible).
const _optimizer_type = Ref{Type}(Nothing)

"""
    Optimizer(args...; kwargs...)

Create a MadNLP optimizer (requires MathOptInterface to be loaded).
"""
function Optimizer(args...; kwargs...)
    T = _optimizer_type[]
    if T === Nothing
        error("MadNLP.Optimizer requires MathOptInterface. Please run `using MathOptInterface` first.")
    end
    return T(args...; kwargs...)
end

end # end module
