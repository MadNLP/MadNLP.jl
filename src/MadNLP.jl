module MadNLP

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot, diagind, normInf, transpose!, issuccess
import LinearAlgebra: cholesky, qr, lu, cholesky!, axpy!
import LinearAlgebra.BLAS: symv!, ger!, libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz, nonzeros
import Base: Base, string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK, CHOLMOD
import NLPModels: NLPModels, AbstractNLPModel, AbstractNLPModel, AbstractNLPModelMeta
# import NLPModels: finalize, AbstractNLPModel, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, NLPModelMeta, get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon

@nospecialize
@noinline Base.@nospecializeinfer obj(nlp::AbstractNLPModel, x::AbstractVector) = NLPModels.obj(Base.inferencebarrier(nlp), x)
@noinline Base.@nospecializeinfer grad!(nlp::AbstractNLPModel, x::AbstractVector, g::AbstractVector) = NLPModels.grad!(Base.inferencebarrier(nlp), x, g)
@noinline Base.@nospecializeinfer cons!(nlp::AbstractNLPModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(Base.inferencebarrier(nlp), x, c)
@noinline Base.@nospecializeinfer jac_coord!(nlp::AbstractNLPModel, x::AbstractVector, J::AbstractVector) = NLPModels.jac_coord!(Base.inferencebarrier(nlp), x, J)
@noinline Base.@nospecializeinfer hess_coord!(nlp::AbstractNLPModel, x::AbstractVector, l::AbstractVector, H::AbstractVector; obj_weight::Real) = NLPModels.hess_coord!(nlp, x, l, H; obj_weight)
@noinline Base.@nospecializeinfer hess_structure!(nlp::AbstractNLPModel, I::AbstractVector, J::AbstractVector) = NLPModels.hess_structure!(Base.inferencebarrier(nlp), I, J)
@noinline Base.@nospecializeinfer jac_structure!(nlp::AbstractNLPModel, I::AbstractVector, J::AbstractVector) = NLPModels.jac_structure!(Base.inferencebarrier(nlp), I, J)
# for f in (get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon)
#     f(nlp::AbstractNLPModel) = NLPModels.(f)(Base.inferencebarrier(nlp))
# end
for f in (:get_nvar, :get_nnzj, :get_nnzh, :get_ncon)
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModel) = NLPModels.$f(Base.inferencebarrier(nlp))::Int
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModelMeta) = NLPModels.$f(Base.inferencebarrier(nlp))::Int
end
for f in (:get_minimize,)
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModel) = NLPModels.$f(Base.inferencebarrier(nlp))::Bool
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModelMeta) = NLPModels.$f(Base.inferencebarrier(nlp))::Bool
end
for f in (:get_x0, :get_y0, :get_lvar, :get_uvar, :get_lcon, :get_ucon)
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModel) = NLPModels.$f(Base.inferencebarrier(nlp))
    @noinline Base.@nospecializeinfer @eval $f(nlp::AbstractNLPModelMeta) = NLPModels.$f(Base.inferencebarrier(nlp))
end
@specialize

import SolverCore: solve!, getStatus, AbstractOptimizationSolver, AbstractExecutionStats
import LDLFactorizations
import MUMPS_seq_jll, OpenBLAS32_jll
import Random

export MadNLPSolver, MadNLPOptions, UmfpackSolver, LDLSolver, CHOLMODSolver, LapackCPUSolver, MumpsSolver, MadNLPExecutionStats, madnlp, solve!

function __init__()
    config = BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
end
using PrecompileTools: @setup_workload, @compile_workload   

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include("nlpmodels.jl")
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers","linearsolvers.jl"))
include("options.jl")
include(joinpath("IPM", "IPM.jl"))
include("extension_templates.jl")
include("precompile.jl")

end # end module
