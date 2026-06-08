module MadNLP

using Reexport
@reexport using MadCore
@reexport using MadCoreLDLFactorizations
@reexport using MadCoreSuiteSparse

# Bring MadCore functions that MadNLP's IPM extends into scope so unqualified
# method definitions in IPM/*.jl extend the canonical MadCore generics.
import MadCore: solve_kkt!, initialize!, default_options, update!, introduce, mul_hess_blk!

# MadCore defines logger macros (@trace, @debug, etc.) that the IPM uses.
# Pull them in explicitly since `using` is the standard way to bring macros
# into scope from another module.
using MadCore: @trace, @debug, @info, @notice, @warn, @error

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra: BLAS, LAPACK, Adjoint, Symmetric, Diagonal,
    mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind,
    transpose!, issuccess, BlasReal,
    bunchkaufman, cholesky, qr, lu,
    bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse,
    getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!,
    jac_coord!, hess_coord!, hess_structure!, jac_structure!,
    hess_dense!, jac_dense!, NLPModelMeta,
    get_nvar, get_ncon, get_minimize, get_x0, get_y0,
    get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: getStatus, AbstractOptimizationSolver, AbstractExecutionStats

using PrecompileTools: @setup_workload, @compile_workload

export MadNLPSolver, MadNLPOptions, MadNLPExecutionStats, madnlp, madsuite

# Version info
version() = string(pkgversion(@__MODULE__))
# MadNLP-specific banner; overrides MadCore.introduce for the package itself.
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include("utils.jl")
include(joinpath("IPM", "IPM.jl"))
include("precompile.jl")

madsuite(::Val{:madnlp}, args...; kwargs...) = madnlp(args...; kwargs...)

global Optimizer

end # end module
