module MadNLP

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot, diagind, normInf, transpose!, issuccess
import LinearAlgebra: cholesky, qr, lu, cholesky!, axpy!
import LinearAlgebra.BLAS: symv!, ger!, libblas, liblapack, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK, CHOLMOD
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, NLPModelMeta, get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: solve!, getStatus, AbstractOptimizationSolver, AbstractExecutionStats
export MadNLPSolver, MadNLPOptions, UmfpackSolver, LDLSolver, CHOLMODSolver, LapackCPUSolver, madnlp, solve!
import LDLFactorizations

# Version info
version() = string(pkgversion(@__MODULE__))
introduce() = "MadNLP version v$(version())"

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

end # end module
