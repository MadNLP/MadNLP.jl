# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

import Pkg.TOML: parsefile
import MathOptInterface
import Libdl: dlopen, dlext, RTLD_DEEPBIND, RTLD_GLOBAL
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot
import LinearAlgebra.BLAS: axpy!, libblas, liblapack, BlasInt, @blasfunc
import SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz
import Base: string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!, NLPModelMeta, get_nvar, get_ncon, get_minimize, get_x0, get_y0, get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon, Counters as _Counters # get_zl,get_zu
import SolverCore: AbstractExecutionStats, getStatus

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const NLPModelsCounters = _Counters

export madnlp, UmfpackSolver, LapackCPUSolver

# Version info
version() = parsefile(joinpath(@__DIR__,"..","Project.toml"))["version"]
introduce() = "MadNLP version v$(version())"

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include("nlpmodels.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers","linearsolvers.jl"))
include("options.jl")
include(joinpath("IPM", "IPM.jl"))
include(joinpath("Interfaces","interfaces.jl"))

# Initialize
function __init__()
    try
        @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
    catch e
        println("Pardiso shared library cannot be loaded")
    end
end

end # end module

