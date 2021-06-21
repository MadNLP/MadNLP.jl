# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

import Pkg.TOML: parsefile
import IterativeSolvers, MathOptInterface
import Libdl: dlopen, dlext, RTLD_DEEPBIND, RTLD_GLOBAL
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot
import LinearAlgebra.BLAS: libblas, liblapack, BlasInt, @blasfunc
import SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz
import Logging: @debug, @info,  @warn, @error
import Base: string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!
import SolverCore: GenericExecutionStats

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

export madnlp

# Version info
version() = parsefile(joinpath(@__DIR__,"..","Project.toml"))["version"]
introduce() = "MadNLP version v$(version())"

# Linear solver dependencies
include("enums.jl")
include("utils.jl")
include("nonlinearprogram.jl")
include("matrixtools.jl")
include(joinpath("LinearSolvers","linearsolvers.jl"))
include(joinpath("interiorpointsolver.jl"))
include(joinpath("Interfaces","interfaces.jl"))

# Initialize
function __init__()
    # check_deps()
    try
        @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
    catch e
        println("Pardiso shared library cannot be loaded")
    end
    set_blas_num_threads(Threads.nthreads(); permanent=true)
end

end # end module

