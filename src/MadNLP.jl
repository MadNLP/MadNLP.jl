# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

include(joinpath("..","deps","deps.jl"))

import Pkg.TOML: parsefile
import IterativeSolvers, MathOptInterface
import Libdl: dlopen, dlext, RTLD_DEEPBIND, RTLD_GLOBAL
import Metis: partition
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot
import LinearAlgebra.BLAS: libblas, liblapack, BlasInt, @blasfunc
import SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz
import Logging: @debug, @info,  @warn, @error
import Base: string, show, print, size, getindex, copyto!, @kwdef
import SuiteSparse: UMFPACK
import LightGraphs: Graph, Edge, add_edge!, edges, src, dst, neighbors, nv
import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, num_variables, getlinkconstraints, getnode
import JuMP: _create_nlp_block_data, set_optimizer, GenericAffExpr, backend, termination_status
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!
import SolverCore: GenericExecutionStats
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix, has_cuda_gpu, toolkit_version, R_64F

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
include("graphtools.jl")
include(joinpath("LinearSolvers","linearsolvers.jl"))
include(joinpath("interiorpointsolver.jl"))
include(joinpath("Interfaces","interfaces.jl"))

# Initialize
function __init__()
    check_deps()
    try
        @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
    catch e
        println("Pardiso shared library cannot be loaded")
    end
    set_blas_num_threads(Threads.nthreads(); permanent=true)
end

end # end module

