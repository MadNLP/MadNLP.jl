# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

import IterativeSolvers, MathOptInterface, MPI, MKL_jll, OpenBLAS32_jll
import Libdl: dlopen, dlext, RTLD_DEEPBIND, RTLD_GLOBAL
import Metis: partition
import Parameters: @with_kw
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot
import SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz
import Logging: @debug, @info,  @warn, @error
import Base: string, show, print, size, getindex, copyto!
import StaticArrays: SVector, setindex
import SuiteSparse: UMFPACK
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix
import LightGraphs: Graph, Edge, add_edge!, edges, src, dst, neighbors, nv
import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, num_variables,
    getlinkconstraints
import JuMP: _create_nlp_block_data, set_optimizer, GenericAffExpr, backend, termination_status
import NLPModels: finalize, AbstractNLPModel,
    obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!
import SolverTools: GenericExecutionStats

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities

export madnlp

# Version info
version() = v"0.1.0"
introduce() = "MadNLP version $(version())"

# Linear solver dependencies
include("../deps/deps.jl")
include("enums.jl")
include("utils.jl")
include("nonlinearprogram.jl")
include("matrixtools.jl")
include("graphtools.jl")
include("LinearSolvers/linearsolvers.jl")
include("interiorpointsolver.jl")
include("Interfaces/interfaces.jl")

# Initialize
function __init__()
    check_deps()
    @isdefined(libmumps) && dlopen(libmumps,RTLD_DEEPBIND)
    @isdefined(libhsl) && dlopen(libhsl,RTLD_DEEPBIND)
    @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
    @isdefined(libmkl32) && dlopen.(joinpath.(MKL_jll.artifact_dir,[
        "lib/libmkl_core.$(dlext)",
        "lib/libmkl_sequential.$(dlext)",
        "lib/libmkl_intel_lp64.$(dlext)"]),RTLD_GLOBAL)
    @isdefined(libopenblas32) && dlopen(libopenblas32,RTLD_GLOBAL)
    set_blas_num_threads(haskey(ENV,"JULIA_NUM_THREADS") ? parse(Int,ENV["JULIA_NUM_THREADS"]) : 1 ;permanent=true)
end

const DefaultLinearSolver =default_linear_solver()
const DefaultSubproblemSolver = default_subproblem_solver()
const DefaultDenseSolver = default_dense_solver()

end # end module

