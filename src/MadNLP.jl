# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

import Pkg.Artifacts: @artifact_str
import Pkg.TOML: parsefile
import IterativeSolvers, MathOptInterface, MPI
import Libdl: dlopen, dlext, RTLD_DEEPBIND, RTLD_GLOBAL
import Metis: partition
import Printf: @sprintf
import LinearAlgebra: BLAS, Adjoint, Symmetric, mul!, ldiv!, norm, dot
import SparseArrays: AbstractSparseMatrix, SparseMatrixCSC, sparse, getcolptr, rowvals, nnz
import Logging: @debug, @info,  @warn, @error
import Base: string, show, print, size, getindex, copyto!, @kwdef
import StaticArrays: SVector, setindex
import SuiteSparse: UMFPACK
import CUDA: CUBLAS, CUSOLVER, CuVector, CuMatrix, has_cuda_gpu
import LightGraphs: Graph, Edge, add_edge!, edges, src, dst, neighbors, nv
import Plasmo: OptiGraph, OptiNode, OptiEdge, all_nodes, all_edges, all_variables, num_all_nodes, num_variables,
    getlinkconstraints
import JuMP: _create_nlp_block_data, set_optimizer, GenericAffExpr, backend, termination_status
import NLPModels: finalize, AbstractNLPModel,
    obj, grad!, cons!, jac_coord!, hess_coord!, hess_structure!, jac_structure!
import SolverTools: GenericExecutionStats
import MUMPS_seq_jll: libdmumps_path
import OpenBLAS32_jll: libopenblas_path

const MOI = MathOptInterface
const MOIU = MathOptInterface.Utilities
const libdmumps = libdmumps_path
const libopenblas32 = libopenblas_path

export madnlp

# Version info
version() = parsefile(joinpath(@__DIR__,"..","Project.toml"))["version"]
introduce() = "MadNLP version v$(version())"

# Linear solver dependencies
include(joinpath("..","deps","deps.jl"))
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
    @isdefined(libhsl) && dlopen(libhsl,RTLD_DEEPBIND)
    @isdefined(libpardiso) && dlopen(libpardiso,RTLD_DEEPBIND)
    @isdefined(libmkl32) && dlopen.(joinpath.(artifact"MKL","lib",[
        "libmkl_core.$(dlext)",
        "libmkl_sequential.$(dlext)",
        "libmkl_intel_lp64.$(dlext)"]),RTLD_GLOBAL)
    @isdefined(libopenblas32) && dlopen(libopenblas32,RTLD_GLOBAL)
    set_blas_num_threads(Threads.nthreads() ;permanent=true)
end

end # end module



