# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLP

using LinearAlgebra, SparseArrays, Libdl, Printf,
    Metis, LightGraphs,Memento, Parameters, MKL_jll, OpenBLAS32_jll

import Base: string, show, print, size, getindex

export NonlinearProgram,InteriorPointSolver,plasmonlp

# Linear solver dependencies
include("../deps/deps.jl")
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
    Memento.register(LOGGER)

    set_blas_num_threads(haskey(ENV,"JULIA_NUM_THREADS") ? parse(Int,ENV["JULIA_NUM_THREADS"]) : 1 ;permanent=true)
end

const DefaultLinearSolver =default_linear_solver()
const DefaultSubproblemSolver = default_subproblem_solver()
const DefaultDenseSolver = default_dense_solver()

end # end module
