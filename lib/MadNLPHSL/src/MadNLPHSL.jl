module MadNLPHSL

import MadNLP: @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz,
    get_tril_to_full, transfer!, input_type, _madnlp_unsafe_wrap,
    is_supported, default_options

import HSL
import HSL: mc68_control, mc68_info, ma77_control, ma77_info, ma86_control, ma86_info, ma97_control, ma97_info

import LinearAlgebra

include("common.jl")
include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver

end # module
