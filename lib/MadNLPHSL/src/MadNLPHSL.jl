module MadNLPHSL

import MadNLP: @kwdef, MadNLPLogger, @debug, @warn, @error,
    AbstractOptions, AbstractLinearSolver, set_options!, SparseMatrixCSC, SubVector,
    SymbolicException,FactorizationException,SolveException,InertiaException,
    introduce, factorize!, solve!, improve!, is_inertia, inertia, findIJ, nnz,
    get_tril_to_full, transfer!, input_type, _madnlp_unsafe_wrap,
    is_supported, default_options
import HSL_jll: libhsl

include("common.jl")
include("mc68.jl")
include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver

end # module
