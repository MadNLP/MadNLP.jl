module MadCoreHSL

import MadCore:
    MadCore,
    @kwdef,
    MadNLPLogger,
    @debug,
    @warn,
    @error,
    AbstractOptions,
    AbstractLinearSolver,
    SparseMatrixCSC,
    SubVector,
    SymbolicException,
    FactorizationException,
    SolveException,
    InertiaException,
    introduce,
    factorize!,
    solve_linear_system!,
    improve!,
    is_inertia,
    inertia,
    findIJ,
    nnz,
    get_tril_to_full,
    transfer!,
    input_type,
    _madnlp_unsafe_wrap,
    is_supported,
    default_options

import HSL
import HSL:
    Mc68Control,
    Mc68Info,
    Ma77Control,
    Ma77Info,
    Ma86Control,
    Ma86Info,
    Ma97Control,
    Ma97Info

import LinearAlgebra

include("common.jl")
include("ma27.jl")
include("ma57.jl")
include("ma77.jl")
include("ma86.jl")
include("ma97.jl")

export Ma27Solver, Ma57Solver, Ma77Solver, Ma86Solver, Ma97Solver

end # module
