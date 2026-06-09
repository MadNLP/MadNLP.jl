module MadCoreMUMPS

import MadCore
import MadCore:
    AbstractOptions, AbstractLinearSolver, MadNLPLogger,
    SubVector, @kwdef, findIJ
import MadCore:
    factorize!, solve_linear_system!, is_inertia, inertia,
    input_type, default_options, improve!, introduce, is_supported,
    SymbolicException, FactorizationException, SolveException

import LinearAlgebra
import SparseArrays: SparseMatrixCSC, nnz, getcolptr, rowvals, nonzeros
import MUMPS_seq_jll

export MumpsSolver, MumpsOptions

include("mumps.jl")

end # module
