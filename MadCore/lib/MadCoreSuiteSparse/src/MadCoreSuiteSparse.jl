module MadCoreSuiteSparse

# UMFPACK and CHOLMOD are gated on Julia being built with SuiteSparse (GPL).
# When USE_GPL_LIBS=false, this module loads but exposes no solver types.

import MadCore
import MadCore:
    AbstractOptions, AbstractLinearSolver, MadNLPLogger, SubVector,
    LinearFactorization, CHOLESKY, LDL,
    get_tril_to_full, @kwdef
import MadCore:
    factorize!, solve_linear_system!, is_inertia, inertia,
    input_type, default_options, improve!, introduce, is_supported,
    InertiaException

import LinearAlgebra: ldiv!, axpy!, issuccess
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nnz, sparse

if Base.USE_GPL_LIBS
    import SuiteSparse: UMFPACK, CHOLMOD
    export UmfpackSolver, UmfpackOptions, CHOLMODSolver, CHOLMODOptions
    include("umfpack.jl")
    include("cholmod.jl")
end

end # module
