module MadCore

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra:
    BLAS, LAPACK, Adjoint, Symmetric, Diagonal,
    mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind,
    transpose!, issuccess, BlasReal,
    bunchkaufman, cholesky, qr, lu,
    bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays:
    SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse,
    getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import NLPModels
import NLPModels:
    finalize, AbstractNLPModel, obj, grad!, cons!,
    jac_coord!, hess_coord!, hess_structure!, jac_structure!,
    hess_dense!, jac_dense!, NLPModelMeta,
    get_nvar, get_ncon, get_minimize, get_x0, get_y0,
    get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore:
    getStatus, AbstractOptimizationSolver, AbstractExecutionStats, solve!
import OpenBLAS32_jll

function __init__()
    config = BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
end

include("enums.jl")
include("utils.jl")
include("matrixtools.jl")
include(joinpath("Callbacks", "nlpmodels.jl"))
include(joinpath("Callbacks", "wrappers.jl"))
include("quasi_newton.jl")
include(joinpath("KKT", "KKTsystem.jl"))
include(joinpath("LinearSolvers", "linearsolvers.jl"))

# Public API. MadCore owns the solver-agnostic surface that used to live in the
# MadNLP monolith (KKT systems, core linear solvers, options, logger, quasi-
# Newton, callbacks, status). MadNLP `@reexport using MadCore`s these so the
# historical `MadNLP.*` names — and the unqualified names its IPM and the
# downstream solvers (MadIPM/MadNCL/CCopt) reach for — keep resolving.
#
# Rather than hand-curate a list (fragile: every internal name the IPM uses
# unqualified must appear, or precompile fails with UndefVarError), auto-export
# every top-level binding MadCore itself owns. This mirrors the old monolith,
# where all these names were top-level in one module.
#
# `names(MadCore, all=true, imported=false)` already lists ONLY MadCore's own
# top-level bindings (structs, functions, consts, macros, enum members) and
# EXCLUDES everything pulled in via `import X: ...` (BLAS, mul!, obj, ...). So
# that single filter is sufficient — we export every owned binding, minus
# gensyms (`#...`), submodules, and eval/include.
#
# Do NOT additionally filter on `parentmodule(val) === MadCore`: that wrongly
# drops MadCore-owned const TYPE ALIASES whose underlying type lives elsewhere
# (e.g. `const SubVector = SubArray{...}` -> parentmodule is Base), which the
# IPM uses unqualified -> precompile fails with `UndefVarError: SubVector`.
let
    skip = Set([:eval, :include, :MadCore])
    for name in names(@__MODULE__, all=true, imported=false)
        s = String(name)
        name in skip && continue
        startswith(s, "#") && continue
        isdefined(@__MODULE__, name) || continue
        getfield(@__MODULE__, name) isa Module && continue
        @eval export $name
    end
end

end # module
