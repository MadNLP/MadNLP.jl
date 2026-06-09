module MadNLPCore

# The MadNLP interior-point solver, built on the lightweight MadCore. This is the
# bare solver: it depends only on MadCore (no external linear-solver backends), so
# its default sparse solver is the no-op `DummyLinearSolver`. The MadNLP package
# bundles the default solver backends (MUMPS/HSL/Pardiso/...) on top of this and
# specializes the default to `MumpsSolver`.

using Reexport
@reexport using MadCore

# Bring MadCore functions that the IPM extends into scope so unqualified method
# definitions in IPM/*.jl extend the canonical MadCore generics.
import MadCore: solve_kkt!, initialize!, default_options, update!, introduce, mul_hess_blk!

# MadCore defines logger macros (@trace, @debug, etc.) that the IPM uses.
using MadCore: @trace, @debug, @info, @notice, @warn, @error

import Pkg.TOML: parsefile
import Printf: @sprintf
import LinearAlgebra: BLAS, LAPACK, Adjoint, Symmetric, Diagonal,
    mul!, ldiv!, rdiv!, lmul!, rmul!, norm, dot, diagind,
    transpose!, issuccess, BlasReal,
    bunchkaufman, cholesky, qr, lu,
    bunchkaufman!, cholesky!, axpy!, LowerTriangular
import LinearAlgebra.BLAS: libblastrampoline, BlasInt, @blasfunc
import SparseArrays: SparseArrays, AbstractSparseMatrix, SparseMatrixCSC, sparse,
    getcolptr, rowvals, nnz, nonzeros
import Base: string, show, print, size, getindex, copyto!, @kwdef
import NLPModels
import NLPModels: finalize, AbstractNLPModel, obj, grad!, cons!,
    jac_coord!, hess_coord!, hess_structure!, jac_structure!,
    hess_dense!, jac_dense!, NLPModelMeta,
    get_nvar, get_ncon, get_minimize, get_x0, get_y0,
    get_nnzj, get_nnzh, get_lvar, get_uvar, get_lcon, get_ucon
import SolverCore: getStatus, AbstractOptimizationSolver, AbstractExecutionStats

# Version info
version() = string(pkgversion(@__MODULE__))
# Solver banner; specializes MadCore.introduce for the package itself.
introduce() = "\033[34mMad\033[31mN\033[32mL\033[35mP\033[0m version v$(version())"

include("utils.jl")
include(joinpath("IPM", "IPM.jl"))

# Auto-export every MadNLPCore-owned binding (mirrors MadCore). MadNLP
# `@reexport`s MadNLPCore, and the GPU backends reference IPM internals
# (is_dense_callback, get_tolerance, the MadNLPOptions constructor, ...) via
# `MadNLP.*`, so they must flow through. Skips gensyms, submodules, eval/include.
let
    skip = Set([:eval, :include, :MadNLPCore])
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
