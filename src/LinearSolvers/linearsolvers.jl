# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

#=
    LinearSolver's interface
=#
"""
    AbstractLinearSolver

Abstract type for linear solver targeting
the resolution of the linear system ``Ax=b``.
"""
abstract type AbstractLinearSolver end

"""
    introduce(::AbstractLinearSolver)

Print the name of the linear solver.
"""
function introduce end

"""
    factorize!(::AbstractLinearSolver)

Factorize the matrix ``A`` and updates the factors
inside the `AbstractLinearSolver` instance.
"""
function factorize! end

"""
    solve!(::AbstractLinearSolver, x)

Solve the linear system ``Ax = b``.

This function assumes the linear system has been
factorized previously with [`factorize!`](@ref).
"""
function solve! end

"""
    is_inertia(::AbstractLinearSolver)

Return `true` if the linear solver supports the
computation of the inertia of the linear system.
"""
function is_inertia end

"""
    inertia(::AbstractLinearSolver)

Return the inertia `(n, m, p)` of the linear system
as a tuple.

### Note
The inertia is defined as a tuple ``(n, m, p)``,
with
- ``n``: number of positive eigenvalues
- ``m``: number of negative eigenvalues
- ``p``: number of zero eigenvalues
"""
function inertia end

function improve! end

#=
    Iterator's interface
=#
abstract type AbstractIterator end

"""
    solve_refine!(x, ::AbstractIterator, b)

Solve the linear system ``Ax = b`` using iterative
refinement. The object `AbstractIterator` stores an instance
of a [`AbstractLinearSolver`](@ref) for the backsolve
operations.

### Notes
This function assumes the matrix stores in the linear solver
has been factorized previously.

"""
function solve_refine! end

# LinearSolverExceptions
struct SymbolicException <: Exception end
struct FactorizationException <: Exception end
struct SolveException <: Exception end
struct InertiaException <: Exception end
LinearSolverException=Union{SymbolicException,FactorizationException,SolveException,InertiaException}

# iterative solvers
include("backsolve.jl")

#=
    DEFAULT SOLVERS
=#

# dense solvers
include("lapack.jl")

# direct solvers
include("umfpack.jl")
