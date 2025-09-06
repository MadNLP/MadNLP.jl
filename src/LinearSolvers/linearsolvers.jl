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
abstract type AbstractLinearSolver{T} end

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
    solve!(::AbstractLinearSolver, x::AbstractVector)

Solve the linear system ``Ax = b``.

This function assumes the linear system has been
factorized previously with [`factorize!`](@ref).
"""
function solve! end

"""
    is_supported(solver,T)

Return `true` if `solver` supports the floating point
number type `T`.

# Examples
```julia-repl
julia> is_supported(UmfpackSolver,Float64)
true

julia> is_supported(UmfpackSolver,Float32)
false
```
"""
function is_supported(::Type{LS},::Type{T}) where {LS <: AbstractLinearSolver, T <: AbstractFloat}
    return false
end

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

# Default function for AbstractKKTVector
function solve!(s::AbstractLinearSolver, x::AbstractKKTVector)
    solve!(s, full(x))
end

function multi_solve!(s::AbstractLinearSolver, X::AbstractMatrix)
    n, nrhs = size(X)
    x = zeros(n)
    for i in 1:nrhs
        copyto!(x, 1, X, (i-1)*n + 1, n)
        solve!(s, x)
        copyto!(X, (i-1)*n + 1, x, 1, n)
    end
end

#=
    Iterator's interface
=#
abstract type AbstractIterator{T} end

"""
    solve_refine!(x::VT, ::AbstractIterator, b::VT, w::VT) where {VT <: AbstractKKTVector}

Solve the linear system ``Ax = b`` using iterative
refinement. The object `AbstractIterator` stores an instance
of a [`AbstractLinearSolver`](@ref) for the backsolve
operations.

### Notes
This function assumes the matrix stored in the linear solver
has been factorized previously.

"""
function solve_refine! end

# LinearSolverExceptions
struct SymbolicException <: Exception end
struct FactorizationException <: Exception end
struct SolveException <: Exception end
struct InertiaException <: Exception end
LinearSolverException=Union{SymbolicException,FactorizationException,SolveException,InertiaException}

@enum(
    LinearFactorization::Int,
    BUNCHKAUFMAN = 1,
    LU = 2,
    QR = 3,
    CHOLESKY = 4,
    LDL = 5,
    EVD = 6,
)

# iterative solvers
include("backsolve.jl")

#=
    DEFAULT SOLVERS
=#

# dense solvers
include("lapack.jl")
include("umfpack.jl")
include("cholmod.jl")
include("ldl.jl")
