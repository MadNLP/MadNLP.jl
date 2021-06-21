# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

abstract type AbstractLinearSolver end
mutable struct EmptyLinearSolver <: AbstractLinearSolver end
abstract type AbstractIterator end
abstract type AbstractLinearSystemScaler end

# dummy functions
introduce(::EmptyLinearSolver) = ""
factorize!(M::EmptyLinearSolver) = M
solve!(::EmptyLinearSolver,x) = x
is_inertia(::EmptyLinearSolver) = false
inertia(::EmptyLinearSolver) = (0,0,0)
improve!(::EmptyLinearSolver) = false
rescale!(::AbstractLinearSystemScaler) = nothing
solve_refine!(y,::AbstractIterator,x) = nothing

# LinearSolverExceptions 
struct SymbolicException <: Exception end
struct FactorizationException <: Exception end
struct SolveException <: Exception end
struct InertiaException <: Exception end
LinearSolverException=Union{SymbolicException,FactorizationException,SolveException,InertiaException}

# iterative solvers
include("richardson.jl")
include("krylov.jl")

# dense solvers
include("lapack.jl")

# direct solvers
include("umfpack.jl")
BLAS.vendor() == :mkl && include("pardisomkl.jl")
