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
include("lapackcpu.jl")

# direct solvers
include("umfpack.jl")
include("mumps.jl")
blasvendor == :mkl && include("pardisomkl.jl")
if @isdefined libhsl
    include("mc68.jl")
    include("ma27.jl")
    include("ma57.jl")
    include("ma77.jl")
    include("ma86.jl")
    include("ma97.jl")
end
@isdefined(libpardiso) && include("pardiso.jl")


# decomposition solvers
include("schwarz.jl")
include("schur.jl")

# scalers
@isdefined(libhsl) && include("mc19.jl")

# generic functions - scalers
scale!(csc::SparseMatrixCSC{Float64},S::AbstractLinearSystemScaler)=scale!(csc.n,csc.colptr,csc.rowval,csc.nzval,S.s)
scale!(vec::AbstractArray{Float64,1},S::AbstractLinearSystemScaler) = (vec.*=S.s)
function scale!(n,colptr,rowval,nzval,s) where {Ti<:Integer}
    for i=1:n
        for j=colptr[i]:colptr[i+1]-1
            @inbounds nzval[j]*=s[i]*s[rowval[j]]
        end
    end
end
