# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module Krylov

using Memento,IterativeSolvers

const LOGGER=getlogger(@__MODULE__)
__init__() = Memento.register(LOGGER)

using LinearAlgebra, Parameters
import ..MadNLP:
    AbstractOptions, AbstractIterator, set_options!, @sprintf, @printf,
    solve_refine!
import IterativeSolvers:
    FastHessenberg, ArnoldiDecomp, Residual, init!, init_residual!, expand!,
    orthogonalize_and_normalize!, update_residual!, gmres_iterable!, GMRESIterable, converged
import LinearAlgebra: mul!, ldiv!
import Base: size

@with_kw mutable struct Options <: AbstractOptions
    krylov_restart::Int = 5
    krylov_max_iter::Int = 10
    krylov_tol::Float64 = 1e-10
    krylov_acceptable_tol::Float64 = 1e-5
    krylov_log_level::String=""
end

struct VirtualMatrix
    mul!::Function
    m::Int
    n::Int
end
mul!(y,A::VirtualMatrix,x) = A.mul!(y,x)
size(A::VirtualMatrix,i) = (A.m,A.n)[i]

struct VirtualPreconditioner
    ldiv!::Function
end
ldiv!(Pl::VirtualPreconditioner,x) = Pl.ldiv!(x)

mutable struct Solver <: AbstractIterator
    g::Union{Nothing,GMRESIterable}
    res::Vector{Float64}
    opt::Options
end

function Solver(res::Vector{Float64},_mul!,_ldiv!;
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),kwargs...)
    opt=Options()
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)
    opt.krylov_log_level=="" || setlevel!(LOGGER,opt.krylov_log_level)

    g=GMRESIterable(VirtualPreconditioner(_ldiv!), 
                    Identity(),Float64[],Float64[],res,
                    ArnoldiDecomp(VirtualMatrix(_mul!,length(res),length(res)),opt.krylov_restart, Float64),
                    Residual(opt.krylov_restart, Float64),
                    0,opt.krylov_restart,1,
                    opt.krylov_max_iter,opt.krylov_tol,0.)
    
    return Solver(g,res,opt)
end

function solve_refine!(x::StridedVector{Float64},
                       is::Solver,
                       b::AbstractVector{Float64})
    debug(LOGGER,"Iterator initiated")
    is.res.=0
    x.=0
    gmres_iterable_update!(is.g,x,b)
    
    noprogress = 0
    oldres = Inf
    iter = 0
    debug(LOGGER,"iter ||res||")
    debug(LOGGER,@sprintf("%4i %6.2e",iter,is.g.residual.current))
    for (~,res) in enumerate(is.g)
        iter += 1
        mod(iter,10)==0 && debug(LOGGER,"iter ||res||")
        debug(LOGGER,@sprintf("%4i %6.2e",iter,res))
        # res >= oldres && res > is.opt.krylov_acceptable_tol && (noprogress += 1)
        # oldres = res
        # if noprogress >= 3
        #     debug(LOGGER,@sprintf(
        #         "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        #         iter,res))
        #     return :Singular
        # end
    end
    debug(LOGGER,@sprintf(
        "Iterative solver terminated with %4i refinement steps and residual = %6.2e",iter,is.g.residual.current))

    return (is.g.residual.current < is.opt.krylov_acceptable_tol ? :Solved : :Failed)
end

function gmres_iterable_update!(g,x,b)
    g.x=x
    g.b=b
    g.mv_products = 0
    g.residual.current = init!(g.arnoldi,g.x,g.b,g.Pl,g.Ax,initially_zero = true)
    init_residual!(g.residual, g.residual.current)
    g.k=1
    g.β=g.residual.current
end


end

# forgiving names
krylov = Krylov
KRYLOV = Krylov
