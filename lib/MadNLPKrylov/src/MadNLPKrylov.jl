module MadNLPKrylov

import MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    AbstractOptions, AbstractIterator, set_options!, @sprintf,
    solve_refine!, mul!, ldiv!, size, StrideOneVector
import IterativeSolvers:
    FastHessenberg, ArnoldiDecomp, Residual, init!, init_residual!, expand!, Identity,
    orthogonalize_and_normalize!, update_residual!, gmres_iterable!, GMRESIterable, converged,
    ModifiedGramSchmidt

@kwdef mutable struct Options <: AbstractOptions
    krylov_restart::Int = 5
    krylov_max_iter::Int = 10
    krylov_tol::Float64 = 1e-10
    krylov_acceptable_tol::Float64 = 1e-5
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
ldiv!(Pl::VirtualPreconditioner,x::StrideOneVector{Float64}) = Pl.ldiv!(x)

mutable struct Solver <: AbstractIterator
    g::Union{Nothing,GMRESIterable}
    res::Vector{Float64}
    opt::Options
    logger::Logger
end

function Solver(res::Vector{Float64},_mul!,_ldiv!;
                opt=Options(),logger=Logger(),
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),kwargs...)
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)

    g = GMRESIterable(VirtualPreconditioner(_ldiv!),
                      Identity(),Float64[],Float64[],res,
                      ArnoldiDecomp(VirtualMatrix(_mul!,length(res),length(res)),opt.krylov_restart, Float64),
                      Residual(opt.krylov_restart, Float64),
                      0,opt.krylov_restart,1,
                      opt.krylov_max_iter,opt.krylov_tol,0.,
                      ModifiedGramSchmidt())

    return Solver(g,res,opt,logger)
end

function solve_refine!(x::StridedVector{Float64},
                       is::Solver,
                       b::AbstractVector{Float64})
    @debug(is.logger,"Iterator initiated")
    is.res.=0
    x.=0
    gmres_iterable_update!(is.g,x,b)

    noprogress = 0
    oldres = Inf
    iter = 0
    @debug(is.logger,"iter ||res||")
    @debug(is.logger,@sprintf("%4i %6.2e",iter,is.g.residual.current))
    for (~,res) in enumerate(is.g)
        iter += 1
        mod(iter,10)==0 && @debug(is.logger,"iter ||res||")
        @debug(is.logger,@sprintf("%4i %6.2e",iter,res))
        # res >= oldres && res > is.opt.krylov_acceptable_tol && (noprogress += 1)
        # oldres = res
        # if noprogress >= 3
        #     @debug(is.logger,@sprintf(
        #         "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        #         iter,res))
        #     return :Singular
        # end
    end
    @debug(is.logger,@sprintf(
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

end # module
