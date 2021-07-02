# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

module MadNLPRichardson

import ..MadNLP:
    @kwdef, Logger, @debug, @warn, @error,
    norm,
    AbstractOptions, AbstractIterator, set_options!, @sprintf, StrideOneVector,
    solve_refine!

@kwdef mutable struct Options <: AbstractOptions
    richardson_max_iter::Int = 10
    richardson_tol::Float64 = 1e-10
    richardson_acceptable_tol::Float64 = 1e-5
end

mutable struct Solver <: AbstractIterator
    res::Vector{Float64}
    mul!::Function
    div!::Function
    opt::Options
    logger::Logger
end
function Solver(res::Vector{Float64},mul!,div!;
                opt=Options(),logger=Logger(),
                option_dict::Dict{Symbol,Any}=Dict{Symbol,Any}(),kwargs...)
    !isempty(kwargs) && (for (key,val) in kwargs; option_dict[key]=val; end)
    set_options!(opt,option_dict)
    return Solver(res,mul!,div!,opt,logger)
end

function solve_refine!(x::StrideOneVector{Float64},
                       IS::Solver,
                       b::AbstractVector{Float64})
    @debug(IS.logger,"Iterative solver initiated")
    norm_b = norm(b,Inf)
    # x.=b
    # IS.div!(x)
    # IS.mul!(IS.res,x)
    x.=0
    IS.res=.-b
    norm_res = norm(IS.res,Inf)
    residual_ratio = norm_res/(1+norm_b)

    iter   = 0
    residual_ratio_old = Inf
    noprogress = 0

    while true
        mod(iter,10)==0 &&
            @debug(IS.logger,"iter ||res||")
        @debug(IS.logger,@sprintf("%4i %6.2e",iter,residual_ratio))
        iter += 1
        iter > IS.opt.richardson_max_iter && break
        residual_ratio < IS.opt.richardson_tol && break
        # residual_ratio>=residual_ratio_old && residual_ratio>IS.opt.richardson_acceptable_tol && (noprogress+=1)
        # if noprogress >= 3
        #     @debug(logger,@sprintf(
        #         "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        #         iter,residual_ratio))
        #     return :Singular
        # end

        IS.div!(IS.res)
        x.-=IS.res
        IS.mul!(IS.res,x)
        IS.res.-=b
        norm_res = norm(IS.res,Inf)

        residual_ratio_old = residual_ratio
        residual_ratio = norm_res/(1+norm_b)
    end

    @debug(IS.logger,@sprintf(
        "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        iter,residual_ratio))

    return (residual_ratio < IS.opt.richardson_acceptable_tol ? :Solved : :Failed)
end
end # module
