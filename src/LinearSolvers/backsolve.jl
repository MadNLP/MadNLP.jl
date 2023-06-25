# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

struct RichardsonIterator{T, VT, KKT} <: AbstractIterator{T}
    kkt::KKT
    residual::VT
    max_iter::Int
    tol::T
    acceptable_tol::T
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function RichardsonIterator(
    kkt::AbstractKKTSystem{T},
    residual::UnreducedKKTVector{T};
    max_iter=10,
    tol=T(1e-8),
    acceptable_tol=T(1e-5),
    logger=MadNLPLogger(),
    cnt=MadNLPCounters()
) where T
    return RichardsonIterator(
        kkt, residual, max_iter, tol, acceptable_tol, cnt, logger
    )
end

function solve_refine!(
    x::UnreducedKKTVector{T, VT},
    iterator::RichardsonIterator{T},
    b::UnreducedKKTVector{T, VT},
) where {T, VT}
    @debug(iterator.logger, "Iterative solver initiated")
    kkt = iterator.kkt
    norm_b = norm(full(b), Inf)
    residual_ratio = 0.0

    w = iterator.residual
    fill!(full(x), 0)
    copyto!(full(w), full(b))
    iter = 0

    while true
        solve!(kkt,w,iterator.cnt)
        axpy!(1., full(w), full(x))
        copyto!(full(w), full(b))
        mul_subtract!(w, kkt, x)
        
        norm_w = norm(full(w), Inf)
        norm_x = norm(full(x), Inf)
        residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)
        
        mod(iter, 10)==0 &&
            @debug(iterator.logger,"iter ||res||")
        @debug(iterator.logger, @sprintf("%4i %6.2e", iter, residual_ratio))
        iter += 1
        if (iter > iterator.max_iter) || (residual_ratio < iterator.tol)
            break
        end
    end

    @debug(iterator.logger, @sprintf(
        "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        iter, residual_ratio),
           )

    if residual_ratio < iterator.acceptable_tol
        return true
    else
        return false
    end
end

