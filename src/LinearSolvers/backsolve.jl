struct RichardsonIterator{T, VT} <: AbstractIterator{T}
    residual::VT
    max_iter::Int
    tol::T
    acceptable_tol::T
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function RichardsonIterator(
    residual::UnreducedKKTVector{T};
    max_iter=10,
    tol=T(1e-10),
    acceptable_tol=T(1e-5),
    logger=MadNLPLogger(),
    cnt=MadNLPCounters()
) where T
    return RichardsonIterator(
        residual, max_iter, tol, acceptable_tol, cnt, logger
    )
end

function solve_refine!(
    x::VT,
    iterator::R,
    b::VT,
    solve!,
    mul!,
    ) where {T, VT, R <: RichardsonIterator{T, VT}}
    @debug(iterator.logger, "Iterative solver initiated")

    norm_b = norm(full(b), Inf)
    residual_ratio = zero(T)

    w = iterator.residual
    fill!(full(x), zero(T))
    copyto!(full(w), full(b))
    iter = 0

    while true
        iterator.cnt.linear_solver_time += @elapsed solve!(w)  # TODO this includes some extra time. Ideally, LinearSolver should count the time
        axpy!(1., full(w), full(x))
        copyto!(full(w), full(b))
        mul!(w, x, -one(T), one(T))
        
        norm_w = norm(full(w), Inf)
        norm_x = norm(full(x), Inf)
        residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)
        
        if mod(iter, 10)==0 
            @debug(iterator.logger,"iter ||res||")
        end
        @debug(iterator.logger, @sprintf("%4i %6.2e", iter, residual_ratio))
        iter += 1
        
        if (iter >= iterator.max_iter) || (residual_ratio < iterator.tol)
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

