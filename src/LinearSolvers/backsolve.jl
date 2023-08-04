@kwdef struct RichardsonOptions
    max_iter::Int = 10
    tol::Float64 = 1e-10
    acceptable_tol::Float64 = 1e-5
end

struct RichardsonIterator{T, KKT <: AbstractKKTSystem{T}} <: AbstractIterator{T}
    kkt::KKT
    opt::RichardsonOptions
    cnt::MadNLPCounters
    logger::MadNLPLogger
end

function RichardsonIterator(
    kkt;
    opt = RichardsonOptions(),
    logger = MadNLPLogger(),
    cnt = MadNLPCounters()
)
    return RichardsonIterator(
        kkt, opt, cnt, logger
    )
end

function solve_refine!(
    x::VT,
    iterator::R,
    b::VT,
    w::VT
    ) where {T, VT, R <: RichardsonIterator{T}}
    @debug(iterator.logger, "Iterative solver initiated")

    norm_b = norm(full(b), Inf)
    residual_ratio = zero(T)

    fill!(full(x), zero(T))

    
    if norm_b == zero(T)
        @debug(
            iterator.logger,
            @sprintf(
                "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
                0, 0
            ),
        )
        return true
    end
    
    copyto!(full(w), full(b))
    iter = 0

    while true
        solve!(iterator.kkt, w)  # TODO this includes some extra time. Ideally, LinearSolver should count the time
        axpy!(1., full(w), full(x))
        copyto!(full(w), full(b))

        mul!(w, iterator.kkt, x, -one(T), one(T))

        norm_w = norm(full(w), Inf)
        norm_x = norm(full(x), Inf)
        residual_ratio = norm_w / (min(norm_x, 1e6 * norm_b) + norm_b)
        
        if mod(iter, 10)==0 
            @debug(iterator.logger,"iter ||res||")
        end
        @debug(iterator.logger, @sprintf("%4i %6.2e", iter, residual_ratio))
        iter += 1
        
        if (iter >= iterator.opt.max_iter) || (residual_ratio < iterator.opt.tol)
            break
        end
    end

    @debug(
        iterator.logger,
        @sprintf(
            "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
            iter, residual_ratio
        ),
    )

    return residual_ratio < iterator.opt.acceptable_tol
end

