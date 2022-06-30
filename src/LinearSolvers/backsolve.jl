# MadNLP.jl
# Created by Sungho Shin (sungho.shin@wisc.edu)

"""
TODO
"""
struct RichardsonIterator{T, VT, KKT, LinSolver} <: AbstractIterator
    linear_solver::LinSolver
    kkt::KKT
    residual::VT
    max_iter::Int
    tol::T
    acceptable_tol::T
    logger::Logger
end
function RichardsonIterator(
    linear_solver::AbstractLinearSolver,
    kkt::AbstractKKTSystem,
    res::AbstractVector;
    max_iter=10, tol=1e-10, acceptable_tol=1e-5, logger=Logger(),
)
    return RichardsonIterator(
        linear_solver, kkt, res, max_iter, tol, acceptable_tol, logger,
    )
end

# Solve reduced KKT system. Require only the primal/dual values.
function solve_refine!(
    x::AbstractKKTVector{T, VT},
    solver::RichardsonIterator{T, VT, KKT},
    b::AbstractKKTVector{T, VT},
) where {T, VT, KKT<:AbstractReducedKKTSystem}
    solve_refine!(primal_dual(x), solver, primal_dual(b))
end

# Solve unreduced KKT system. Require UnreducedKKTVector as inputs.
function solve_refine!(
    x::UnreducedKKTVector{T, VT},
    solver::RichardsonIterator{T, VT, KKT},
    b::UnreducedKKTVector{T, VT},
) where {T, VT, KKT<:AbstractUnreducedKKTSystem}
    solve_refine!(full(x), solver, full(b))
end

function solve_refine!(
    x::AbstractVector{T},
    solver::RichardsonIterator{T},
    b::AbstractVector{T},
) where T
    @debug(solver.logger, "Iterative solver initiated")

    ε = solver.residual
    norm_b = norm(b, Inf)

    fill!(x, zero(T))
    fill!(ε, zero(T))

    ε = solver.residual
    axpy!(-1, b, ε)
    norm_res = norm(ε, Inf)
    residual_ratio = norm_res / (one(T) + norm_b)

    iter = 0
    residual_ratio_old = Inf
    noprogress = 0

    while true
        mod(iter, 10)==0 &&
            @debug(solver.logger,"iter ||res||")
        @debug(solver.logger, @sprintf("%4i %6.2e", iter, residual_ratio))
        iter += 1
        if (iter > solver.max_iter) || (residual_ratio < solver.tol)
            break
        end

        solve!(solver.linear_solver, ε)
        axpy!(-1, ε, x)
        mul!(ε, solver.kkt, x)
        axpy!(-1, b, ε)
        norm_res = norm(ε, Inf)

        residual_ratio_old = residual_ratio
        residual_ratio = norm_res / (one(T)+norm_b)
    end

    @debug(solver.logger, @sprintf(
        "Iterative solver terminated with %4i refinement steps and residual = %6.2e",
        iter, residual_ratio),
    )

    if residual_ratio < solver.acceptable_tol
        return :Solved
    else
        return :Failed
    end
end

