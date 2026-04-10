using MadNLP
using MadNLP: MadNLPSolver, MadNLPOptions, MadNLPExecutionStats, madnlp
using NLPModels

struct HS15 <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
end

function HS15()
    return HS15(
        NLPModels.NLPModelMeta(
            2;
            ncon=2, nnzj=4, nnzh=3,
            x0=zeros(Float64, 2), y0=zeros(Float64, 2),
            lvar=Float64[-Inf, -Inf], uvar=Float64[0.5, Inf],
            lcon=Float64[1.0, 0.0], ucon=Float64[Inf, Inf],
            minimize=true,
        ),
        NLPModels.Counters(),
    )
end

function NLPModels.obj(::HS15, x::AbstractVector)
    return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function NLPModels.grad!(::HS15, x::AbstractVector, g::AbstractVector)
    z = x[2] - x[1]^2
    g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * z
    return g
end

function NLPModels.cons!(::HS15, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
    return c
end

function NLPModels.jac_structure!(::HS15, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
    return (I, J)
end

function NLPModels.jac_coord!(::HS15, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]; J[2] = x[1]; J[3] = 1.0; J[4] = 2 * x[2]
    return J
end

function NLPModels.hess_structure!(::HS15, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
    return (I, J)
end

function NLPModels.hess_coord!(::HS15, x, y, H::AbstractVector; obj_weight=1.0)
    H[1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
    H[2] = obj_weight * (-400.0 * x[1])
    H[3] = obj_weight * 200.0
    H[2] += y[1] * 1.0
    H[3] += y[2] * 2.0
    return H
end

function (@main)(args)
    nlp = HS15()
    result = madnlp(nlp; print_level=MadNLP.ERROR)
    println("Status: ", result.status)
    println("Objective: ", result.objective)
    println("Solution: ", result.solution)
    return 0
end
