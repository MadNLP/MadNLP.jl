struct HS15Model <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
end

function HS15Model(; x0=zeros(2), y0=zeros(2))
    return HS15Model(
        NLPModels.NLPModelMeta(
            2,     #nvar
            ncon = 2,
            nnzj = 4,
            nnzh = 3,
            x0 = x0,
            y0 = y0,
            lvar = [-Inf, -Inf],
            uvar = [0.5, Inf],
            lcon = [1.0, 0.0],
            ucon = [Inf, Inf],
            minimize = true
        ),
        NLPModels.Counters()
    )
end

function NLPModels.obj(nlp::HS15Model, x::AbstractVector)
    return 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
end

function NLPModels.grad!(nlp::HS15Model, x::AbstractVector, g::AbstractVector)
    z = x[2] - x[1]^2
    g[1] = -400.0 * z * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * z
    return
end

function NLPModels.cons!(nlp::HS15Model, x::AbstractVector, c::AbstractVector)
    c[1] = x[1] * x[2]
    c[2] = x[1] + x[2]^2
end

function NLPModels.jac_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 1, 2, 2])
    copyto!(J, [1, 2, 1, 2])
end

function NLPModels.jac_coord!(nlp::HS15Model, x::AbstractVector, J::AbstractVector)
    J[1] = x[2]    # (1, 1)
    J[2] = x[1]    # (1, 2)
    J[3] = 1.0     # (2, 1)
    J[4] = 2*x[2]  # (2, 2)
    return J
end

function MadNLP.jac_dense!(nlp::HS15Model, x::AbstractVector, J::AbstractMatrix)
    J[1, 1] = x[2]    # (1, 1)
    J[1, 2] = x[1]    # (1, 2)
    J[2, 1] = 1.0     # (2, 1)
    J[2, 2] = 2*x[2]  # (2, 2)
    return J
end

function NLPModels.hess_structure!(nlp::HS15Model, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, [1, 2, 2])
    copyto!(J, [1, 1, 2])
end

function NLPModels.hess_coord!(nlp::HS15Model, x, y, H::AbstractVector; obj_weight=1.0)
    # Objective
    H[1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
    H[2] = obj_weight * (-400.0 * x[1])
    H[3] = obj_weight * 200.0
    # First constraint
    H[2] += y[1] * 1.0
    # Second constraint
    H[3] += y[2] * 2.0
    return H
end

function MadNLP.hess_dense!(nlp::HS15Model, x, y, H::AbstractMatrix; obj_weight=1.0)
    H[1, 1] = obj_weight * (-400.0 * x[2] + 1200.0 * x[1]^2 + 2.0)
    H[2, 1] = obj_weight * (-400.0 * x[1])
    H[2, 2] = obj_weight * 200.0
    # First constraint
    H[2, 1] += y[1] * 1.0
    # Second constraint
    H[2, 2] += y[2] * 2.0
    H[1, 2] = H[2, 1]
    return H
end

