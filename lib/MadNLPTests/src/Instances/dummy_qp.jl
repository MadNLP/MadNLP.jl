struct DenseDummyQP <: NLPModels.AbstractNLPModel{Float64,Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    P::Matrix{Float64} # primal hessian
    A::Matrix{Float64} # constraint jacobian
    q::Vector{Float64}
    hrows::Vector{Int}
    hcols::Vector{Int}
    jrows::Vector{Int}
    jcols::Vector{Int}
    counters::NLPModels.Counters
end

function NLPModels.jac_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
end
function NLPModels.hess_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
end

function NLPModels.obj(qp::DenseDummyQP, x::AbstractVector)
    return 0.5 * dot(x, qp.P, x) + dot(qp.q, x)
end
function NLPModels.grad!(qp::DenseDummyQP, x::AbstractVector, g::AbstractVector)
    mul!(g, qp.P, x)
    g .+= qp.q
    return
end
function NLPModels.cons!(qp::DenseDummyQP, x::AbstractVector, c::AbstractVector)
    mul!(c, qp.A, x)
end
# Jacobian: sparse callback
function NLPModels.jac_coord!(qp::DenseDummyQP, x::AbstractVector, J::AbstractVector)

    NLPModels.@lencheck NLPModels.get_nnzj(qp) J

    index = 1
    for (i, j) in zip(qp.jrows, qp.jcols)
        J[index] = qp.A[i, j]
        index += 1
    end
end
# Jacobian: dense callback
MadNLP.jac_dense!(qp::DenseDummyQP, x, J::AbstractMatrix) = copyto!(J, qp.A)
# Hessian: sparse callback
function NLPModels.hess_coord!(qp::DenseDummyQP,x, l, hess::AbstractVector; obj_weight=1.0)
    NLPModels.@lencheck NLPModels.get_nnzh(qp) hess

    index = 1
    for i in 1:NLPModels.get_nvar(qp) , j in 1:i
        hess[index] = obj_weight * qp.P[j, i]
        index += 1
    end
end
# Hessian: dense callback
function MadNLP.hess_dense!(qp::DenseDummyQP, x, l,hess::AbstractMatrix; obj_weight=1.0)
    copyto!(hess, obj_weight .* qp.P)
end

function DenseDummyQP(; n=100, m=10, fixed_variables=Int[], equality_cons=[])
    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = randn(n , n)
    P += P' # P is symmetric
    P += 100.0 * I

    q = randn(n)

    # Build constraints gl <= Ax <= gu
    A = zeros(m, n)
    for j in 1:m
        A[j, j]  = 1.0
        A[j, j+1]  = -1.0
    end

    x0 = zeros(n)
    y0 = zeros(m)

    # Bound constraints
    xu = fill(1.0, n)
    xl = fill(0.0, n)
    gl = fill(0.0, m)
    gu = fill(1.0, m)
    # Update gu to load equality constraints
    gu[equality_cons] .= 0.0

    xl[fixed_variables] .= xu[fixed_variables]

    hrows = [i for i in 1:n for j in 1:i]
    hcols = [j for i in 1:n for j in 1:i]
    nnzh = div(n * (n + 1), 2)

    jrows = [j for i in 1:n for j in 1:m]
    jcols = [i for i in 1:n for j in 1:m]
    nnzj = n * m

    return DenseDummyQP(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            minimize = true
        ),
        P,A,q,hrows,hcols,jrows,jcols,
        NLPModels.Counters()
    )
end

