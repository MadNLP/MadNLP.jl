struct DenseDummyQP{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VI <: AbstractVector{Int}
    } <: NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    P::MT # primal hessian
    A::MT # constraint jacobian
    q::VT
    buffer::VT
    hrows::VI
    hcols::VI
    jrows::VI
    jcols::VI
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
    mul!(qp.buffer, qp.P, x)
    return 0.5 * dot(x, qp.buffer) + dot(qp.q, x)
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
    index = 1
    for (i, j) in zip(qp.jrows, qp.jcols)
        J[index] = qp.A[i, j]
        index += 1
    end
end

function NLPModels.jprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A, v)
    return jv
end

function NLPModels.jtprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A', v)
    return jv
end


# Jacobian: dense callback
MadNLP.jac_dense!(qp::DenseDummyQP, x, J::AbstractMatrix) = copyto!(J, qp.A)
# Hessian: sparse callback
function NLPModels.hess_coord!(qp::DenseDummyQP{T},x, l, hess::AbstractVector; obj_weight=1.0) where T
    index = 1
    for i in 1:NLPModels.get_nvar(qp) , j in 1:i
        hess[index] = obj_weight * qp.P[j, i]
        index += 1
    end
end
# Hessian: dense callback
function MadNLP.hess_dense!(qp::DenseDummyQP{T}, x, l,hess::AbstractMatrix; obj_weight=1.0) where T
    copyto!(hess, obj_weight .* qp.P)
end

function DenseDummyQP(
    x0::VT = zeros(100);
    m=10, fixed_variables=similar(x0,Int,0), equality_cons=similar(x0,Int,0)
    ) where {T, VT <: AbstractVector{T}}
    
    n = length(x0)
    
    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)

    y0 = fill!(similar(x0, m), zero(T))
    q = copyto!(similar(x0, n), randn(n))
    buffer = similar(x0, n)

    # Bound constraints
    xl = fill!(similar(x0, n), zero(T))
    xu = fill!(similar(x0, n), one(T))
    gl = fill!(similar(x0, m), zero(T))
    gu = fill!(similar(x0, m), one(T))
    
    # Update gu to load equality constraints
    gu[equality_cons] .= zero(T)
    xl[fixed_variables] .= @view(xu[fixed_variables])

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = copyto!(similar(x0, n , n), randn(n,n))
    P = P*P' # P is symmetric
    P += T(100.0) * I


    # Build constraints gl <= Ax <= gu
    A = fill!(similar(x0, m, n), zero(T))
    A[1:m+1:m^2] .= one(T)
    A[m+1:m+1:m^2+m] .=-one(T)
    # for j in 1:m
    #     A[j, j]  = one(T)
    #     A[j, j+1]  = -one(T)
    # end

    nnzh = div(n * (n + 1), 2)
    hrows = copyto!(similar(x0, Int, nnzh), [i for i in 1:n for j in 1:i])
    hcols = copyto!(similar(x0, Int, nnzh), [j for i in 1:n for j in 1:i])

    nnzj = n * m
    jrows = copyto!(similar(x0, Int, nnzj), [j for i in 1:n for j in 1:m])
    jcols = copyto!(similar(x0, Int, nnzj), [i for i in 1:n for j in 1:m])

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
        P,A,q,buffer,
        hrows,hcols,jrows,jcols,
        NLPModels.Counters()
    )
end
