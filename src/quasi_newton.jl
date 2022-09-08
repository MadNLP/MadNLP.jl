
@enum(BFGSInitStrategy::Int,
    SCALAR1  = 1,
    SCALAR2  = 2,
    SCALAR3  = 3,
    SCALAR4  = 4,
    CONSTANT = 5,
)

"""
    AbstractQuasiNewton

Abstract type for quasi-Newton algorithm.

"""
abstract type AbstractQuasiNewton end

"""
    update!(
        qn::AbstractQuasiNewton,
        Bk::AbstractMatrix,
        sk::AbstractVector,
        yk::AbstractVector,
    )

Update the matrix `Bk` encoding the (direct) Hessian approximation
with the secant vectors `sk` and `yk`.

Return `true` if the update succeeded, `false` otherwise.
"""
function update! end

"""
    BFGS{T, VT, MT} <: AbstractQuasiNewton

BFGS quasi-Newton method. Update the direct Hessian approximation using
```math
B_{k+1} = B_k - \frac{(B_k s_k)(B_k s_k)^⊤}{s_k^⊤ B_k s_k} + \frac{y_k y_k^⊤}{y_k^⊤ s_k}
```

### Notes
The matrix is not updated if ``s_k^⊤ y_k < 10^{-8}``.

"""
struct BFGS{T, VT, MT} <: AbstractQuasiNewton
    init_strategy::BFGSInitStrategy
    bsk::VT
end
function BFGS(n::Int; init_strategy=SCALAR1)
    return BFGS{Float64, Vector{Float64}, Matrix{Float64}}(init_strategy, zeros(n))
end

function update!(qn::BFGS{T, VT, MT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT, MT}
    if dot(sk, yk) < T(1e-8)
        return false
    end
    mul!(qn.bsk, Bk, sk)
    alpha1 = one(T) / dot(sk, qn.bsk)
    alpha2 = one(T) / dot(yk, sk)
    BLAS.ger!(-alpha1, qn.bsk, qn.bsk, Bk)  # Bk = Bk - alpha1 * bsk * bsk'
    BLAS.ger!(alpha2, yk, yk, Bk)           # Bk = Bk + alpha2 * yk * yk'
    return true
end

struct DampedBFGS{T, VT, MT} <: AbstractQuasiNewton
    init_strategy::BFGSInitStrategy
    bsk::VT
    rk::VT
end
function DampedBFGS(n::Int; init_strategy=SCALAR1)
    return DampedBFGS{Float64, Vector{Float64}, Matrix{Float64}}(init_strategy, zeros(n), zeros(n))
end

function update!(qn::DampedBFGS{T, VT, MT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT, MT}
    mul!(qn.bsk, Bk, sk)
    sBs = dot(sk, qn.Bsk)

    # Procedure 18.2 (Nocedal & Wright, page 537)
    theta = if dot(sk, yk) < T(0.2) * sBs
        T(0.8) * sBs / (sBs - dot(sk, yk))
    else
        one(T)
    end

    fill!(qn.rk, zero(T))
    axpy!(theta, yk, qn.rk)
    axpy!(one(T) - theta, qn.bsk, qn.rk)

    alpha1 = one(T) / sBs
    alpha2 = one(T) / dot(qn.rk, qn.sk)

    BLAS.ger!(-alpha1, qn.bsk, qn.bsk, Bk)
    BLAS.ger!(alpha2, qn.rk, qn.rk, Bk)
    return true
end

