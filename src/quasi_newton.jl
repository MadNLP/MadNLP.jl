
"""
    AbstractHessian{T, VT}

Abstract type for representation of second-order information.

"""
abstract type AbstractHessian{T, VT} end

"""
    AbstractQuasiNewton{T, VT} <: AbstractHessian{T, VT}

Abstract type for quasi-Newton approximation.

"""
abstract type AbstractQuasiNewton{T, VT} <: AbstractHessian{T, VT} end

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


struct ExactHessian{T, VT} <: AbstractHessian{T, VT} end
ExactHessian{T, VT}(n::Int) where {T, VT} = ExactHessian{T, VT}()

"""
    BFGS{T, VT} <: AbstractQuasiNewton

BFGS quasi-Newton method. Update the direct Hessian approximation using
```math
B_{k+1} = B_k - \frac{(B_k s_k)(B_k s_k)^⊤}{s_k^⊤ B_k s_k} + \frac{y_k y_k^⊤}{y_k^⊤ s_k}
```

### Notes
The matrix is not updated if ``s_k^⊤ y_k < 10^{-8}``.

"""
struct BFGS{T, VT} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    sk::VT
    yk::VT
    bsk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end
function BFGS{T, VT}(n::Int; init_strategy=SCALAR1) where {T, VT}
    return BFGS{T, VT}(
        init_strategy,
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::BFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    if dot(sk, yk) < T(1e-8)
        return false
    end
    mul!(qn.bsk, Bk, sk)
    alpha1 = one(T) / dot(sk, qn.bsk)
    alpha2 = one(T) / dot(yk, sk)
    _ger!(-alpha1, qn.bsk, qn.bsk, Bk)  # Bk = Bk - alpha1 * bsk * bsk'
    _ger!(alpha2, yk, yk, Bk)           # Bk = Bk + alpha2 * yk * yk'
    return true
end

struct DampedBFGS{T, VT} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    sk::VT
    yk::VT
    bsk::VT
    rk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end
function DampedBFGS{T, VT}(n::Int; init_strategy=SCALAR1) where {T, VT}
    return DampedBFGS{T, VT}(
        init_strategy,
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::DampedBFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    mul!(qn.bsk, Bk, sk)
    sBs = dot(sk, qn.bsk)

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

    _ger!(-alpha1, qn.bsk, qn.bsk, Bk)
    _ger!(alpha2, qn.rk, qn.rk, Bk)
    return true
end

