
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

curvature(::Val{SCALAR1}, sk, yk) = dot(yk, sk) / dot(sk, sk)
curvature(::Val{SCALAR2}, sk, yk) = dot(yk, yk) / dot(sk, yk)
curvature(::Val{SCALAR3}, sk, yk) = 0.5 * (curvature(Val(SCALAR1), sk, yk) + curvature(Val(SCALAR2), sk, yk))
curvature(::Val{SCALAR4}, sk, yk) = sqrt(curvature(Val(SCALAR1), sk, yk) * curvature(Val(SCALAR2), sk, yk))


"""
    BFGS{T, VT} <: AbstractQuasiNewton{T, VT}

BFGS quasi-Newton method. Update the direct Hessian approximation using
```math
B_{k+1} = B_k - \frac{(B_k s_k)(B_k s_k)^⊤}{s_k^⊤ B_k s_k} + \frac{y_k y_k^⊤}{y_k^⊤ s_k}
```

### Notes
The matrix is not updated if ``s_k^⊤ y_k < 10^{-8}``.

"""
struct BFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    is_instantiated::Base.RefValue{Bool}
    sk::VT
    yk::VT
    bsk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end

function create_quasi_newton(
    ::Type{BFGS},
    cb::AbstractCallback{T,VT},
    n;
    init_strategy = SCALAR1
    ) where {T,VT}
    BFGS(
        init_strategy,
        Ref(false),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
        VT(undef, n),
    )
end

function update!(qn::BFGS{T, VT}, Bk::AbstractMatrix, sk::AbstractVector, yk::AbstractVector) where {T, VT}
    yksk = dot(sk, yk)
    if yksk < T(1e-8)
        return false
    end
    # Initial approximation (Nocedal & Wright, p.143)
    if !qn.is_instantiated[]
        sksk = dot(sk, sk)
        Bk[diagind(Bk)] .= yksk ./ sksk
        qn.is_instantiated[] = true
    end
    # BFGS update
    mul!(qn.bsk, Bk, sk)
    alpha1 = one(T) / dot(sk, qn.bsk)
    alpha2 = one(T) / yksk
    _ger!(-alpha1, qn.bsk, qn.bsk, Bk)  # Bk = Bk - alpha1 * bsk * bsk'
    _ger!(alpha2, yk, yk, Bk)           # Bk = Bk + alpha2 * yk * yk'
    return true
end

struct DampedBFGS{T, VT <: AbstractVector{T}} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    is_instantiated::Base.RefValue{Bool}
    sk::VT
    yk::VT
    bsk::VT
    rk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
end

function create_quasi_newton(
    ::Type{DampedBFGS},
    cb::AbstractCallback{T,VT},
    n;
    init_strategy = SCALAR1
    ) where {T,VT}
    return DampedBFGS(
        init_strategy,
        Ref(false),
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
    yksk = dot(sk, yk)
    # Initial approximation (Nocedal & Wright, p.143)
    if !qn.is_instantiated[]
        sksk = dot(sk, sk)
        Bk[diagind(Bk)] .= yksk ./ sksk
        qn.is_instantiated[] = true
    end

    mul!(qn.bsk, Bk, sk)
    sBs = dot(sk, qn.bsk)

    # Procedure 18.2 (Nocedal & Wright, page 537)
    theta = if dot(sk, yk) < T(0.2) * sBs
        T(0.8) * sBs / (sBs - yksk)
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

function init!(qn::Union{BFGS, DampedBFGS}, Bk::AbstractMatrix{T}, g0::AbstractVector{T}, f0::T) where T
    norm_g0 = dot(g0, g0)
    # Initiate B0 with Gilbert & Lemaréchal rule.
    rho0 = if norm_g0 < sqrt(eps(T))
        one(T)
    elseif f0 ≈ zero(T)
        one(T) / norm_g0
    else
        abs(f0) / norm_g0
    end
    Bk[diagind(Bk)] .= T(2) * rho0
    return
end


"""
    CompactLBFGS{T, VT} <: AbstractQuasiNewton
"""
mutable struct CompactLBFGS{T, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}} <: AbstractQuasiNewton{T, VT}
    init_strategy::BFGSInitStrategy
    sk::VT
    yk::VT
    last_g::VT
    last_x::VT
    last_jv::VT
    init_value::T
    sigma_min::T
    sigma_max::T
    max_mem::Int
    current_mem::Int
    skipped_iter::Int
    Sk::MT       # n x p
    Yk::MT       # n x p
    Lk::MT       # p x p
    Mk::MT       # p x p (for Cholesky factorization Mₖ = Jₖᵀ Jₖ)
    Tk::MT       # 2p x 2p
    Jk::MT       # p x p
    SdotS::MT    # p x p
    DkLk::MT     # p x p
    U::MT        # n x 2p
    V1::MT       # m x 2p
    V2::MT       # m x 2p
    Dk::VT       # p
    _w1::VT
    _w2::VT
end

function create_quasi_newton(
    ::Type{CompactLBFGS},
    cb::AbstractCallback{T,VT},
    n;
    max_mem=6,
    sigma_min=1e-8,
    sigma_max=1e8,
    init_value=one(T),
    init_strategy = SCALAR1
    ) where {T, VT}
    return CompactLBFGS(
        init_strategy,
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        fill!(create_array(cb, n), zero(T)),
        init_value,
        sigma_min,
        sigma_max,
        max_mem,
        0,
        0,
        fill!(create_array(cb, n, 0), zero(T)),
        fill!(create_array(cb, n, 0), zero(T)),
        fill!(create_array(cb, n, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
        fill!(create_array(cb, 0), zero(T)),
    )
end

Base.size(qn::CompactLBFGS) = (size(qn.Sk, 1), qn.current_mem)

function _resize!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    n, k = size(qn)
    qn.Lk     = zeros(T, k, k)
    qn.SdotS  = zeros(T, k, k)
    qn.Mk     = zeros(T, k, k)
    qn.Jk     = zeros(T, k, k)
    qn.Tk     = zeros(T, 2*k, 2*k)
    qn.DkLk   = zeros(T, k, k)
    qn.U      = zeros(T, n, 2*k)
    qn._w1    = zeros(T, k)
    qn._w2    = zeros(T, 2*k)
    return
end

function _reset!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    n, _ = size(qn)
    qn.current_mem = 0
    qn.skipped_iter = 0
    fill!(qn.last_jv, zero(T))
    qn.Dk  = zeros(T, 0)
    qn.Sk  = zeros(T, n, 0)
    qn.Yk  = zeros(T, n, 0)
    _resize!(qn)
end

# augment / shift
function _update_SY!(qn::CompactLBFGS, s, y)
    if qn.current_mem < qn.max_mem
        qn.current_mem += 1
        qn.Sk = hcat(qn.Sk, s)
        qn.Yk = hcat(qn.Yk, y)
        _resize!(qn)
    else
        n, k = size(qn)
        # Shift
        @inbounds for i_ in 1:k-1, j in 1:n
            qn.Sk[j, i_] = qn.Sk[j, i_+1]
            qn.Yk[j, i_] = qn.Yk[j, i_+1]
        end
        # Latest element
        @inbounds for j in 1:n
            qn.Sk[j, k] = s[j]
            qn.Yk[j, k] = y[j]
        end
    end
end

function _refresh_D!(qn::CompactLBFGS, sk, yk)
    k = qn.current_mem
    sTy = dot(sk, yk)
    if length(qn.Dk) < qn.max_mem
        push!(qn.Dk, sTy)
    else
        # shift
        @inbounds for i in 1:k-1
            qn.Dk[i] = qn.Dk[i+1]
        end
        qn.Dk[k] = sTy
    end
end

function _refresh_L!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    p = size(qn.Lk, 1)
    mul!(qn.Lk, qn.Sk', qn.Yk)
    @inbounds for i in 1:p, j in i:p
        qn.Lk[i, j] = zero(T)
    end
end

function _refresh_STS!(qn::CompactLBFGS{T, VT, MT}) where {T, VT, MT}
    mul!(qn.SdotS, qn.Sk', qn.Sk, one(T), zero(T))
end

function update!(qn::CompactLBFGS{T, VT, MT}, Bk, sk, yk) where {T, VT, MT}
    norm_sk, norm_yk = norm(sk), norm(yk)
    # Skip update if vectors are too small or local curvature is negative.
    if ((norm_sk < T(100) * eps(T)) ||
        (norm_yk < T(100) * eps(T)) ||
        (dot(sk, yk) < sqrt(eps(T)) * norm_sk * norm_yk)
    )
        qn.skipped_iter += 1
        if qn.skipped_iter >= 2
            _reset!(qn)
        end
        return false
    end

    # Refresh internal structures
    _update_SY!(qn, sk, yk)
    _refresh_D!(qn, sk, yk)
    _refresh_L!(qn)
    _refresh_STS!(qn)

    # Load buffers
    k = qn.current_mem
    δ = qn._w1

    # Compute compact representation Bₖ = σₖ I + Uₖ Vₖᵀ
    #       Uₖ = [ U₁ ]     Vₖ = [ -U₁ ]
    #            [ U₂ ]          [  U₂ ]

    # Step 1: σₖ I
    sigma = curvature(Val(qn.init_strategy), sk, yk)  # σₖ
    sigma = clamp(sigma, qn.sigma_min, qn.sigma_max)
    Bk .= sigma                                       # Hₖ .= σₖ I (diagonal Hessian approx.)

    # Step 2: Mₖ = σₖ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ
    qn.DkLk .= (one(T) ./ qn.Dk) .* qn.Lk'            # DₖLₖ = Dₖ⁻¹ Lₖᵀ
    qn.Mk .= qn.SdotS                                 # Mₖ = Sₖᵀ Sₖ
    mul!(qn.Mk, qn.Lk, qn.DkLk, one(T), sigma)        # Mₖ = σₖ Sₖᵀ Sₖ + Lₖ Dₖ⁻¹ Lₖᵀ
    symmetrize!(qn.Mk)

    copyto!(qn.Jk, qn.Mk)
    cholesky!(qn.Jk)                                  # Mₖ = Jₖᵀ Jₖ (factorization)

    # Step 3: Nₖ = [U₁ U₂]
    U1 = view(qn.U, :, 1:k)
    copyto!(U1, qn.Sk)                                # U₁ = Sₖ
    mul!(U1, qn.Yk, qn.DkLk, one(T), sigma)           # U₁ = σₖ Sₖ + Yₖ Dₖ⁻¹ Lₖ
    BLAS.trsm!('R', 'U', 'N', 'N', one(T), qn.Jk, U1) # U₁ = Jₖ⁻ᵀ (σₖ Sₖ + Yₖ Dₖ⁻¹ Lₖ)
    U2 = view(qn.U, :, 1+k:2*k)
    δ .= .-one(T) ./ sqrt.(qn.Dk)                     # δ = 1 / √Dₖ
    U2 .= δ' .* qn.Yk                                 # U₂ = (1 / √Dₖ) * Yₖ
    return true
end

function init!(qn::CompactLBFGS{T}, Bk::AbstractVector{T}, g0::AbstractVector{T}, f0::T) where T
    norm_g0 = dot(g0, g0)
    # Initiate B0 with Gilbert & Lemaréchal rule.
    rho0 = if norm_g0 < sqrt(eps(T))
        one(T)
    elseif f0 ≈ zero(T)
        one(T) / norm_g0
    else
        abs(f0) / norm_g0
    end
    Bk .= (T(2) * rho0 * qn.init_value)
    return
end


struct ExactHessian{T, VT} <: AbstractHessian{T, VT} end
create_quasi_newton(::Type{ExactHessian}, cb::AbstractCallback{T,VT}, n) where {T,VT} = ExactHessian{T, VT}()
