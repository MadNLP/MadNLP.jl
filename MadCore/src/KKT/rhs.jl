"""
    AbstractKKTVector{T, VT}

Supertype for KKT's right-hand-side vectors ``(x, s, y, z, ν, w)``.
"""
abstract type AbstractKKTVector{T, VT} end

"""
    full(X::AbstractKKTVector)

Return the all the values stored inside the KKT vector `X`.
"""
function full end

Base.length(rhs::AbstractKKTVector) = length(full(rhs))

"""
    number_primal(X::AbstractKKTVector)

Get total number of primal values ``(x, s)`` in KKT vector `X`.
"""
number_primal(rhs::AbstractKKTVector) = length(primal(rhs))

"""
    number_dual(X::AbstractKKTVector)

Get total number of dual values ``(y, z)`` in KKT vector `X`.
"""
number_dual(rhs::AbstractKKTVector) = length(dual(rhs))

"""
    primal(X::AbstractKKTVector)

Return the primal values ``(x, s)`` stored in the KKT vector `X`.
"""
function primal end

"""
    dual(X::AbstractKKTVector)

Return the dual values ``(y, z)`` stored in the KKT vector `X`.
"""
function dual end

"""
    primal_dual(X::AbstractKKTVector)

Return both the primal and the dual values ``(x, s, y, z)`` stored in the KKT vector `X`.
"""
function primal_dual end

"""
    dual_lb(X::AbstractKKTVector)

Return the dual values ``ν`` associated to the lower-bound
stored in the KKT vector `X`.
"""
function dual_lb end

"""
    dual_ub(X::AbstractKKTVector)

Return the dual values ``w`` associated to the upper-bound
stored in the KKT vector `X`.
"""
function dual_ub end

function Base.fill!(rhs::AbstractKKTVector{T}, val::T) where {T}
    return fill!(full(rhs), val)
end

# Overload basic BLAS operations.
norm(X::AbstractKKTVector, p::Real = 2.0) = norm(full(X), p)
dot(X::AbstractKKTVector, Y::AbstractKKTVector) = dot(full(X), full(Y))
function axpy!(a::Number, X::AbstractKKTVector, Y::AbstractKKTVector)
    return axpy!(a, full(X), full(Y))
end

#=
    UnreducedKKTVector
=#

"""
    UnreducedKKTVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Full KKT vector ``(x, s, y, z, ν, w)``, associated to a [`AbstractUnreducedKKTSystem`](@ref).

"""
struct UnreducedKKTVector{T, VT <: AbstractVector{T}, VI} <: AbstractKKTVector{T, VT}
    values::VT
    x::VT  # unsafe view
    xp::VT # unsafe view
    xp_lr::SubVector{T, VT, VI}
    xp_ur::SubVector{T, VT, VI}
    xl::VT # unsafe view
    xzl::VT # unsafe view
    xzu::VT # unsafe view
end

function UnreducedKKTVector(
        ::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, ind_lb, ind_ub
    ) where {T, VT <: AbstractVector{T}}
    values = VT(undef, n + m + nlb + nub)
    fill!(values, zero(T))
    # Wrap directly array x to avoid dealing with views
    x = _madnlp_unsafe_wrap(values, n + m) # Primal-Dual
    xp = _madnlp_unsafe_wrap(values, n) # Primal
    xl = _madnlp_unsafe_wrap(values, m, n + 1) # Dual
    xzl = _madnlp_unsafe_wrap(values, nlb, n + m + 1) # Lower bound
    xzu = _madnlp_unsafe_wrap(values, nub, n + m + nlb + 1) # Upper bound

    xp_lr = view(xp, ind_lb)
    xp_ur = view(xp, ind_ub)

    return UnreducedKKTVector(values, x, xp, xp_lr, xp_ur, xl, xzl, xzu)
end

function UnreducedKKTVector(kkt::AbstractKKTSystem{T, VT}) where {T, VT}
    return UnreducedKKTVector(
        VT,
        length(kkt.pr_diag),
        length(kkt.du_diag),
        length(kkt.l_diag),
        length(kkt.u_diag),
        kkt.ind_lb,
        kkt.ind_ub,
    )
end

function Base.copy(rhs::UnreducedKKTVector{T, VT}) where {T, VT}
    new_rhs = UnreducedKKTVector(
        VT,
        length(rhs.xp),
        length(rhs.xl),
        length(rhs.xzl),
        length(rhs.xzu),
        rhs.xp_lr.indices[1],
        rhs.xp_ur.indices[1],
    )
    copyto!(full(new_rhs), full(rhs))
    return new_rhs
end

full(rhs::UnreducedKKTVector) = rhs.values
primal(rhs::UnreducedKKTVector) = rhs.xp
dual(rhs::UnreducedKKTVector) = rhs.xl
primal_dual(rhs::UnreducedKKTVector) = rhs.x
dual_lb(rhs::UnreducedKKTVector) = rhs.xzl
dual_ub(rhs::UnreducedKKTVector) = rhs.xzu


"""
    PrimalVector{T, VT<:AbstractVector{T}} <: AbstractKKTVector{T, VT}

Primal vector ``(x, s)``.

"""
struct PrimalVector{T, VT <: AbstractVector{T}, VI} <: AbstractKKTVector{T, VT}
    values::VT
    values_lr::SubVector{T, VT, VI}
    values_ur::SubVector{T, VT, VI}
    x::VT  # unsafe view
    s::VT # unsafe view
end

function PrimalVector(::Type{VT}, nx::Int, ns::Int, ind_lb, ind_ub) where {T, VT <: AbstractVector{T}}
    values = VT(undef, nx + ns)
    fill!(values, zero(T))
    x = _madnlp_unsafe_wrap(values, nx)
    s = _madnlp_unsafe_wrap(values, ns, nx + 1)
    values_lr = view(values, ind_lb)
    values_ur = view(values, ind_ub)

    return PrimalVector(
        values, values_lr, values_ur, x, s,
    )
end

full(rhs::PrimalVector) = rhs.values
primal(rhs::PrimalVector) = rhs.values
variable(rhs::PrimalVector) = rhs.x
slack(rhs::PrimalVector) = rhs.s

#=
    Core KKT-vector solve kernels. Solver-agnostic (operate on AbstractKKTVector /
    AbstractKKTSystem, both owned by MadCore) and used by every KKT system's solve
    path, so they live in MadCore. MadNLP's IPM reaches them via `@reexport`.
=#
@inbounds function _kktmul!(
        w::AbstractKKTVector,
        x::AbstractKKTVector,
        reg,
        du_diag,
        l_lower,
        u_lower,
        l_diag,
        u_diag,
        alpha,
        beta,
    )
    primal(w) .+= alpha .* reg .* primal(x)
    dual(w) .+= alpha .* du_diag .* dual(x)
    w.xp_lr .-= alpha .* dual_lb(x)
    w.xp_ur .+= alpha .* dual_ub(x)
    dual_lb(w) .= beta .* dual_lb(w) .+ alpha .* (x.xp_lr .* l_lower .- dual_lb(x) .* l_diag)
    dual_ub(w) .= beta .* dual_ub(w) .+ alpha .* (x.xp_ur .* u_lower .+ dual_ub(x) .* u_diag)
    return
end

@inbounds function reduce_rhs!(
        xp_lr, wl, l_diag,
        xp_ur, wu, u_diag,
    )
    xp_lr .-= wl ./ l_diag
    xp_ur .-= wu ./ u_diag
    return
end
function reduce_rhs!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    return reduce_rhs!(
        d.xp_lr, dual_lb(d), kkt.l_diag,
        d.xp_ur, dual_ub(d), kkt.u_diag,
    )
end

function finish_aug_solve!(kkt::AbstractKKTSystem, d::AbstractKKTVector)
    dlb = dual_lb(d)
    dub = dual_ub(d)
    dlb .= (.-dlb .+ kkt.l_lower .* d.xp_lr) ./ kkt.l_diag
    dub .= (dub .- kkt.u_lower .* d.xp_ur) ./ kkt.u_diag
    return
end
