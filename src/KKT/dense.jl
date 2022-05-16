
#=
    DenseKKTSystem
=#

"""
    DenseKKTSystem{T, VT, MT} <: AbstractReducedKKTSystem{T, MT}

Implement [`AbstractReducedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to be factorized (otherwise an error is returned).

"""
struct DenseKKTSystem{T, VT, MT} <: AbstractReducedKKTSystem{T, MT}
    hess::MT
    jac::MT
    pr_diag::VT
    du_diag::VT
    diag_hess::VT
    # KKT system
    aug_com::MT
    # Info
    n_ineq::Int
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    constraint_scaling::VT
    # Buffers
    etc::Dict{Symbol, Any}
end

"""
    DenseCondensedKKTSystem{T, VT, MT} <: AbstractCondensedKKTSystem{T, MT}

Implement [`AbstractCondensedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to factorize the associated KKT system (otherwise an error is returned).

"""
struct DenseCondensedKKTSystem{T, VT, MT} <: AbstractCondensedKKTSystem{T, MT}
    hess::MT
    jac::MT
    jac_ineq::MT
    pr_diag::VT
    du_diag::VT
    # KKT system
    aug_com::MT
    # Info
    n_eq::Int
    ind_eq::Vector{Int}
    ind_eq_shifted::Vector{Int}
    n_ineq::Int
    ind_ineq::Vector{Int}
    ind_ineq_shifted::Vector{Int}
    ind_fixed::Vector{Int}
    constraint_scaling::VT
    # Buffers
    etc::Dict{Symbol, Any}
end

# For templating
const AbstractDenseKKTSystem{T, VT, MT} = Union{DenseKKTSystem{T, VT, MT}, DenseCondensedKKTSystem{T, VT, MT}}

#=
    Generic functions
=#

function jtprod!(y::AbstractVector, kkt::AbstractDenseKKTSystem, x::AbstractVector)
    nx = size(kkt.hess, 1)
    ns = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns)
    # / x
    mul!(yx, kkt.jac', x)
    # / s
    ys .= -x[kkt.ind_ineq] .* kkt.constraint_scaling[kkt.ind_ineq]
    return
end

function set_jacobian_scaling!(kkt::AbstractDenseKKTSystem, constraint_scaling::AbstractVector)
    copyto!(kkt.constraint_scaling, constraint_scaling)
end

function compress_jacobian!(kkt::AbstractDenseKKTSystem{T, VT, MT}) where {T, VT, MT}
    # Scale
    kkt.jac .*= kkt.constraint_scaling
    return
end

get_raw_jacobian(kkt::AbstractDenseKKTSystem) = kkt.jac
nnz_jacobian(kkt::AbstractDenseKKTSystem) = length(kkt.jac)


#=
    DenseKKTSystem
=#

function DenseKKTSystem{T, VT, MT}(n, m, ind_ineq, ind_fixed) where {T, VT, MT}
    ns = length(ind_ineq)
    hess = MT(undef, n, n)
    jac = MT(undef, m, n)
    pr_diag = VT(undef, n+ns)
    du_diag = VT(undef, m)
    diag_hess = VT(undef, n)

    # If the the problem is unconstrained, then KKT system is directly equal
    # to the Hessian (+ some regularization terms)
    aug_com = if (m == 0)
        hess
    else
        MT(undef, n+ns+m, n+ns+m)
    end

    constraint_scaling = VT(undef, m)

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(diag_hess, zero(T))
    fill!(constraint_scaling, one(T))

    return DenseKKTSystem{T, VT, MT}(
        hess, jac, pr_diag, du_diag, diag_hess, aug_com,
        ns, ind_ineq, ind_fixed, constraint_scaling, Dict{Symbol, Any}(),
    )
end

function DenseKKTSystem{T, VT, MT}(nlp::AbstractNLPModel, info_constraints=get_index_constraints(nlp)) where {T, VT, MT}
    return DenseKKTSystem{T, VT, MT}(
        get_nvar(nlp), get_ncon(nlp), info_constraints.ind_ineq, info_constraints.ind_fixed
    )
end

is_reduced(::DenseKKTSystem) = true
num_variables(kkt::DenseKKTSystem) = length(kkt.pr_diag)

function mul!(y::AbstractVector, kkt::DenseKKTSystem, x::AbstractVector)
    mul!(y, kkt.aug_com, x)
end
function mul!(y::ReducedKKTVector, kkt::DenseKKTSystem, x::ReducedKKTVector)
    mul!(values(y), kkt.aug_com, values(x))
end

# Special getters for Jacobian
function get_jacobian(kkt::DenseKKTSystem)
    n = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    return view(kkt.jac, :, 1:n)
end

function diag_add!(dest::AbstractMatrix, d1::AbstractVector, d2::AbstractVector)
    n = length(d1)
    @inbounds for i in 1:n
        dest[i, i] = d1[i] + d2[i]
    end
end

function _build_dense_kkt_system!(dest, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq, con_scale, n, m, ns)
    # Transfer Hessian
    for i in 1:n, j in 1:i
        if i == j
            dest[i, i] = pr_diag[i] + diag_hess[i]
        else
            dest[i, j] = hess[i, j]
            dest[j, i] = hess[j, i]
        end
    end
    # Transfer slack diagonal
    for i in 1:ns
        dest[i+n, i+n] = pr_diag[i+n]
    end
    # Transfer Jacobian / variables
    for i in 1:m, j in 1:n
        dest[i + n + ns, j] = jac[i, j]
        dest[j, i + n + ns] = jac[i, j]
    end
    # Transfer Jacobian / slacks
    for j in 1:ns
        is = ind_ineq[j]
        dest[is + n + ns, j + n] = - con_scale[is]
        dest[j + n, is + n + ns] = - con_scale[is]
    end
    # Transfer dual regularization
    for i in 1:m
        dest[i + n + ns, i + n + ns] = du_diag[i]
    end
end

function build_kkt!(kkt::DenseKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    m = size(kkt.jac, 1)
    ns = length(kkt.ind_ineq)
    if m == 0 # If problem is unconstrained, just need to update the diagonal
        diag_add!(kkt.aug_com, kkt.diag_hess, kkt.pr_diag)
    else # otherwise, we update the full matrix
        _build_dense_kkt_system!(kkt.aug_com, kkt.hess, kkt.jac,
                                 kkt.pr_diag, kkt.du_diag, kkt.diag_hess,
                                 kkt.ind_ineq, kkt.constraint_scaling,
                                 n, m, ns)
    end
    treat_fixed_variable!(kkt)
end

function compress_hessian!(kkt::DenseKKTSystem)
    # Transfer diagonal term for future regularization
    diag!(kkt.diag_hess, kkt.hess)
end


#=
    DenseCondensedKKTSystem
=#

function DenseCondensedKKTSystem{T, VT, MT}(nlp::AbstractNLPModel, info_constraints=get_index_constraints(nlp)) where {T, VT, MT}
    n = get_nvar(nlp)
    m = get_ncon(nlp)
    ns = length(info_constraints.ind_ineq)
    n_eq = m - ns

    aug_com  = MT(undef, n+m-ns, n+m-ns)
    hess     = MT(undef, n, n)
    jac      = MT(undef, m, n)
    jac_ineq = MT(undef, ns, n)

    pr_diag  = VT(undef, n+ns)
    du_diag  = VT(undef, m)
    constraint_scaling = VT(undef, m)

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(constraint_scaling, one(T))

    ind_eq = setdiff(1:m, info_constraints.ind_ineq)

    # Shift indexes to avoid additional allocation in views
    ind_eq_shifted = ind_eq .+ n .+ ns
    ind_ineq_shifted = info_constraints.ind_ineq .+ n .+ ns

    return DenseCondensedKKTSystem{T, VT, MT}(
        hess, jac, jac_ineq, pr_diag, du_diag, aug_com,
        n_eq, ind_eq, ind_eq_shifted,
        ns, info_constraints.ind_ineq, ind_ineq_shifted,
        info_constraints.ind_fixed,
        constraint_scaling, Dict{Symbol, Any}(),
    )
end

is_reduced(kkt::DenseCondensedKKTSystem) = true
num_variables(kkt::DenseCondensedKKTSystem) = size(kkt.hess, 1)

function get_slack_regularization(kkt::DenseCondensedKKTSystem)
    n, ns = num_variables(kkt), kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns)
end
get_scaling_inequalities(kkt::DenseCondensedKKTSystem) = kkt.constraint_scaling[kkt.ind_ineq]

function _build_condensed_kkt_system!(
    dest::AbstractMatrix, hess::AbstractMatrix, jac::AbstractMatrix,
    pr_diag::AbstractVector, du_diag::AbstractVector, ind_eq::AbstractVector, n, m_eq,
)
    # Transfer Hessian
    @inbounds for i in 1:n, j in 1:i
        if i == j
            dest[i, i] += pr_diag[i] + hess[i, i]
        else
            dest[i, j] += hess[i, j]
            dest[j, i] += hess[j, i]
        end
    end
    # Transfer Jacobian / variables
    @inbounds for i in 1:m_eq, j in 1:n
        is = ind_eq[i]
        dest[i + n, j] = jac[is, j]
        dest[j, i + n] = jac[is, j]
    end
    # Transfer dual regularization
    @inbounds for i in 1:m_eq
        is = ind_eq[i]
        dest[i + n, i + n] = du_diag[is]
    end
end

function _build_ineq_jac!(
    dest::AbstractMatrix, jac::AbstractMatrix, pr_diag::AbstractVector,
    ind_ineq::AbstractVector, ind_fixed::AbstractVector, con_scale::AbstractVector,
    n, m_ineq,
)
    @inbounds for i in 1:m_ineq, j in 1:n
        is = ind_ineq[i]
        dest[i, j] = jac[is, j] * sqrt(pr_diag[n+i]) / con_scale[is]
    end
    # need to zero the fixed components
    dest[:, ind_fixed] .= 0.0
end

function build_kkt!(kkt::DenseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    ns = kkt.n_ineq
    n_eq = length(kkt.ind_eq)
    m = size(kkt.jac, 1)

    kkt.pr_diag[kkt.ind_fixed] .= 0
    fill!(kkt.aug_com, zero(T))
    # Build √Σₛ * J
    _build_ineq_jac!(kkt.jac_ineq, kkt.jac, kkt.pr_diag, kkt.ind_ineq, kkt.ind_fixed, kkt.constraint_scaling, n, ns)

    # Select upper-left block
    W = if n_eq > 0
        view(kkt.aug_com, 1:n, 1:n) # TODO: does not work on GPU
    else
        kkt.aug_com
    end
    # Build J' * Σₛ * J
    mul!(W, kkt.jac_ineq', kkt.jac_ineq)

    _build_condensed_kkt_system!(
        kkt.aug_com, kkt.hess, kkt.jac,
        kkt.pr_diag, kkt.du_diag,
        kkt.ind_eq, n, kkt.n_eq,
    )
    treat_fixed_variable!(kkt)
end

# TODO: check how to handle inertia with the condensed form
function is_inertia_correct(kkt::DenseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0)
end

# For inertia-free regularization
function _mul_expanded!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess, 1)
    ns = kkt.n_ineq
    m = size(kkt.jac, 1)

    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, 1+n:n+ns)
    Σd = kkt.du_diag

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, 1+n:n+ns)
    xy = view(x, 1+n+ns:n+ns+m)

    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, 1+n:n+ns)
    yy = view(y, 1+n+ns:n+ns+m)

    # / x (variable)
    yx .= Σx .* xx
    mul!(yx, kkt.hess, xx, 1.0, 1.0)
    mul!(yx, kkt.jac', xy, 1.0, 1.0)

    # / s (slack)
    ys .= Σs .* xs
    ys .-= kkt.constraint_scaling[kkt.ind_ineq] .* xy[kkt.ind_ineq]

    # / y (multiplier)
    yy .= Σd .* xy
    mul!(yy, kkt.jac, xx, 1.0, 1.0)
    yy[kkt.ind_ineq] .-= kkt.constraint_scaling[kkt.ind_ineq] .* xs
    return
end

function mul!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    # TODO: implement properly with AbstractKKTRHS
    if length(y) == length(x) == size(kkt.aug_com, 1)
        mul!(y, kkt.aug_com, x)
    else
        _mul_expanded!(y, kkt, x)
    end
end

function jprod_ineq!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_ineq, x)
end

