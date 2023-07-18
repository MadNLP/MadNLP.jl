
"""
    DenseKKTSystem{T, VT, MT, QN, VI} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement [`AbstractReducedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to be factorized (otherwise an error is returned).

"""
mutable struct DenseKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int},
    } <: AbstractReducedKKTSystem{T, VT, MT, QN}
    
    hess::MT
    jac::MT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    diag_hess::VT
    # KKT system
    aug_com::MT
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    # Regularization
    del_w::T
    del_w_last::T
    del_c::T
    # Linear Solver
    linear_solver::LS
    # Buffers
    etc::Dict{Symbol, Any}
end

function create_kkt_system(
    ::Type{DenseKKTSystem},
    cb::AbstractCallback{T,VT}, opt, 
    opt_linear_solver, cnt, ind_cons) where {T,VT}
    
    ind_ineq = ind_cons.ind_ineq
    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub
    
    n = cb.nvar
    m = cb.ncon
    ns = length(ind_ineq)
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    
    hess = create_array(cb, n, n)
    jac = create_array(cb, m, n)
    aug_com = create_array(cb, n+ns+m, n+ns+m)
    reg = create_array(cb, n+ns)
    pr_diag = create_array(cb, n+ns)
    du_diag = create_array(cb, m)
    diag_hess = create_array(cb, n)
    
    l_diag = fill!(VT(undef, nlb), one(T))
    u_diag = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(diag_hess, zero(T))

    quasi_newton = create_quasi_newton(opt.hessian_approximation, cb, n)
    cnt.linear_solver_time += @elapsed linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)
    
    del_w = one(T)
    del_w_last = zero(T)
    del_c = zero(T)

    return DenseKKTSystem(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        diag_hess, aug_com,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        del_w, del_w_last, del_c,
        linear_solver,
        Dict{Symbol, Any}(), 
    )
end

"""
    DenseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement [`AbstractCondensedKKTSystem`](@ref) with dense matrices.

Requires a dense linear solver to factorize the associated KKT system (otherwise an error is returned).

"""
mutable struct DenseCondensedKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int}
    } <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    
    hess::MT
    jac::MT
    quasi_newton::QN
    jac_ineq::MT

    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    pd_buffer::VT
    diag_buffer::VT
    buffer::VT
    # KKT system
    aug_com::MT
    # Info
    n_eq::Int
    ind_eq::VI
    ind_eq_shifted::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    ind_ineq_shifted::VI
    # Regularization
    del_w::T
    del_w_last::T
    del_c::T
    # Linear Solver
    linear_solver::LS
    # Buffers
    etc::Dict{Symbol, Any}
end

function create_kkt_system(
    ::Type{DenseCondensedKKTSystem},
    cb::AbstractCallback{T,VT}, opt, 
    opt_linear_solver, cnt, ind_cons) where {T,VT}
    
    n = cb.nvar
    m = cb.ncon
    ns = length(ind_cons.ind_ineq)
    n_eq = m - ns
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    aug_com  = create_array(cb, n+m-ns, n+m-ns)
    hess     = create_array(cb, n, n)
    jac      = create_array(cb, m, n)
    jac_ineq = create_array(cb, ns, n)

    reg  = VT(undef, n+ns)
    pr_diag  = VT(undef, n+ns)
    du_diag  = VT(undef, m)
    l_diag = fill!(VT(undef, nlb), one(T))
    u_diag = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))
    
    pd_buffer = VT(undef, n + n_eq)
    diag_buffer = VT(undef, ns)
    buffer = VT(undef, m)
    
    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # Shift indexes to avoid additional allocation in views
    ind_eq_shifted = ind_cons.ind_eq .+ n .+ ns
    ind_ineq_shifted = ind_cons.ind_ineq .+ n .+ ns

    quasi_newton = create_quasi_newton(opt.hessian_approximation, cb, n)
    cnt.linear_solver_time += @elapsed linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)
    
    del_w = one(T)
    del_w_last = zero(T)
    del_c = zero(T)

    return DenseCondensedKKTSystem(
        hess, jac, quasi_newton, jac_ineq,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        pd_buffer, diag_buffer, buffer,
        aug_com, 
        n_eq, ind_cons.ind_eq, ind_eq_shifted,
        ns,
        ind_cons.ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        ind_ineq_shifted,
        del_w, del_w_last, del_c,
        linear_solver,
        Dict{Symbol, Any}(),
    )
end

# For templating
const AbstractDenseKKTSystem{T, VT, MT, QN} = Union{
    DenseKKTSystem{T, VT, MT, QN},
    DenseCondensedKKTSystem{T, VT, MT, QN},
}

#=
    Generic functions
=#

@views function jtprod!(y::AbstractVector, kkt::AbstractDenseKKTSystem, x::AbstractVector)
    nx = size(kkt.hess, 1)
    ind_ineq = kkt.ind_ineq
    ns = length(ind_ineq)
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns)
    # / x
    mul!(yx, kkt.jac', x)
    # / s
    ys .= -x[ind_ineq]
    return
end

function compress_jacobian!(kkt::AbstractDenseKKTSystem)
    return
end

get_raw_jacobian(kkt::AbstractDenseKKTSystem) = kkt.jac
nnz_jacobian(kkt::AbstractDenseKKTSystem) = length(kkt.jac)

#=
    DenseKKTSystem
=#

is_reduced(::DenseKKTSystem) = true
num_variables(kkt::DenseKKTSystem) = length(kkt.pr_diag)

function mul!(y::AbstractVector, kkt::DenseKKTSystem, x::AbstractVector)
    symul!(y, kkt.aug_com, x)
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

function _build_dense_kkt_system!(dest::VT, hess, jac, pr_diag, du_diag, diag_hess, ind_ineq, n, m, ns) where {T, VT <: AbstractMatrix{T}}
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
        dest[is + n + ns, j + n] = - one(T)
        dest[j + n, is + n + ns] = - one(T)
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
    
    kkt.pr_diag .= kkt.reg
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag

    _build_dense_kkt_system!(kkt.aug_com, kkt.hess, kkt.jac,
                                kkt.pr_diag, kkt.du_diag, kkt.diag_hess,
                                kkt.ind_ineq, 
                                n, m, ns)
end

function compress_hessian!(kkt::DenseKKTSystem)
    # Transfer diagonal term for future regularization
    diag!(kkt.diag_hess, kkt.hess)
end

#=
    DenseCondensedKKTSystem
=#

is_reduced(kkt::DenseCondensedKKTSystem) = true
num_variables(kkt::DenseCondensedKKTSystem) = size(kkt.hess, 1)

function get_slack_regularization(kkt::DenseCondensedKKTSystem)
    n, ns = num_variables(kkt), kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns)
end

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
    dest::AbstractMatrix, jac::AbstractMatrix, diag_buffer::AbstractVector,
    ind_ineq::AbstractVector, 
    n, m_ineq,
)
    @inbounds for i in 1:m_ineq, j in 1:n
        is = ind_ineq[i]
        dest[i, j] = jac[is, j] * sqrt(diag_buffer[i]) 
    end
end

function build_kkt!(kkt::DenseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT}
    n = size(kkt.hess, 1)
    ns = kkt.n_ineq
    n_eq = length(kkt.ind_eq)
    m = size(kkt.jac, 1)

    kkt.pr_diag .= kkt.reg
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag

    fill!(kkt.aug_com, zero(T))

    # Build √Σₛ * J
    Σs = view(kkt.pr_diag, n+1:n+ns)
    Σd = @view(kkt.du_diag[kkt.ind_ineq])
    kkt.diag_buffer .= Σs ./ ( 1 .- Σd .* Σs)
    _build_ineq_jac!(kkt.jac_ineq, kkt.jac, kkt.diag_buffer, kkt.ind_ineq, n, ns)

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
end

# TODO: check how to handle inertia with the condensed form
function is_inertia_correct(kkt::DenseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0 && num_neg == kkt.n_eq)
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
    symul!(yx, kkt.hess, xx)
    mul!(yx, kkt.jac', xy, 1.0, 1.0)

    # / s (slack)
    ys .= Σs .* xs
    ys .-= xy[kkt.ind_ineq]

    # / y (multiplier)
    yy .= Σd .* xy
    mul!(yy, kkt.jac, xx, 1.0, 1.0)
    yy[kkt.ind_ineq] .-= xs
    return
end

function mul!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    # TODO: implement properly with AbstractKKTRHS
    if length(y) == length(x) == size(kkt.aug_com, 1)
        symul!(y, kkt.aug_com, x)
    else
        _mul_expanded!(y, kkt, x)
    end
end

function jprod_ineq!(y::AbstractVector, kkt::DenseCondensedKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_ineq, x)
end

