
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
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    jacobian_scaling::VT
    # Buffers
    etc::Dict{Symbol, Any}
end

function DenseKKTSystem{T, VT, MT}(n, m, ind_ineq, ind_fixed) where {T, VT, MT}
    ns = length(ind_ineq)
    hess = MT(undef, n, n)
    jac = MT(undef, m, n+ns)
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

    jacobian_scaling = VT(undef, m)

    # Init!
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))
    fill!(diag_hess, zero(T))
    fill!(jacobian_scaling, one(T))

    return DenseKKTSystem{T, VT, MT}(
        hess, jac, pr_diag, du_diag, diag_hess, aug_com,
        ind_ineq, ind_fixed, jacobian_scaling, Dict{Symbol, Any}(),
    )
end

function DenseKKTSystem{T, VT, MT}(nlp::AbstractNLPModel, info_constraints=get_index_constraints(nlp)) where {T, VT, MT}
    return DenseKKTSystem{T, VT, MT}(
        get_nvar(nlp), get_ncon(nlp), info_constraints.ind_ineq, info_constraints.ind_fixed
    )
end

is_reduced(::DenseKKTSystem) = true
num_variables(kkt::DenseKKTSystem) = length(kkt.pr_diag)

# Special getters for Jacobian
function get_jacobian(kkt::DenseKKTSystem)
    n = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    return view(kkt.jac, :, 1:n)
end

get_raw_jacobian(kkt::DenseKKTSystem) = kkt.jac

nnz_jacobian(kkt::DenseKKTSystem) = length(kkt.jac)

function diag_add!(dest::AbstractMatrix, d1::AbstractVector, d2::AbstractVector)
    n = length(d1)
    @inbounds for i in 1:n
        dest[i, i] = d1[i] + d2[i]
    end
end

function _build_dense_kkt_system!(dest, hess, jac, pr_diag, du_diag, diag_hess, n, m, ns)
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
    # Transfer Jacobian
    for i in 1:m, j in 1:(n+ns)
        dest[i + n + ns, j] = jac[i, j]
        dest[j, i + n + ns] = jac[i, j]
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
        _build_dense_kkt_system!(kkt.aug_com, kkt.hess, kkt.jac, kkt.pr_diag, kkt.du_diag, kkt.diag_hess, n, m, ns)
    end
    treat_fixed_variable!(kkt)
end

function compress_jacobian!(kkt::DenseKKTSystem{T, VT, MT}) where {T, VT, MT}
    m = size(kkt.jac, 1)
    n = size(kkt.hess, 1)
    # Add slack indexes
    for i in kkt.ind_ineq
        kkt.jac[i, i+n] = -one(T)
    end
    # Scale
    kkt.jac .*= kkt.jacobian_scaling
    return
end

function compress_hessian!(kkt::DenseKKTSystem)
    # Transfer diagonal term for future regularization
    diag!(kkt.diag_hess, kkt.hess)
end

function mul!(y::AbstractVector, kkt::DenseKKTSystem, x::AbstractVector)
    mul!(y, kkt.aug_com, x)
end

function jtprod!(y::AbstractVector, kkt::DenseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac', x)
end

function set_jacobian_scaling!(kkt::DenseKKTSystem, constraint_scaling::AbstractVector)
    copyto!(kkt.jacobian_scaling, constraint_scaling)
end

