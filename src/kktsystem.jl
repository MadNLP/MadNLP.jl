
abstract type AbstractKKTSystem{T, MT<:AbstractMatrix{T}} end

"Sparse KKT system"
abstract type AbstractSparseKKTSystem{T, MT} <: AbstractKKTSystem{T, MT} end

#=
    Templates
=#
"Initialize KKT system with default values."
function initialize! end

"Assemble KKT matrix."
function build_kkt! end

"Compress Hessian matrix."
function compress_hessian! end

"Compress Jacobian matrix."
function compress_jacobian! end

"Get KKT system"
function get_kkt end

"Get Jacobian matrix"
function get_jacobian end

"Get Hessian matrix"
function get_hessian end

"Multiply with Jacobian"
function jtprod! end

"Regularize values in the diagonal of the KKT system."
function regularize_diagonal! end

"Set scaling of Jacobian"
function set_jacobian_scaling! end

"Return true if KKT system is reduced."
function is_reduced end

"Nonzero in Jacobian"
function nnz_jacobian end

"Nonzero in Hessian"
function nnz_kkt end

"Dense Jacobian callback"
function jac_dense! end

"Dense Hessian callback"
function hess_dense! end

"Number of variables associated to the KKT system."
function n_variables end

"Check if inertia is correct."
function is_inertia_correct end

#=
    Generic functions
=#
function initialize!(kkt::AbstractKKTSystem)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
end

function regularize_diagonal!(kkt::AbstractKKTSystem, primal, dual)
    kkt.pr_diag .+= primal
    kkt.du_diag .= .-dual
end

Base.size(kkt::AbstractKKTSystem) = size(kkt.aug_com)
Base.size(kkt::AbstractKKTSystem, dim::Int) = size(kkt.aug_com, dim)

# Getters
get_kkt(kkt::AbstractKKTSystem) = kkt.aug_com
get_jacobian(kkt::AbstractKKTSystem) = kkt.jac
get_hessian(kkt::AbstractKKTSystem) = kkt.hess
get_raw_jacobian(kkt::AbstractKKTSystem) = kkt.jac_raw


# Fix variable treatment
function treat_fixed_variable!(kkt::AbstractKKTSystem{T, MT}) where {T, MT<:SparseMatrixCSC{T, Int32}}
    length(kkt.ind_fixed) == 0 && return
    aug = kkt.aug_com

    fixed_aug_diag = view(aug.nzval, aug.colptr[kkt.ind_fixed])
    fixed_aug_diag .= 1.0
    fixed_aug = view(aug.nzval, kkt.ind_aug_fixed)
    fixed_aug .= 0.0
    return
end
function treat_fixed_variable!(kkt::AbstractKKTSystem{T, MT}) where {T, MT<:Matrix{T}}
    length(kkt.ind_fixed) == 0 && return
    aug = kkt.aug_com
    @inbounds for i in kkt.ind_fixed
        aug[i, :] .= 0.0
        aug[:, i] .= 0.0
        aug[i, i]  = 1.0
    end
end

function is_inertia_correct(kkt::AbstractKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == n_variables(kkt))
end

#=
    SparseKKTSystem
=#

# Generic function for sparse KKT systems
function build_kkt!(kkt::AbstractSparseKKTSystem{T, MT}) where {T, MT<:Matrix{T}}
    copyto!(kkt.aug_com, kkt.aug_raw)
    treat_fixed_variable!(kkt)
end

function build_kkt!(kkt::AbstractSparseKKTSystem{T, MT}) where {T, MT<:SparseMatrixCSC{T, Int32}}
    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
    treat_fixed_variable!(kkt)
end

function set_jacobian_scaling!(kkt::AbstractSparseKKTSystem{T, MT}, constraint_scaling::AbstractVector) where {T, MT}
    nnzJ = length(kkt.jac)::Int
    @inbounds for i in 1:nnzJ
        index = kkt.jac_raw.I[i]
        kkt.jacobian_scaling[i] = constraint_scaling[index]
    end
end

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T, MT}) where {T, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    kkt.jac .*= kkt.jacobian_scaling # scaling
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T, MT}) where {T, MT<:Matrix{T}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    kkt.jac .*= kkt.jacobian_scaling # scaling
    copyto!(kkt.jac_com, kkt.jac_raw)
end

function compress_hessian!(kkt::AbstractSparseKKTSystem)
    nothing
end

function mul!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, Symmetric(kkt.aug_com, :L), x)
end

function jtprod!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

nnz_jacobian(kkt::AbstractSparseKKTSystem) = nnz(kkt.jac_raw)
nnz_kkt(kkt::AbstractSparseKKTSystem) = nnz(kkt.aug_raw)

#=
    SparseKKTSystem
=#

struct SparseKKTSystem{T, MT} <: AbstractSparseKKTSystem{T, MT}
    hess::StrideOneVector{T}
    jac::StrideOneVector{T}
    pr_diag::StrideOneVector{T}
    du_diag::StrideOneVector{T}
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32}
    aug_com::MT
    aug_csc_map::Union{Nothing, Vector{Int}}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32}
    jac_com::MT
    jac_csc_map::Union{Nothing, Vector{Int}}
    # Info
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_aug_fixed::Vector{Int}
    jacobian_scaling::Vector{T}
end

function SparseKKTSystem{T, MT}(
    n::Int, m::Int, ind_ineq::Vector{Int}, ind_fixed::Vector{Int},
    hess_sparsity_I, hess_sparsity_J,
    jac_sparsity_I, jac_sparsity_J,
) where {T, MT}
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)

    aug_vec_length = n+m
    aug_mat_length = n+m+n_hess+n_jac

    I = Vector{Int32}(undef, aug_mat_length)
    J = Vector{Int32}(undef, aug_mat_length)
    V = Vector{T}(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n+n_jac+n_hess+m

    I[1:n] .= 1:n
    I[n+1:n+n_hess] = hess_sparsity_I
    I[n+n_hess+1:n+n_hess+n_jac] .= (jac_sparsity_I.+n)
    I[n+n_hess+n_jac+1:offset] .= (n+1:n+m)

    J[1:n] .= 1:n
    J[n+1:n+n_hess] = hess_sparsity_J
    J[n+n_hess+1:n+n_hess+n_jac] .= jac_sparsity_J
    J[n+n_hess+n_jac+1:offset] .= (n+1:n+m)

    pr_diag = view(V, 1:n)
    du_diag = view(V, n_jac+n_hess+n+1:n_jac+n_hess+n+m)

    hess = view(V, n+1:n+n_hess)
    jac = view(V, n_hess+n+1:n_hess+n+n_jac)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(m,n,jac_sparsity_I,jac_sparsity_J,jac)

    aug_com = MT(aug_raw)
    jac_com = MT(jac_raw)

    aug_csc_map = get_mapping(aug_com, aug_raw)
    jac_csc_map = get_mapping(jac_com, jac_raw)

    ind_aug_fixed = if isa(aug_com, SparseMatrixCSC)
        _get_fixed_variable_index(aug_com, ind_fixed)
    else
        zeros(Int, 0)
    end
    jac_scaling = ones(T, n_jac)

    return SparseKKTSystem{T, MT}(
        hess, jac, pr_diag, du_diag,
        aug_raw, aug_com, aug_csc_map,
        jac_raw, jac_com, jac_csc_map,
        ind_ineq, ind_fixed, ind_aug_fixed, jac_scaling,
    )
end

# Build KKT system directly from AbstractNLPModel
function SparseKKTSystem{T, MT}(nlp::AbstractNLPModel, ind_cons=get_index_constraints(nlp)) where {T, MT}
    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.
    n = get_nvar(nlp) + n_slack
    m = get_ncon(nlp)
    # Evaluate sparsity pattern
    jac_I = Vector{Int32}(undef, get_nnzj(nlp))
    jac_J = Vector{Int32}(undef, get_nnzj(nlp))
    jac_structure!(nlp,jac_I, jac_J)

    hess_I = Vector{Int32}(undef, get_nnzh(nlp))
    hess_J = Vector{Int32}(undef, get_nnzh(nlp))
    hess_structure!(nlp,hess_I,hess_J)

    force_lower_triangular!(hess_I,hess_J)
    # Incorporate slack's sparsity pattern
    append!(jac_I, ind_cons.ind_ineq)
    append!(jac_J, get_nvar(nlp)+1:get_nvar(nlp)+n_slack)

    return SparseKKTSystem{T, MT}(
        n, m, ind_cons.ind_ineq, ind_cons.ind_fixed,
        hess_I, hess_J, jac_I, jac_J,
    )
end

is_reduced(::SparseKKTSystem) = true
n_variables(kkt::SparseKKTSystem) = length(kkt.pr_diag)

#=
    SparseUnreducedKKTSystem
=#

struct SparseUnreducedKKTSystem{T, MT} <: AbstractSparseKKTSystem{T, MT}
    hess::StrideOneVector{T}
    jac::StrideOneVector{T}
    pr_diag::StrideOneVector{T}
    du_diag::StrideOneVector{T}

    l_diag::StrideOneVector{T}
    u_diag::StrideOneVector{T}
    l_lower::StrideOneVector{T}
    u_lower::StrideOneVector{T}

    aug_raw::SparseMatrixCOO{T,Int32}
    aug_com::MT
    aug_csc_map::Union{Nothing, Vector{Int}}

    jac_raw::SparseMatrixCOO{T,Int32}
    jac_com::MT
    jac_csc_map::Union{Nothing, Vector{Int}}
    ind_ineq::Vector{Int}
    ind_fixed::Vector{Int}
    ind_aug_fixed::Vector{Int}
    jacobian_scaling::Vector{T}
end

function SparseUnreducedKKTSystem{T, MT}(
    n::Int, m::Int, nlb::Int, nub::Int, ind_ineq, ind_fixed,
    hess_sparsity_I, hess_sparsity_J,
    jac_sparsity_I, jac_sparsity_J,
    ind_lb, ind_ub,
) where {T, MT}
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)

    aug_mat_length = n + m + n_hess + n_jac + 2*nlb + 2*nub
    aug_vec_length = n+m+nlb+nub

    I = Vector{Int32}(undef, aug_mat_length)
    J = Vector{Int32}(undef, aug_mat_length)
    V = zeros(aug_mat_length)

    offset = n+n_jac+n_hess+m

    I[1:n] .= 1:n
    I[n+1:n+n_hess] = hess_sparsity_I
    I[n+n_hess+1:n+n_hess+n_jac].=(jac_sparsity_I.+n)
    I[n+n_hess+n_jac+1:offset].=(n+1:n+m)

    J[1:n] .= 1:n
    J[n+1:n+n_hess] = hess_sparsity_J
    J[n+n_hess+1:n+n_hess+n_jac].=jac_sparsity_J
    J[n+n_hess+n_jac+1:offset].=(n+1:n+m)

    I[offset+1:offset+nlb] .= (1:nlb).+(n+m)
    I[offset+nlb+1:offset+2nlb] .= (1:nlb).+(n+m)
    I[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n+m+nlb)
    I[offset+2nlb+nub+1:offset+2nlb+2nub] .= (1:nub).+(n+m+nlb)
    J[offset+1:offset+nlb] .= (1:nlb).+(n+m)
    J[offset+nlb+1:offset+2nlb] .= ind_lb
    J[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n+m+nlb)
    J[offset+2nlb+nub+1:offset+2nlb+2nub] .= ind_ub

    pr_diag = view(V,1:n)
    du_diag = view(V,n_jac+n_hess+n+1:n_jac+n_hess+n+m)

    l_diag = view(V,offset+1:offset+nlb)
    l_lower= view(V,offset+nlb+1:offset+2nlb)
    u_diag = view(V,offset+2nlb+1:offset+2nlb+nub)
    u_lower= view(V,offset+2nlb+nub+1:offset+2nlb+2nub)

    hess = view(V, n+1:n+n_hess)
    jac = view(V, n_hess+n+1:n_hess+n+n_jac)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(m,n,jac_sparsity_I,jac_sparsity_J,jac)

    aug_com = MT(aug_raw)
    jac_com = MT(jac_raw)

    aug_csc_map = get_mapping(aug_com, aug_raw)
    jac_csc_map = get_mapping(jac_com, jac_raw)

    jac_scaling = ones(T, n_jac)

    ind_aug_fixed = if isa(aug_com, SparseMatrixCSC)
        _get_fixed_variable_index(aug_com, ind_fixed)
    else
        zeros(Int, 0)
    end

    return SparseUnreducedKKTSystem{T, MT}(
        hess, jac, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        aug_raw, aug_com, aug_csc_map,
        jac_raw, jac_com, jac_csc_map,
        ind_ineq, ind_fixed, ind_aug_fixed, jac_scaling,
    )
end

function SparseUnreducedKKTSystem{T, MT}(nlp::AbstractNLPModel, ind_cons=get_index_constraints(nlp)) where {T, MT}
    n_slack = length(ind_cons.ind_ineq)
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    # Deduce KKT size.
    n = get_nvar(nlp) + n_slack
    m = get_ncon(nlp)
    # Evaluate sparsity pattern
    jac_I = Vector{Int32}(undef, get_nnzj(nlp))
    jac_J = Vector{Int32}(undef, get_nnzj(nlp))
    jac_structure!(nlp,jac_I, jac_J)

    hess_I = Vector{Int32}(undef, get_nnzh(nlp))
    hess_J = Vector{Int32}(undef, get_nnzh(nlp))
    hess_structure!(nlp,hess_I,hess_J)

    force_lower_triangular!(hess_I,hess_J)
    # Incorporate slack's sparsity pattern
    append!(jac_I, ind_cons.ind_ineq)
    append!(jac_J, get_nvar(nlp)+1:get_nvar(nlp)+n_slack)

    return SparseUnreducedKKTSystem{T, MT}(
        n, m, nlb, nub, ind_cons.ind_ineq, ind_cons.ind_fixed,
        hess_I, hess_J, jac_I, jac_J, ind_cons.ind_lb, ind_cons.ind_ub,
    )
end

function initialize!(kkt::SparseUnreducedKKTSystem)
    kkt.pr_diag.=1
    kkt.du_diag.=0
    kkt.hess.=0
    kkt.l_lower.=0
    kkt.u_lower.=0
    kkt.l_diag.=1
    kkt.u_diag.=1
end

is_reduced(::SparseUnreducedKKTSystem) = false
n_variables(kkt::SparseUnreducedKKTSystem) = length(kkt.pr_diag)

#=
    DenseKKTSystem
=#

struct DenseKKTSystem{T, VT, MT} <: AbstractKKTSystem{T, MT}
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
n_variables(kkt::DenseKKTSystem) = length(kkt.pr_diag)

# Special getters for Jacobian
function get_jacobian(kkt::DenseKKTSystem)
    n = size(kkt.hess, 1)
    ns = length(kkt.ind_ineq)
    return view(kkt.jac, :, 1:n)
end
get_raw_jacobian(kkt::DenseKKTSystem) = kkt.jac

nnz_jacobian(kkt::DenseKKTSystem) = length(kkt.jac)
nnz_kkt(kkt::DenseKKTSystem) = length(kkt.aug_com)

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

