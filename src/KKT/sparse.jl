"""
    SparseKKTSystem{T, VT, MT, QN} <: AbstractReducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractReducedKKTSystem`](@ref) in sparse COO format.

"""
mutable struct SparseKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}
    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}
    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end


"""
    SparseUnreducedKKTSystem{T, VT, MT, QN} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractUnreducedKKTSystem`](@ref) in sparse COO format.

"""
mutable struct SparseUnreducedKKTSystem{T, VT, MT, QN, LS, VI, VI32} <: AbstractUnreducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac_callback::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    l_lower_aug::VT
    u_lower_aug::VT
    
    # Augmented system
    aug_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    aug_com::MT
    aug_csc_map::Union{Nothing, VI}

    # Hessian
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    jac_com::MT
    jac_csc_map::Union{Nothing, VI}
    
    # LinearSolver
    linear_solver::LS
    
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

"""
    SparseCondensedKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

Implement the [`AbstractCondensedKKTSystem`](@ref) in sparse COO format.

"""
mutable struct SparseCondensedKKTSystem{T, VT, MT, QN, LS, VI, VI32, VTu1, VTu2, EXT} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT
    hess_raw::SparseMatrixCOO{T,Int32,VT, VI32}
    hess_com::MT
    hess_csc_map::Union{Nothing, VI}

    # Jacobian
    jac::VT
    jt_coo::SparseMatrixCOO{T,Int32,VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}
    
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # buffer
    buffer::VT
    buffer2::VT

    # Augmented system
    aug_com::MT
    
    # slack diagonal buffer
    diag_buffer::VT
    dptr::VTu1
    hptr::VTu1
    jptr::VTu2
    
    # LinearSolver
    linear_solver::LS

    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # extra
    ext::EXT
end
 
# Template to dispatch on sparse representation
const AbstractSparseKKTSystem{T, VT, MT, QN} = Union{
    SparseKKTSystem{T, VT, MT, QN},
    SparseCondensedKKTSystem{T, VT, MT, QN},
    SparseUnreducedKKTSystem{T, VT, MT, QN},
}

#=
    Generic sparse methods
=#
function build_hessian_structure(nlp::SparseCallback, ::Type{<:ExactHessian})
    hess_I = create_array(nlp, Int32, nlp.nnzh)
    hess_J = create_array(nlp, Int32, nlp.nnzh)
    _hess_sparsity_wrapper!(nlp,hess_I,hess_J)
    return hess_I, hess_J
end
# NB. Quasi-Newton methods require only the sparsity pattern
#     of the diagonal term to store the term ξ I.
function build_hessian_structure(nlp::SparseCallback, ::Type{<:AbstractQuasiNewton})
    hess_I = collect(Int32, 1:nlp.nvar)
    hess_J = collect(Int32, 1:nlp.nvar)
    return hess_I, hess_J
end

function mul!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, Symmetric(kkt.aug_com, :L), x)
end
function mul!(y::AbstractKKTVector, kkt::AbstractSparseKKTSystem, x::AbstractKKTVector)
    mul!(full(y), Symmetric(kkt.aug_com, :L), full(x))
end

function jtprod!(y::AbstractVector, kkt::AbstractSparseKKTSystem, x::AbstractVector)
    mul!(y, kkt.jac_com', x)
end

get_jacobian(kkt::AbstractSparseKKTSystem) = kkt.jac_callback

nnz_jacobian(kkt::AbstractSparseKKTSystem) = nnz(kkt.jac_raw)

function compress_jacobian!(kkt::AbstractSparseKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    transfer!(kkt.jac_com, kkt.jac_raw, kkt.jac_csc_map)
end

function compress_jacobian!(kkt::AbstractSparseKKTSystem{T, VT, MT}) where {T, VT, MT<:Matrix{T}}
    ns = length(kkt.ind_ineq)
    kkt.jac[end-ns+1:end] .= -1.0
    copyto!(kkt.jac_com, kkt.jac_raw)
end

function compress_hessian!(kkt::AbstractSparseKKTSystem)
    transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end



#=
    SparseKKTSystem
=#

# Build KKT system directly from SparseCallback
function create_kkt_system(
    ::Type{SparseKKTSystem},
    nlp::SparseCallback{T,VT}, opt, 
    opt_linear_solver,cnt, ind_cons
    ) where {T,VT}
    
    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.

    n = nlp.nvar
    m = nlp.ncon
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(nlp, Int32, nlp.nnzj)
    jac_sparsity_J = create_array(nlp, Int32, nlp.nnzj)
    _jac_sparsity_wrapper!(nlp,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(opt.hessian_approximation, nlp, n)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(nlp, opt.hessian_approximation)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    # TODO make this work on GPU
    # force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    ind_ineq = ind_cons.ind_ineq
    
    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack


    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = create_array(nlp, Int32, aug_mat_length)
    J = create_array(nlp, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    pr_diag = _madnlp_unsafe_wrap(V, n_tot)
    du_diag = _madnlp_unsafe_wrap(V, m, n_jac+n_slack+n_hess+n_tot+1)

    reg = VT(undef, n_tot)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    
    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac+n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )
    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    cnt.linear_solver_time += @elapsed linear_solver = opt.linear_solver(
        aug_com; opt = opt_linear_solver
    )

    return SparseKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower, 
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        linear_solver,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )

end

is_reduced(::SparseKKTSystem) = true
num_variables(kkt::SparseKKTSystem) = length(kkt.pr_diag)


#=
    SparseUnreducedKKTSystem
=#

function create_kkt_system(
    ::Type{SparseUnreducedKKTSystem},
    nlp::SparseCallback{T,VT}, opt, 
    opt_linear_solver,cnt, ind_cons
    ) where {T, VT}
    ind_ineq = ind_cons.ind_ineq
    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub
    
    n_slack = length(ind_ineq)
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    # Deduce KKT size.
    n = nlp.nvar
    m = nlp.ncon

    # Quasi-newton
    quasi_newton = create_quasi_newton(opt.hessian_approximation, nlp, n)
    
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(nlp, Int32, nlp.nnzj)
    jac_sparsity_J = create_array(nlp, Int32, nlp.nnzj)
    _jac_sparsity_wrapper!(nlp,jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I = create_array(nlp, Int32, nlp.nnzh)
    hess_sparsity_J = create_array(nlp, Int32, nlp.nnzh)
    _hess_sparsity_wrapper!(nlp,hess_sparsity_I,hess_sparsity_J)

    # TODO make this work on GPU
    # force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)
    
    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack + 2*nlb + 2*nub
    aug_vec_length = n_tot + m + nlb + nub

    I = create_array(nlp, Int32, aug_mat_length)
    J = create_array(nlp, Int32, aug_mat_length)
    V = zeros(aug_mat_length)

    offset = n_tot + n_jac + n_slack + n_hess + m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac].=(jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack].=(ind_ineq.+n_tot)
    I[n_tot+n_hess+n_jac+n_slack+1:offset].=(n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset].=(n_tot+1:n_tot+m)

    I[offset+1:offset+nlb] .= (1:nlb).+(n_tot+m)
    I[offset+nlb+1:offset+2nlb] .= (1:nlb).+(n_tot+m)
    I[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n_tot+m+nlb)
    I[offset+2nlb+nub+1:offset+2nlb+2nub] .= (1:nub).+(n_tot+m+nlb)
    J[offset+1:offset+nlb] .= (1:nlb).+(n_tot+m)
    J[offset+nlb+1:offset+2nlb] .= ind_lb
    J[offset+2nlb+1:offset+2nlb+nub] .= (1:nub).+(n_tot+m+nlb)
    J[offset+2nlb+nub+1:offset+2nlb+2nub] .= ind_ub

    pr_diag = _madnlp_unsafe_wrap(V,n_tot)
    du_diag = _madnlp_unsafe_wrap(V,m, n_jac + n_slack+n_hess+n_tot+1)

    l_diag = _madnlp_unsafe_wrap(V, nlb, offset+1)
    u_diag = _madnlp_unsafe_wrap(V, nub, offset+2nlb+1)
    l_lower_aug = _madnlp_unsafe_wrap(V, nlb, offset+nlb+1)
    u_lower_aug = _madnlp_unsafe_wrap(V, nub, offset+2nlb+nub+1)
    reg = VT(undef, n_tot)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    hess = _madnlp_unsafe_wrap(V, n_hess, n_tot+1)
    jac = _madnlp_unsafe_wrap(V, n_jac + n_slack, n_hess+n_tot+1)
    jac_callback = _madnlp_unsafe_wrap(V, n_jac, n_hess+n_tot+1)

    hess_raw = SparseMatrixCOO(
        n_tot, n_tot,
        hess_sparsity_I,
        hess_sparsity_J,
        hess,
    )
    aug_raw = SparseMatrixCOO(aug_vec_length,aug_vec_length,I,J,V)
    jac_raw = SparseMatrixCOO(
        m, n_tot,
        Int32[jac_sparsity_I; ind_ineq],
        Int32[jac_sparsity_J; n+1:n+n_slack],
        jac,
    )

    aug_com, aug_csc_map = coo_to_csc(aug_raw)
    jac_com, jac_csc_map = coo_to_csc(jac_raw)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)

    cnt.linear_solver_time += @elapsed linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)
opt.linear_solver(
        aug_com; opt = opt_linear_solver
    )
    return SparseUnreducedKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower, l_lower_aug, u_lower_aug,
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        linear_solver,
        ind_ineq, ind_lb, ind_ub,
    )
end

function initialize!(kkt::AbstractSparseKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(kkt.hess_com.nzval, 0.) # so that mul! in the initial primal-dual solve has no effect 
end

function initialize!(kkt::SparseUnreducedKKTSystem) 
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(kkt.l_lower_aug, 0.0)
    fill!(kkt.u_lower_aug, 0.0)
    fill!(kkt.hess_com.nzval, 0.) # so that mul! in the initial primal-dual solve has no effect
end

is_reduced(::SparseUnreducedKKTSystem) = false
num_variables(kkt::SparseUnreducedKKTSystem) = length(kkt.pr_diag)


#=
    SparseCondensedKKTSystem
=#

# Build KKT system directly from SparseCallback
function create_kkt_system(
    ::Type{SparseCondensedKKTSystem},
    nlp::SparseCallback{T,VT},
    opt, 
    opt_linear_solver,
    cnt,
    ind_cons
    ) where {T, VT}

    ind_ineq = ind_cons.ind_ineq
    n = nlp.nvar
    m = nlp.ncon
    n_slack = length(ind_ineq)
    
    if n_slack != m
        error("SparseCondensedKKTSystem does not support equality constrained NLPs.")
    end
    
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(nlp, Int32, nlp.nnzj)
    jac_sparsity_J = create_array(nlp, Int32, nlp.nnzj)
    _jac_sparsity_wrapper!(nlp,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(opt.hessian_approximation, nlp, n)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(nlp, opt.hessian_approximation)

    # TODO make this work on GPU
    # force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)


    reg = VT(undef, n_tot)
    pr_diag = VT(undef, n_tot)
    du_diag = VT(undef, m)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    buffer = VT(undef, m)
    buffer2= VT(undef, m)
    hess = VT(undef, n_hess)
    jac = VT(undef, n_jac)
    diag_buffer = VT(undef, m)
    
    hess_raw = SparseMatrixCOO(n, n, hess_sparsity_I, hess_sparsity_J, hess)
    
    jt_coo = SparseMatrixCOO(
        n, m, 
        jac_sparsity_J,
        jac_sparsity_I,
        jac,
    )
    
    jt_csc, jt_csc_map = coo_to_csc(jt_coo)
    hess_com, hess_csc_map = coo_to_csc(hess_raw)
    
    aug_com, dptr, hptr, jptr = build_condensed_aug_symbolic(
        hess_com,
        jt_csc
    )


    cnt.linear_solver_time += @elapsed linear_solver = opt.linear_solver(aug_com; opt = opt_linear_solver)

    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)

    return SparseCondensedKKTSystem( 
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map, 
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer, buffer2,
        aug_com, diag_buffer, dptr, hptr, jptr,
        linear_solver,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        ext
    )
end

    
get_sparse_condensed_ext(::Type{Vector{T}},args...) where T = nothing


is_reduced(::SparseCondensedKKTSystem) = true
num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)
function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == num_variables(kkt))
end


Base.size(kkt::SparseCondensedKKTSystem,n::Int) = size(kkt.aug_com,n)
# nnz_jacobian(kkt::SparseCondensedKKTSystem) = nnz(kkt.jac_raw)


function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    # kkt.jac[end-ns+1:end] .= -1.0
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function mul!(y::AbstractKKTVector, kkt::SparseCondensedKKTSystem, x::AbstractKKTVector)
    mul!(full(y), kkt, full(x))
end

function mul!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    # TODO: implement properly with AbstractKKTRHS
    if length(y) == length(x) == size(kkt.aug_com, 1)
        mul!(y, Symmetric(kkt.aug_com, :L), x)
        return
    end

    n = size(kkt.hess_csc, 1)
    m = size(kkt.jt_csc, 2)


    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+m)
    Σd = kkt.du_diag

    # Decompose x
    xx = view(x, 1:n)
    xs = view(x, 1+n:n+m)
    xy = view(x, 1+n+m:n+2*m)

    # Decompose y
    yx = view(y, 1:n)
    ys = view(y, 1+n:n+m)
    yy = view(y, 1+n+m:n+2*m)

    # / x (variable)
    mul!(yx, Symmetric(kkt.hess_csc, :L), xx)
    yx .+= Σx .* xx
    mul!(yx, kkt.jt_csc, xy, 1.0, 1.0)

    # / s (slack)
    ys .= Σs .* xs
    ys .-= xy

    # / y (multiplier)
    yy .= Σd .* xy
    mul!(yy, kkt.jt_csc', xx, 1.0, 1.0)
    yy .-= xs
end

function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[size(kkt.jt_csc,1)+1:end] .= -x
end

function _sym_length(Jt)
    len = 0
    for i=1:size(Jt,2)
        n = Jt.colptr[i+1] - Jt.colptr[i]
        len += div(n^2 + n, 2)
    end
    return len
end

@inbounds function build_condensed_aug_symbolic(H::SparseMatrixCSC{Tv,Ti}, Jt::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    nnzjtsj = _sym_length(Jt)
    
    sym = Vector{Tuple{Int,Int,Int}}(
        undef,
        size(H,2) + nnz(H) + nnzjtsj
    )
    sym2 = Vector{Tuple{Int,Int}}(
        undef,
        size(H,2) + nnz(H) + nnzjtsj
    )
    dptr = Vector{Tuple{Ti,Ti}}(
        undef,
        size(H,2)
    )
    hptr = Vector{Tuple{Ti,Ti}}(
        undef,
        nnz(H)
    )
    jptr = Vector{Tuple{Ti,Tuple{Ti,Ti,Ti}}}(
        undef,
        nnzjtsj
    )
    colptr = fill!(
        Vector{Ti}(undef,size(H,1)+1),
        one(Tv)
    )
    rowval = Ti[]
    
    n = size(H,2)
    
    map!(
        i->(-1,i,0),
        @view(sym[1:n]),
        1:size(H,2)
    )
    map!(
        i->(i,i),
        @view(sym2[1:n]),
        1:size(H,2)
    )

    cnt = n
    
    for i in 1:size(H,2)
        for j in H.colptr[i]:H.colptr[i+1]-1
            c = H.rowval[j]
            sym[cnt+=1] = (0,j,0)
            sym2[cnt] = (c,i)
        end
    end

    for i in 1:size(Jt,2)
        for j in Jt.colptr[i]:Jt.colptr[i+1]-1
            for k in j:Jt.colptr[i+1]-1
                c1 = Jt.rowval[j]
                c2 = Jt.rowval[k]
                sym[cnt+=1] = (i,j,k)
                sym2[cnt] = (c2,c1)
            end
        end
    end
    p = sortperm(sym2; by = ((i,j),) -> (j,i), alg=Base.Sort.MergeSort)
    permute!(sym, p)
    permute!(sym2, p)

    a = (0,0)
    cnt = 0
    dcnt = 0
    hcnt = 0
    jcnt = 0
    prevcol = 0
    
    for (new, tuple) in zip(sym2,sym)

        if new != a
            cnt += 1
            
            (row,col) = new
            push!(rowval, row)
            a = new
            if prevcol != col
                fill!(@view(colptr[prevcol+1:col]), cnt)
                prevcol = col
            end
        end

        if tuple[1] == -1
            dptr[dcnt += 1] = (cnt, tuple[2])
        elseif tuple[1] == 0
            hptr[hcnt += 1] = (cnt, tuple[2])
        else
            jptr[jcnt += 1] = (cnt, tuple)
        end
    end

    fill!(
        @view(colptr[prevcol+1:end]),
        cnt+1
    )

    aug_com = SparseMatrixCSC{Tv,Ti}(
        size(H)...,
        colptr, rowval, zeros(cnt)
    )

    return aug_com, dptr, hptr, jptr
end

@inbounds function _build_condensed_aug_coord!(aug_com::SparseMatrixCSC{Tv,Ti}, pr_diag, H, Jt, diag_buffer, dptr, hptr, jptr) where {Tv, Ti}
    fill!(aug_com.nzval, zero(Tv))
    
    @simd for idx in eachindex(hptr)
        i,j = hptr[idx]
        aug_com.nzval[i] += H.nzval[j]
    end
    
    @simd for idx in eachindex(dptr)
        i,j = dptr[idx]
        aug_com.nzval[i] += pr_diag[j]
    end

    @simd for idx in eachindex(jptr)
        (i,(j,k,l)) = jptr[idx]
        aug_com.nzval[i] += diag_buffer[j] * Jt.nzval[k] * Jt.nzval[l]
    end
end

function build_condensed_aug_coord!(kkt::SparseCondensedKKTSystem{T,VT,MT}) where {T, VT, MT <: SparseMatrixCSC{T}}
    _build_condensed_aug_coord!(
        kkt.aug_com, kkt.pr_diag, kkt.hess_com, kkt.jt_csc, kkt.diag_buffer,
        kkt.dptr, kkt.hptr, kkt.jptr
    )
end


function build_kkt!(kkt::SparseKKTSystem)

    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end

function build_kkt!(kkt::SparseUnreducedKKTSystem)
    
    transfer!(kkt.aug_com, kkt.aug_raw, kkt.aug_csc_map)
end

function build_kkt!(kkt::SparseCondensedKKTSystem)

    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)


    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, n+1:n+m)
    Σd = kkt.du_diag
        
    kkt.diag_buffer .= Σs ./ ( 1 .- Σd .* Σs)
    build_condensed_aug_coord!(kkt)
end

get_jacobian(kkt::SparseCondensedKKTSystem) = kkt.jac
