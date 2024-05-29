#=
    SparseKKTSystem
=#

# Build KKT system directly from SparseCallback
function create_kkt_system(
    ::Type{SparseKKTSystem},
    cb::SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
) where {T,VT}

    n_slack = length(ind_cons.ind_ineq)
    # Deduce KKT size.

    n = cb.nvar
    m = cb.ncon
    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)

    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    ind_ineq = ind_cons.ind_ineq

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack


    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = create_array(cb, Int32, aug_mat_length)
    J = create_array(cb, Int32, aug_mat_length)
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

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )

    return SparseKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
    )

end

num_variables(kkt::SparseKKTSystem) = length(kkt.pr_diag)


#=
    SparseUnreducedKKTSystem
=#

function create_kkt_system(
    ::Type{SparseUnreducedKKTSystem},
    cb::SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
) where {T, VT}
    ind_ineq = ind_cons.ind_ineq
    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub

    n_slack = length(ind_ineq)
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)
    # Deduce KKT size.
    n = cb.nvar
    m = cb.ncon

    # Quasi-newton
    quasi_newton = create_quasi_newton(hessian_approximation, cb, n)

    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I = create_array(cb, Int32, cb.nnzh)
    hess_sparsity_J = create_array(cb, Int32, cb.nnzh)
    _hess_sparsity_wrapper!(cb,hess_sparsity_I,hess_sparsity_J)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_mat_length = n_tot + m + n_hess + n_jac + n_slack + 2*nlb + 2*nub
    aug_vec_length = n_tot + m + nlb + nub

    I = create_array(cb, Int32, aug_mat_length)
    J = create_array(cb, Int32, aug_mat_length)
    V = VT(undef, aug_mat_length)
    fill!(V, 0.0)  # Need to initiate V to avoid NaN

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

    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    return SparseUnreducedKKTSystem(
        hess, jac_callback, jac, quasi_newton, reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower, l_lower_aug, u_lower_aug,
        aug_raw, aug_com, aug_csc_map,
        hess_raw, hess_com, hess_csc_map,
        jac_raw, jac_com, jac_csc_map,
        _linear_solver,
        ind_ineq, ind_lb, ind_ub,
    )
end

function initialize!(kkt::AbstractSparseKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

function initialize!(kkt::SparseUnreducedKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(kkt.l_lower_aug, zero(T))
    fill!(kkt.u_lower_aug, zero(T))
    fill!(nonzeros(kkt.hess_com), zero(T)) # so that mul! in the initial primal-dual solve has no effect
end

num_variables(kkt::SparseUnreducedKKTSystem) = length(kkt.pr_diag)

function is_inertia_correct(kkt::SparseUnreducedKKTSystem, num_pos, num_zero, num_neg)
    n, nlb, nub = num_variables(kkt), length(kkt.ind_lb), length(kkt.ind_ub)
    return (num_zero == 0) && (num_pos == n + nlb + nub)
end

#=
    SparseCondensedKKTSystem
=#

# Build KKT system directly from SparseCallback
function create_kkt_system(
    ::Type{SparseCondensedKKTSystem},
    cb::SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
) where {T, VT}
    ind_ineq = ind_cons.ind_ineq
    n = cb.nvar
    m = cb.ncon
    n_slack = length(ind_ineq)

    if n_slack != m
        error("SparseCondensedKKTSystem does not support equality constrained NLPs.")
    end

    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb,jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, hessian_approximation)

    force_lower_triangular!(hess_sparsity_I,hess_sparsity_J)

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
    fill!(jac, zero(T))

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
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    return SparseCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer, buffer2,
        aug_com, diag_buffer, dptr, hptr, jptr,
        _linear_solver,
        ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        ext
    )
end

num_variables(kkt::SparseCondensedKKTSystem) = length(kkt.pr_diag)

function is_inertia_correct(kkt::SparseCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end

Base.size(kkt::SparseCondensedKKTSystem,n::Int) = size(kkt.aug_com,n)

function compress_jacobian!(kkt::SparseCondensedKKTSystem{T, VT, MT}) where {T, VT, MT<:SparseMatrixCSC{T, Int32}}
    ns = length(kkt.ind_ineq)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
end

function jtprod!(y::AbstractVector, kkt::SparseCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[size(kkt.jt_csc,1)+1:end] .= -x
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
