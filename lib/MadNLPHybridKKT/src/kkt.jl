struct HybridCondensedKKTSystem{T, VT, MT, QN, VI, VI32, VInd, SC, LS, LS2, EXT} <: AbstractCondensedKKTSystem{T, VT, MT, QN}
    # Hessian
    hess::VT      # dimension nnzh
    hess_raw::SparseMatrixCOO{T, Int32, VT, VI32}
    hess_com::MT  # dimension n x n
    hess_csc_map::Union{Nothing, VI}

    # Full Jacobian
    jac::VT       # dimension nnzj
    jt_coo::SparseMatrixCOO{T, Int32, VT, VI32}
    jt_csc::MT
    jt_csc_map::Union{Nothing, VI}

    # Jacobian of equality constraints
    G_csc::MT
    G_csc_map::Union{Nothing, VI}
    ind_eq_jac::Union{Nothing, VI}

    # Schur-complement operator
    S::SC

    gamma::Ref{T}

    quasi_newton::QN
    reg::VT       # dimension n + mi
    pr_diag::VT   # dimension n + mi
    du_diag::VT   # dimension me + mi
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT

    # Buffer
    buffer1::VT   # dimension me + mi
    buffer2::VT   # dimension me + mi
    buffer3::VT   # dimension n
    buffer4::VT   # dimension mi
    buffer5::VT   # dimension mi
    buffer6::VT   # dimension me

    # Condensed system Kγ
    aug_com::MT   # dimension n x n

    # slack diagonal buffer
    diag_buffer::VT
    dptr::AbstractVector
    hptr::AbstractVector
    jptr::AbstractVector

    # LinearSolver
    linear_solver::LS
    iterative_linear_solver::LS2

    # Info
    ind_ineq::VInd  # dimension mi
    ind_eq::VInd    # dimension me
    ind_lb::VI
    ind_ub::VI

    ext::EXT
    # Stats
    etc::Dict{Symbol, Any}
end

# Build KKT system directly from SparseCallback
function create_kkt_system(
        ::Type{HybridCondensedKKTSystem},
        cb::SparseCallback{T, VT},
        linear_solver;
        opt_linear_solver = default_options(linear_solver),
        hessian_approximation = ExactHessian,
        qn_options = QuasiNewtonOptions(),
        cg_algorithm = :cg,
    ) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ind_ineq = cb.ind_ineq
    mi = length(ind_ineq)
    VI = typeof(ind_ineq)

    ind_eq = if isa(ind_ineq, Vector)
        setdiff(1:m, ind_ineq)
    else
        ind_ineq_host = Vector(ind_ineq)
        VI(setdiff(1:m, ind_ineq_host))
    end
    me = m - mi

    # Evaluate sparsity pattern
    jac_sparsity_I = create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = create_array(cb, Int32, cb.nnzj)
    _jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    quasi_newton = create_quasi_newton(ExactHessian, cb, n)
    hess_sparsity_I, hess_sparsity_J = build_hessian_structure(cb, ExactHessian)

    force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + mi
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    reg = VT(undef, n_tot)
    pr_diag = VT(undef, n_tot)
    du_diag = VT(undef, m)
    l_diag = VT(undef, nlb)
    u_diag = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)
    buffer1 = VT(undef, m)
    buffer2 = VT(undef, m)
    buffer3 = VT(undef, n)
    buffer4 = VT(undef, mi)
    buffer5 = VT(undef, mi)
    buffer6 = VT(undef, me)
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
    init_condensation = @elapsed_hykkt begin
        aug_com, dptr, hptr, jptr = build_condensed_aug_symbolic(
            hess_com,
            jt_csc
        )
    end

    # Build Jacobian of equality constraints
    jac_coo = SparseMatrixCOO(
        m, n,
        Vector(jac_sparsity_I),
        Vector(jac_sparsity_J),
        Vector(jac),
    )
    G_csc_, G_csc_map_, ind_eq_jac_ = _extract_subjacobian(jac_coo, Vector(ind_eq))
    MT = typeof(hess_com)
    G_csc = MT(G_csc_)
    G_csc_map = VI(G_csc_map_)
    ind_eq_jac = VI(ind_eq_jac_)

    gamma = Ref{T}(1000)

    init_linear_solver = @elapsed_hykkt begin
        linear_solver = linear_solver(aug_com; opt = opt_linear_solver)
    end

    buf1 = VT(undef, n)
    S = if cg_algorithm ∈ (:cg, :gmres, :cr, :minres, :car)
        SchurComplementOperator(linear_solver, G_csc, buf1)
    elseif cg_algorithm ∈ (:craigmr,)
        CondensedOperator(linear_solver, buf1)
    end

    iterative_linear_solver = if cg_algorithm == :cg
        Krylov.CgWorkspace(me, me, VT)
    elseif cg_algorithm == :cr
        Krylov.CrWorkspace(me, me, VT)
    elseif cg_algorithm == :car
        Krylov.CarWorkspace(me, me, VT)
    elseif cg_algorithm == :gmres
        Krylov.GmresWorkspace(me, me, 10, VT)
    elseif cg_algorithm == :minres
        Krylov.MinresWorkspace(me, me, VT)
    elseif cg_algorithm == :craigmr
        Krylov.CraigmrWorkspace(me, n, VT)
    end

    ext = get_sparse_condensed_ext(VT, hess_com, jptr, jt_csc_map, hess_csc_map)
    etc = Dict{Symbol, Any}(
        :cg_algorithm => cg_algorithm,
        :cg_iters => Int[],
        :accuracy => Float64[],
        :time_cg => 0.0,
        :time_backsolve => 0.0,
        :time_condensation => 0.0,
        :time_init_condensation => init_condensation,
        :time_init_linear_solver => init_linear_solver,
    )

    return HybridCondensedKKTSystem(
        hess, hess_raw, hess_com, hess_csc_map,
        jac, jt_coo, jt_csc, jt_csc_map,
        G_csc, G_csc_map, ind_eq_jac,
        S,
        gamma,
        quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        buffer1, buffer2, buffer3, buffer4, buffer5, buffer6,
        aug_com, diag_buffer, dptr, hptr, jptr,
        linear_solver, iterative_linear_solver,
        ind_ineq, ind_eq, cb.ind_lb, cb.ind_ub,
        ext, etc,
    )
end

function initialize!(kkt::HybridCondensedKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.hess, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    return fill!(nonzeros(kkt.hess_com), 0.0) # so that mul! in the initial primal-dual solve has no effect
end

function is_inertia_correct(kkt::HybridCondensedKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == size(kkt.aug_com, 1))
end

# mul!
function LinearAlgebra.mul!(w::AbstractKKTVector{T}, kkt::HybridCondensedKKTSystem, x::AbstractKKTVector, alpha, beta) where {T}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), (n + 1):(n + mi))
    xz = view(full(x), (n + mi + 1):(n + mi + m))

    # Decompose buffers
    wx = view(full(w), 1:n)
    ws = view(full(w), (n + 1):(n + mi))
    wz = view(full(w), (n + mi + 1):(n + mi + m))

    wz_ineq = view(wz, kkt.ind_ineq)
    xz_ineq = view(xz, kkt.ind_ineq)

    mul!(wx, Symmetric(kkt.hess_com, :L), xx, alpha, beta)

    mul!(wx, kkt.jt_csc, xz, alpha, one(T))
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    axpy!(-alpha, xs, wz_ineq)

    ws .= beta .* ws .- alpha .* xz_ineq

    _kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

# get_jacobian
get_jacobian(kkt::HybridCondensedKKTSystem) = kkt.jac

# compress_jacobian!
function compress_jacobian!(kkt::HybridCondensedKKTSystem)
    fill!(nonzeros(kkt.jt_csc), 0.0)
    transfer!(kkt.jt_csc, kkt.jt_coo, kkt.jt_csc_map)
    transfer_coef!(kkt.G_csc, kkt.G_csc_map, kkt.jac, kkt.ind_eq_jac)
    return
end

# jtprod!
function jtprod!(y::AbstractVector, kkt::HybridCondensedKKTSystem, x::AbstractVector)
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)

    x_ineq = view(x, kkt.ind_ineq)
    mul!(view(y, 1:n), kkt.jt_csc, x)
    y[(size(kkt.jt_csc, 1) + 1):end] .= -x_ineq
    return y
end

function compress_hessian!(kkt::HybridCondensedKKTSystem)
    return transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

# build_kkt!
function build_kkt!(kkt::HybridCondensedKKTSystem)
    n = size(kkt.hess_com, 1)
    mi = length(kkt.ind_ineq)
    m = size(kkt.jt_csc, 2)

    Σx = view(kkt.pr_diag, 1:n)
    Σs = view(kkt.pr_diag, (n + 1):(n + mi))
    Σd = kkt.du_diag # TODO: add support

    fill!(kkt.diag_buffer, 0.0)
    index_copy!(kkt.diag_buffer, kkt.ind_ineq, Σs)
    # Regularization for equality
    fixed!(kkt.diag_buffer, kkt.ind_eq, kkt.gamma[])
    # Condensation
    kkt.etc[:time_condensation] += @elapsed_hykkt begin
        build_condensed_aug_coord!(kkt)
    end
    return
end

# solve!
function solve_kkt!(kkt::HybridCondensedKKTSystem{T}, w::AbstractKKTVector) where {T}
    (n, m) = size(kkt.jt_csc)
    mi = length(kkt.ind_ineq)
    G = kkt.G_csc

    # Decompose buffers
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), (n + 1):(n + mi))
    wc = view(full(w), (n + mi + 1):(n + mi + m))

    r1 = kkt.buffer3
    vs = kkt.buffer4
    wz = kkt.buffer5
    wy = kkt.buffer6

    index_copy!(wy, wc, kkt.ind_eq)
    index_copy!(wz, wc, kkt.ind_ineq)

    Σs = view(kkt.pr_diag, (n + 1):(n + mi))

    reduce_rhs!(w.xp_lr, dual_lb(w), kkt.l_diag, w.xp_ur, dual_ub(w), kkt.u_diag)

    # Condensation
    fill!(kkt.buffer1, zero(T))
    vs .= Σs .* wz .+ ws
    index_copy!(kkt.buffer1, kkt.ind_ineq, vs)
    mul!(wx, kkt.jt_csc, kkt.buffer1, one(T), one(T))

    #  Golub & Greif
    r1 .= wx
    mul!(r1, G', wy, kkt.gamma[], one(T))                   # r1 = wx + γ Gᵀ wy
    wx .= r1                                                # (save for later)
    kkt.etc[:time_backsolve] += @elapsed_hykkt begin
        solve_linear_system!(kkt.linear_solver, r1)  # r1 = (Kγ)⁻¹ [wx + γ Gᵀ wy]
    end
    mul!(wy, G, r1, one(T), -one(T))                        # -wy + G (Kγ)⁻¹ [wx + γ Gᵀ wy]

    # Solve Schur-complement system with a Krylov iterative method.
    if kkt.etc[:cg_algorithm] ∈ (:cg, :gmres, :cr, :minres, :car)
        t_cg = @elapsed_hykkt Krylov.krylov_solve!(
            kkt.iterative_linear_solver,
            kkt.S,
            wy;
            atol = 0.0,
            rtol = 1.0e-10,
            verbose = 0,
        )
        copyto!(wy, kkt.iterative_linear_solver.x)
    elseif kkt.etc[:cg_algorithm] ∈ (:craigmr,)
        t_cg = @elapsed_hykkt Krylov.krylov_solve!(
            kkt.iterative_linear_solver,
            kkt.G_csc,
            wy;
            N = kkt.S,
            atol = 0.0,
            rtol = 1.0e-10,
            verbose = 0,
        )
        copyto!(wy, kkt.iterative_linear_solver.y)
    end
    kkt.etc[:time_cg] += t_cg

    # Extract solution of Golub & Greif
    mul!(wx, G', wy, -one(T), one(T))
    kkt.etc[:time_backsolve] += @elapsed_hykkt begin
        solve_linear_system!(kkt.linear_solver, wx)
    end

    # Extract condensation
    mul!(kkt.buffer2, kkt.jt_csc', wx)
    vj = view(kkt.buffer2, kkt.ind_ineq)
    vs .= ws    # (save a copy of ws for later)
    ws .= vj .- wz
    wz .= Σs .* ws .- vs

    index_copy!(wc, kkt.ind_ineq, wz)
    index_copy!(wc, kkt.ind_eq, wy)

    finish_aug_solve!(kkt, w)

    # Save current number of CG iterations for later.
    cg_iter = kkt.iterative_linear_solver.stats.niter
    push!(kkt.etc[:cg_iters], cg_iter)

    return w
end

# Custom iterative-refinement
function solve_refine_wrapper!(
        d,
        solver::MadNLPSolver{T, VT, VI, KKT},
        p,
        w,
    ) where {T, VT, VI, KKT <: HybridCondensedKKTSystem{T}}
    copyto!(d.values, p.values)

    solver.cnt.linear_solver_time += @elapsed_hykkt begin
        solve_kkt!(solver.kkt, d)
    end

    # Compute backsolve's error
    copyto!(full(w), full(p))
    mul!(w, solver.kkt, d, -one(T), one(T))
    norm_w = norm(full(w), 2)
    norm_b = norm(full(p), 2)

    residual_ratio = norm_w / (1.0 + norm_b)
    @debug(solver.logger, @sprintf("%4i %6.2e", 0, residual_ratio))

    push!(solver.kkt.etc[:accuracy], residual_ratio)
    return true
end

#=
    GPU-specific code. Dispatched on the vector type being an AbstractGPUVector
    (GPUArraysCore) and run on whatever backend the arrays live on (get_backend),
    so this stays backend-agnostic — no hard CUDA dependency. The transfer/diag
    kernels come from MadCoreKernelAbstractions.
=#

function compress_hessian!(kkt::HybridCondensedKKTSystem{T, VT, MT}) where {T, VT <: AbstractGPUVector{T}, MT}
    fill!(kkt.hess_com.nzVal, zero(T))
    return if length(kkt.ext.hess_com_ptrptr) > 1
        backend = get_backend(kkt.hess_com.nzVal)
        MadCoreKernelAbstractions._transfer_to_csc_kernel!(backend)(
            kkt.hess_com.nzVal,
            kkt.ext.hess_com_ptr,
            kkt.ext.hess_com_ptrptr,
            kkt.hess_raw.V;
            ndrange = length(kkt.ext.hess_com_ptrptr) - 1,
        )
        KernelAbstractions.synchronize(backend)
    end
end

function compress_jacobian!(kkt::HybridCondensedKKTSystem{T, VT, MT}) where {T, VT <: AbstractGPUVector{T}, MT}
    fill!(kkt.jt_csc.nzVal, zero(T))
    if length(kkt.ext.jt_csc_ptrptr) > 1 # otherwise error is thrown
        backend = get_backend(kkt.jt_csc.nzVal)
        MadCoreKernelAbstractions._transfer_to_csc_kernel!(backend)(
            kkt.jt_csc.nzVal,
            kkt.ext.jt_csc_ptr,
            kkt.ext.jt_csc_ptrptr,
            kkt.jt_coo.V;
            ndrange = length(kkt.ext.jt_csc_ptrptr) - 1,
        )
        KernelAbstractions.synchronize(backend)
    end
    return if length(kkt.ind_eq) > 0
        transfer_coef!(kkt.G_csc, kkt.G_csc_map, kkt.jac, kkt.ind_eq_jac)
    end
end

# N.B: we use the custom diag kernel from MadCoreKernelAbstractions for KKT
# multiplication as symv is not supported on the GPU.
function LinearAlgebra.mul!(
        w::AbstractKKTVector{T, VT},
        kkt::HybridCondensedKKTSystem,
        x::AbstractKKTVector{T, VT},
        alpha = one(T), beta = zero(T)
    ) where {T, VT <: AbstractGPUVector{T}}
    n = size(kkt.hess_com, 1)
    m = size(kkt.jt_csc, 2)
    mi = length(kkt.ind_ineq)

    # Decompose results
    xx = view(full(x), 1:n)
    xs = view(full(x), (n + 1):(n + mi))
    xz = view(full(x), (n + mi + 1):(n + mi + m))

    # Decompose buffers
    wx = view(full(w), 1:n)
    ws = view(full(w), (n + 1):(n + mi))
    wz = view(full(w), (n + mi + 1):(n + mi + m))

    wz_ineq = kkt.buffer4
    xz_ineq = kkt.buffer5

    # First block / x
    mul!(wx, kkt.hess_com, xx, alpha, beta)
    mul!(wx, kkt.hess_com', xx, alpha, one(T))
    mul!(wx, kkt.jt_csc, xz, alpha, one(T))
    if !isempty(kkt.ext.diag_map_to)
        backend = get_backend(wx)
        MadCoreKernelAbstractions._diag_operation_kernel!(backend)(
            wx,
            kkt.hess_com.nzVal,
            xx,
            alpha,
            kkt.ext.diag_map_to,
            kkt.ext.diag_map_fr;
            ndrange = length(kkt.ext.diag_map_to)
        )
        KernelAbstractions.synchronize(backend)
    end

    # Second block / s
    # N.B. axpy! does not support SubArray
    index_copy!(xz_ineq, xz, kkt.ind_ineq)
    ws .= beta .* ws .- alpha .* xz_ineq

    # Third block / y
    # TODO: memory issue
    mul!(wz, kkt.jt_csc', xx, alpha, beta)
    # Implements wz = wz - alpha * xs
    index_copy!(wz_ineq, wz, kkt.ind_ineq)
    axpy!(-alpha, xs, wz_ineq)
    index_copy!(wz, kkt.ind_ineq, wz_ineq)

    _kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return
end
