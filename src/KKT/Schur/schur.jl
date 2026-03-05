
"""
    SchurComplementKKTSystem{T, VT, MT, QN} <: AbstractCondensedKKTSystem{T, VT, MT, QN}

KKT system exploiting block-arrowhead structure from two-stage stochastic programs
via Schur complement decomposition.

Variable layout: `[v_1, ..., v_ns, d]` where `v_k ∈ R^nv`, `d ∈ R^nd`.
Constraint layout: `[c_1, ..., c_ns]` where `c_k ∈ R^nc`.

The augmented per-scenario block is `(nv + nc) × (nv + nc)`:
```
A_k = [H_kk + Σx_k + J_kv' diag(db_k) J_kv    J_kv' diag(db_k) J_kd    J_kv_eq' ]
      [J_kd' diag(db_k) J_kv                    (coupling to S)                     ]
      [J_kv_eq                                   du_diag_eq_k                        ]
```

For inequalities, they get condensed (standard MadNLP condensed form).
For equalities, they enter the augmented blocks.

The Schur complement eliminates `(Δv_k, Δy_eq_k)` per scenario, leaving a dense
`nd × nd` system on design variables solved by the pluggable linear solver.
"""
struct SchurComplementKKTSystem{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    QN,
    LS,
    VI <: AbstractVector{Int}
    } <: AbstractCondensedKKTSystem{T, VT, MT, QN}

    # Full dense Hessian/Jacobian (filled by NLP callback)
    hess::MT                    # n_total × n_total
    jac::MT                     # m_total × n_total
    quasi_newton::QN

    # Standard MadNLP diagonal vectors
    reg::VT                     # n_total + n_ineq
    pr_diag::VT                 # n_total + n_ineq
    du_diag::VT                 # m_total
    l_diag::VT                  # nlb
    u_diag::VT                  # nub
    l_lower::VT                 # nlb
    u_lower::VT                 # nub

    # Two-stage dimensions
    ns::Int                     # number of scenarios
    nv::Int                     # recourse variables per scenario
    nd::Int                     # design variables
    nc::Int                     # constraints per scenario

    # Per-scenario augmented blocks (nv+nc_eq_k) × (nv+nc_eq_k)
    # We use full nc for simplicity (equality + inequality all per scenario)
    # Block size = nv + nc_eq_per_s (number of equality constraints per scenario)
    nc_eq_per_s::Int            # equality constraints per scenario
    nc_ineq_per_s::Int          # inequality constraints per scenario
    blk_size::Int               # = nv + nc_eq_per_s

    A_kk::Vector{MT}            # ns × (blk_size × blk_size) scenario augmented blocks
    C_dk::Vector{MT}            # ns × (nd × blk_size) coupling blocks
    A_kk_work::Vector{MT}       # ns × (blk_size × blk_size) factorization workspace
    A_kk_ipiv::Vector{Vector{BLAS.BlasInt}}  # pivot arrays for sytrf

    # Schur complement = aug_com (what the linear solver sees)
    aug_com::MT                 # nd × nd

    # Buffers
    diag_buffer::VT             # n_ineq — condensing diagonal
    buffer::VT                  # m_total — general
    rhs_d::VT                   # nd — design RHS
    rhs_k::Vector{VT}          # ns × blk_size — scenario RHS buffers
    tmp_blk_nd::Vector{MT}     # ns × (blk_size × nd) — for A_kk^{-1} * C_dk'

    # Mapping: which global equality constraints belong to scenario k
    eq_per_scenario::Vector{Vector{Int}}  # eq_per_scenario[k] = list of indices into ind_eq
    ineq_per_scenario::Vector{Vector{Int}}  # ineq_per_scenario[k] = list of indices into ind_ineq

    # Inequality/equality/bound index info
    n_eq::Int
    ind_eq::VI
    n_ineq::Int
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI

    # Linear solver (for Schur complement S)
    linear_solver::LS
    etc::Dict{Symbol, Any}
end

function create_kkt_system(
    ::Type{SchurComplementKKTSystem},
    cb::AbstractCallback{T,VT},
    linear_solver::Type;
    opt_linear_solver=default_options(linear_solver),
    hessian_approximation=ExactHessian,
    qn_options=QuasiNewtonOptions(),
    schur_ns::Int=0,
    schur_nv::Int=0,
    schur_nd::Int=0,
    schur_nc::Int=0,
) where {T, VT}

    n = cb.nvar
    m = cb.ncon
    ns_ineq = length(cb.ind_ineq)
    n_eq = m - ns_ineq
    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)

    @assert schur_ns > 0 "schur_ns must be specified and positive"
    @assert schur_nv > 0 "schur_nv must be specified and positive"
    @assert schur_nd > 0 "schur_nd must be specified and positive"
    @assert n == schur_ns * schur_nv + schur_nd "Variable count mismatch: n=$n != ns*nv+nd=$(schur_ns*schur_nv+schur_nd)"
    @assert m == schur_ns * schur_nc "Constraint count mismatch: m=$m != ns*nc=$(schur_ns*schur_nc)"

    ns = schur_ns
    nv = schur_nv
    nd = schur_nd
    nc = schur_nc

    # Classify constraints per scenario as equality or inequality
    ind_eq_set = Set(cb.ind_eq)
    ind_ineq_set = Set(cb.ind_ineq)

    eq_per_scenario = Vector{Vector{Int}}(undef, ns)
    ineq_per_scenario = Vector{Vector{Int}}(undef, ns)

    nc_eq_per_s = 0
    nc_ineq_per_s = 0
    for k in 1:ns
        cr = (k-1)*nc+1 : k*nc
        eq_per_scenario[k] = Int[]
        ineq_per_scenario[k] = Int[]
        for gi in cr
            if gi in ind_eq_set
                push!(eq_per_scenario[k], gi)
            end
            if gi in ind_ineq_set
                push!(ineq_per_scenario[k], gi)
            end
        end
        if k == 1
            nc_eq_per_s = length(eq_per_scenario[k])
            nc_ineq_per_s = length(ineq_per_scenario[k])
        end
    end

    blk_size = nv + nc_eq_per_s

    # Dense full matrices
    hess = create_array(cb, n, n)
    jac  = create_array(cb, m, n)
    aug_com = create_array(cb, nd, nd)

    # Diagonal vectors
    reg     = VT(undef, n + ns_ineq)
    pr_diag = VT(undef, n + ns_ineq)
    du_diag = VT(undef, m)
    l_diag  = fill!(VT(undef, nlb), one(T))
    u_diag  = fill!(VT(undef, nub), one(T))
    l_lower = fill!(VT(undef, nlb), zero(T))
    u_lower = fill!(VT(undef, nub), zero(T))

    # Per-scenario augmented blocks
    A_kk      = [Matrix{T}(undef, blk_size, blk_size) for _ in 1:ns]
    C_dk      = [Matrix{T}(undef, nd, blk_size) for _ in 1:ns]
    A_kk_work = [Matrix{T}(undef, blk_size, blk_size) for _ in 1:ns]
    A_kk_ipiv = [Vector{BLAS.BlasInt}(undef, blk_size) for _ in 1:ns]

    # Buffers
    diag_buffer = VT(undef, ns_ineq)
    buffer      = VT(undef, m)
    rhs_d       = VT(undef, nd)
    rhs_k       = [VT(undef, blk_size) for _ in 1:ns]
    tmp_blk_nd  = [Matrix{T}(undef, blk_size, nd) for _ in 1:ns]

    # Init
    fill!(aug_com, zero(T))
    fill!(hess,    zero(T))
    fill!(jac,     zero(T))
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    quasi_newton = create_quasi_newton(hessian_approximation, cb, n; options=qn_options)
    _linear_solver = linear_solver(aug_com; opt = opt_linear_solver)

    return SchurComplementKKTSystem(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag, l_diag, u_diag, l_lower, u_lower,
        ns, nv, nd, nc,
        nc_eq_per_s, nc_ineq_per_s, blk_size,
        A_kk, C_dk, A_kk_work, A_kk_ipiv,
        aug_com,
        diag_buffer, buffer, rhs_d, rhs_k, tmp_blk_nd,
        eq_per_scenario, ineq_per_scenario,
        n_eq, cb.ind_eq,
        ns_ineq, cb.ind_ineq, cb.ind_lb, cb.ind_ub,
        _linear_solver,
        Dict{Symbol, Any}(),
    )
end

num_variables(kkt::SchurComplementKKTSystem) = size(kkt.hess, 1)

function get_slack_regularization(kkt::SchurComplementKKTSystem)
    n = num_variables(kkt)
    ns_ineq = kkt.n_ineq
    return view(kkt.pr_diag, n+1:n+ns_ineq)
end

function is_inertia_correct(kkt::SchurComplementKKTSystem, num_pos, num_zero, num_neg)
    # The Schur complement S is nd × nd and should be positive definite
    # (equality constraints are absorbed into per-scenario blocks)
    return (num_zero == 0 && num_neg == 0)
end

function jtprod!(y::AbstractVector, kkt::SchurComplementKKTSystem, x::AbstractVector)
    nx = size(kkt.hess, 1)
    ns_ineq = kkt.n_ineq
    yx = view(y, 1:nx)
    ys = view(y, 1+nx:nx+ns_ineq)
    mul!(yx, kkt.jac', x)
    ys .= -@view(x[kkt.ind_ineq])
    return
end

compress_jacobian!(kkt::SchurComplementKKTSystem) = nothing
nnz_jacobian(kkt::SchurComplementKKTSystem) = length(kkt.jac)

# Helper: find index of gi in ind_ineq, return diag_buffer value (0 if not found)
function _get_ineq_diag(kkt::SchurComplementKKTSystem{T}, gi::Int) where T
    @inbounds for idx in 1:length(kkt.ind_ineq)
        if kkt.ind_ineq[idx] == gi
            return kkt.diag_buffer[idx]
        end
    end
    return zero(T)
end

function build_kkt!(kkt::SchurComplementKKTSystem{T, VT, MT}) where {T, VT, MT}
    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    nc = kkt.nc
    n = num_variables(kkt)
    blk = kkt.blk_size  # nv + nc_eq_per_s
    nc_eq = kkt.nc_eq_per_s

    # Compute condensing diagonal for inequalities
    if kkt.n_ineq > 0
        Sigma_s = view(kkt.pr_diag, n+1:n+kkt.n_ineq)
        Sigma_d = @view(kkt.du_diag[kkt.ind_ineq])
        kkt.diag_buffer .= Sigma_s ./ (one(T) .- Sigma_d .* Sigma_s)
    end

    # Initialize Schur complement with design block
    S = kkt.aug_com
    fill!(S, zero(T))

    # S starts with H_dd + pr_diag_dd
    @inbounds for i in 1:nd, j in 1:nd
        S[i, j] = kkt.hess[ns*nv+i, ns*nv+j]
    end
    @inbounds for i in 1:nd
        S[i, i] += kkt.pr_diag[ns*nv+i]
    end

    for k in 1:ns
        vr = (k-1)*nv+1 : k*nv    # variable range for scenario k
        eq_cons = kkt.eq_per_scenario[k]
        ineq_cons = kkt.ineq_per_scenario[k]

        A_kk = kkt.A_kk[k]
        C_dk = kkt.C_dk[k]
        fill!(A_kk, zero(T))
        fill!(C_dk, zero(T))

        # === Top-left block of A_kk: H_kk + pr_diag_kk + J_ineq' diag(db) J_ineq ===
        @inbounds for i in 1:nv, j in 1:nv
            A_kk[i, j] = kkt.hess[vr[1]+i-1, vr[1]+j-1]
        end
        @inbounds for i in 1:nv
            A_kk[i, i] += kkt.pr_diag[vr[1]+i-1]
        end

        # Add inequality condensation: J_ineq_k' * diag(db_k) * J_ineq_k to A_kk[1:nv, 1:nv]
        for gi in ineq_cons
            db_val = _get_ineq_diag(kkt, gi)
            if db_val == zero(T)
                continue
            end
            # Contribution to A_kk (top-left, nv × nv)
            @inbounds for i in 1:nv
                jac_i = kkt.jac[gi, vr[1]+i-1]
                for j in 1:nv
                    A_kk[i, j] += db_val * jac_i * kkt.jac[gi, vr[1]+j-1]
                end
            end
            # Contribution to C_dk (nd × nv part only) — coupling
            @inbounds for i in 1:nd
                jac_di = kkt.jac[gi, ns*nv+i]
                for j in 1:nv
                    C_dk[i, j] += db_val * jac_di * kkt.jac[gi, vr[1]+j-1]
                end
            end
            # Contribution to S (nd × nd)
            @inbounds for i in 1:nd
                jac_di = kkt.jac[gi, ns*nv+i]
                for j in 1:nd
                    S[i, j] += db_val * jac_di * kkt.jac[gi, ns*nv+j]
                end
            end
        end

        # === Bottom-left / top-right blocks of A_kk: J_eq_kv (nc_eq × nv) ===
        for (ci, gi) in enumerate(eq_cons)
            @inbounds for j in 1:nv
                val = kkt.jac[gi, vr[1]+j-1]
                A_kk[nv+ci, j] = val
                A_kk[j, nv+ci] = val
            end
            # du_diag on the diagonal of the equality block
            A_kk[nv+ci, nv+ci] = kkt.du_diag[gi]
        end

        # === C_dk: coupling block (nd × blk_size) ===
        # C_dk[1:nd, 1:nv] = H_dk (Hessian cross-term) — already has ineq contribution above
        @inbounds for i in 1:nd, j in 1:nv
            C_dk[i, j] += kkt.hess[ns*nv+i, vr[1]+j-1]
        end
        # C_dk[1:nd, nv+1:nv+nc_eq] = J_eq_kd' (Jacobian of eq constraints w.r.t. design)
        for (ci, gi) in enumerate(eq_cons)
            @inbounds for i in 1:nd
                C_dk[i, nv+ci] = kkt.jac[gi, ns*nv+i]
            end
        end

        # === Factor A_kk ===
        copyto!(kkt.A_kk_work[k], A_kk)
        LAPACK.sytrf!('L', kkt.A_kk_work[k], kkt.A_kk_ipiv[k])

        # === Compute tmp = A_kk^{-1} * C_dk' (blk × nd) ===
        @inbounds for i in 1:blk, j in 1:nd
            kkt.tmp_blk_nd[k][i, j] = C_dk[j, i]  # transpose
        end
        LAPACK.sytrs!('L', kkt.A_kk_work[k], kkt.A_kk_ipiv[k], kkt.tmp_blk_nd[k])

        # === S -= C_dk * A_kk^{-1} * C_dk' ===
        mul!(S, C_dk, kkt.tmp_blk_nd[k], -one(T), one(T))
    end

    return
end

function factorize_kkt!(kkt::SchurComplementKKTSystem)
    return factorize!(kkt.linear_solver)
end

function solve_kkt!(
    kkt::SchurComplementKKTSystem,
    w::AbstractKKTVector{T},
) where T

    ns = kkt.ns
    nv = kkt.nv
    nd = kkt.nd
    nc = kkt.nc
    n = num_variables(kkt)
    blk = kkt.blk_size

    # Decompose rhs into primal/slack/dual views
    wx = _madnlp_unsafe_wrap(full(w), n)
    ws = view(full(w), n+1:n+kkt.n_ineq)
    # dual(w) gives the dual part (length m)
    wy = dual(w)

    Sigma_s = get_slack_regularization(kkt)

    reduce_rhs!(kkt, w)

    # Condense inequalities into primal RHS
    if kkt.n_ineq > 0
        wz_ineq = view(full(w), n+kkt.n_ineq+1:n+2*kkt.n_ineq) # This doesn't exist for condensed...
    end

    # For condensed systems, the rhs layout is different.
    # With AbstractCondensedKKTSystem, the KKT vector has:
    #   primal: [x (n), s (n_ineq)]
    #   dual: [y (m)]
    #   dual_lb, dual_ub
    # After reduce_rhs!, bound duals are folded into primal.
    #
    # The condensed solve needs to:
    # 1. Fold inequality info: wx += J' * diag(db) * (wy_ineq + ws / Σs)
    # 2. Solve the condensed system for Δx (and Δy_eq if n_eq > 0)
    # 3. Recover: wy = J * Δx (with corrections), ws = (ws + wy_ineq) / Σs

    # Step 1: condense inequality contributions
    fill!(kkt.buffer, zero(T))
    if kkt.n_ineq > 0
        kkt.buffer[kkt.ind_ineq] .= kkt.diag_buffer .* (wy[kkt.ind_ineq] .+ ws ./ Sigma_s)
        mul!(wx, kkt.jac', kkt.buffer, one(T), one(T))
    end

    # Step 2: Extract per-scenario RHS blocks
    for k in 1:ns
        vr = (k-1)*nv+1 : k*nv
        rhs = kkt.rhs_k[k]
        # Primal part
        @inbounds for i in 1:nv
            rhs[i] = wx[vr[1]+i-1]
        end
        # Equality dual part
        for (ci, gi) in enumerate(kkt.eq_per_scenario[k])
            rhs[nv+ci] = wy[gi]
        end
    end
    # Design RHS
    @inbounds for i in 1:nd
        kkt.rhs_d[i] = wx[ns*nv+i]
    end

    # Step 3: Forward elimination
    for k in 1:ns
        # Solve A_kk * z_k = rhs_k
        LAPACK.sytrs!('L', kkt.A_kk_work[k], kkt.A_kk_ipiv[k], kkt.rhs_k[k])
        # rhs_d -= C_dk * z_k
        mul!(kkt.rhs_d, kkt.C_dk[k], kkt.rhs_k[k], -one(T), one(T))
    end

    # Step 4: Solve Schur complement
    solve_linear_system!(kkt.linear_solver, kkt.rhs_d)

    # Step 5: Back-substitution
    # z_k was A_kk^{-1} * rhs_k_orig
    # Δ_k = z_k - A_kk^{-1} * C_dk' * Δd = rhs_k - tmp_blk_nd * Δd
    for k in 1:ns
        mul!(kkt.rhs_k[k], kkt.tmp_blk_nd[k], kkt.rhs_d, -one(T), one(T))
    end

    # Step 6: Write back to w
    for k in 1:ns
        vr = (k-1)*nv+1 : k*nv
        rhs = kkt.rhs_k[k]
        # Primal
        @inbounds for i in 1:nv
            wx[vr[1]+i-1] = rhs[i]
        end
        # Equality duals
        for (ci, gi) in enumerate(kkt.eq_per_scenario[k])
            wy[gi] = rhs[nv+ci]
        end
    end
    @inbounds for i in 1:nd
        wx[ns*nv+i] = kkt.rhs_d[i]
    end

    # Step 7: Recover inequality duals and slacks
    if kkt.n_ineq > 0
        # wy_ineq = diag(db) * (J * Δx) - buffer
        # Actually, for condensed form: recover wy[ineq] from J * Δx
        # Following DenseCondensed pattern:
        # mul!(dual(w), kkt.jac, wx) replaces all of wy
        # But we already have equality duals set correctly, so we need to be careful.
        # Save equality duals
        eq_duals_backup = [wy[gi] for k in 1:ns for (_, gi) in enumerate(kkt.eq_per_scenario[k])]

        mul!(wy, kkt.jac, wx)

        # Restore equality duals
        idx = 1
        for k in 1:ns
            for (_, gi) in enumerate(kkt.eq_per_scenario[k])
                wy[gi] = eq_duals_backup[idx]
                idx += 1
            end
        end

        # For inequality constraints (following DenseCondensed pattern):
        # wz = diag(db) * (J * Δx) - buffer
        @inbounds for idx in 1:length(kkt.ind_ineq)
            gi = kkt.ind_ineq[idx]
            wy[gi] = kkt.diag_buffer[idx] * wy[gi] - kkt.buffer[gi]
        end
        ws .= (ws .+ view(wy, kkt.ind_ineq)) ./ Sigma_s
    else
        # No inequality: equality duals are already set from the Schur complement solve.
        # No slacks to recover.
    end

    finish_aug_solve!(kkt, w)
    return w
end

# KKT matrix-vector product for iterative refinement (matches AbstractDenseKKTSystem pattern)
function mul!(w::AbstractKKTVector{T}, kkt::SchurComplementKKTSystem{T}, x::AbstractKKTVector, alpha = one(T), beta = zero(T)) where T
    (m, n) = size(kkt.jac)
    wx = @view(primal(w)[1:n])
    ws = @view(primal(w)[n+1:end])
    wy = dual(w)
    wz = @view(dual(w)[kkt.ind_ineq])

    xx = @view(primal(x)[1:n])
    xs = @view(primal(x)[n+1:end])
    xy = dual(x)
    xz = @view(dual(x)[kkt.ind_ineq])

    _symv!('L', alpha, kkt.hess, xx, beta, wx)
    if m > 0
        mul!(wx, kkt.jac', dual(x), alpha, one(T))
        mul!(wy, kkt.jac,  xx, alpha, beta)
    end
    ws .= beta.*ws .- alpha.* xz
    wz .-= alpha.* xs
    _kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

function mul_hess_blk!(wx, kkt::SchurComplementKKTSystem, t)
    n = size(kkt.hess, 1)
    mul!(@view(wx[1:n]), Symmetric(kkt.hess, :L), @view(t[1:n]))
    fill!(@view(wx[n+1:end]), 0)
    wx .+= t .* kkt.pr_diag
end
